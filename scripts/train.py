from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from phyrd.config import load_config
from phyrd.data import CachedTrendDataset, DiffCastH5Dataset, SEVIRDataset
from phyrd.evaluation.results_registry import append_full_test_result
from phyrd.models import build_composite_from_config, checkpoint_backbone_spec
from phyrd.models.composer import ForecastComposer
from phyrd.motion import build_motion_fields
from phyrd.physics import weak_transport_loss
from phyrd.train import (
    CheckpointManager,
    build_experiment_directory,
    learning_rate_multiplier,
)
from phyrd.utils import seed_everything, write_json
try:
    from scripts.evaluation.post_training import evaluate_post_training
except ModuleNotFoundError:
    # ``python scripts/train.py`` places ``scripts/`` rather than the repository
    # root on sys.path.
    from evaluation.post_training import evaluate_post_training


def setup_runtime(config_device: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return torch.device("cuda", local_rank), rank, local_rank, world_size
    return torch.device(config_device), rank, local_rank, world_size


def distributed_mean(value: torch.Tensor, world_size: int) -> float:
    reduced = value.detach().float().clone()
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= world_size
    return float(reduced.item())


def autocast_context(device: torch.device, precision: str) -> Any:
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("optimization.precision must be 'fp32', 'fp16', or 'bf16'")


def build_dataset(data_config: dict[str, Any], *, split: str, max_samples: int | None):
    dataset_kwargs = {
        "input_frames": int(data_config.get("input_frames", 13)),
        "output_frames": int(data_config.get("output_frames", 12)),
        "window_start_index": int(data_config.get("window_start_index", 12)),
        "model_resolution": int(data_config.get("model_resolution", 384)),
        "spatial_preprocess": str(data_config.get("spatial_preprocess", "none")),
        "max_samples": max_samples,
    }
    data_format = str(data_config.get("format", "catalog"))
    if data_format == "catalog":
        dataset = SEVIRDataset(data_config["root"], split, **dataset_kwargs)
    elif data_format == "diffcast_h5":
        dataset = DiffCastH5Dataset(data_config["root"], split, **dataset_kwargs)
    else:
        raise ValueError("data.format must be 'catalog' or 'diffcast_h5'")
    cache_dir = data_config.get("trend_cache_dir")
    if cache_dir:
        dataset = CachedTrendDataset(dataset, Path(str(cache_dir)) / f"{split}.npy")
    return dataset


def build_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    rank: int,
    world_size: int,
    seed: int,
    training: bool,
):
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            seed=seed,
            drop_last=training,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=training and world_size > 1,
    )
    return loader, sampler


@torch.no_grad()
def validate(
    train_model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    precision: str,
    stage: str,
    world_size: int,
    max_batches: int | None,
) -> float:
    train_model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        history = batch["x"].to(device, non_blocking=True)
        target = batch["y"].to(device, non_blocking=True)
        cached_trend = batch.get("trend")
        if cached_trend is not None:
            cached_trend = cached_trend.to(device, non_blocking=True)
        with autocast_context(device, precision):
            if stage == "deterministic":
                prediction = train_model(history, stage="deterministic")
                batch_loss = torch.nn.functional.l1_loss(prediction, target)
            elif stage == "residual":
                result = train_model(history, target, stage="residual", trend=cached_trend)
                batch_loss = result["loss_gen"]
            else:
                result = train_model(history, target, stage="joint_residual")
                joint = getattr(train_model, "module", train_model)
                weights = getattr(joint, "_joint_loss_weights", (0.5, 0.5))
                batch_loss = weights[0] * result["loss_det"] + weights[1] * result["loss_diff"]
        batch_size = history.shape[0]
        loss_sum += batch_loss.detach().double() * batch_size
        sample_count += batch_size
    if world_size > 1:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_count, op=dist.ReduceOp.SUM)
    train_model.train()
    if stage == "residual":
        unwrapped = (
            train_model.module
            if isinstance(train_model, DistributedDataParallel)
            else train_model
        )
        unwrapped.deterministic.eval()
    if sample_count.item() == 0:
        raise RuntimeError("validation loader produced no samples")
    return float((loss_sum / sample_count).item())


def checkpoint_payload(
    *,
    stage: str,
    model: ForecastComposer,
    optimizer: torch.optim.Optimizer,
    dataset,
    data_config: dict[str, Any],
    world_size: int,
    precision: str,
    step: int,
    epoch: int,
    val_loss: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
        "protocol": {
            "input_frames": dataset.input_frames,
            "output_frames": dataset.output_frames,
            "native_resolution": dataset.native_resolution,
            "model_resolution": dataset.model_resolution,
            "spatial_preprocess": dataset.spatial_preprocess,
            "deterministic": {
                "name": model.deterministic_name,
                "params": model.deterministic_params,
            },
        },
        "distributed": {
            "world_size": world_size,
            "per_rank_batch_size": int(data_config["batch_size"]),
            "global_batch_size": int(data_config["batch_size"]) * world_size,
            "precision": precision,
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    pool_specs = getattr(model.deterministic, "member_specs", None)
    if pool_specs is None:
        payload["deterministic"] = model.deterministic.state_dict()
    else:
        payload["deterministic_pool"] = pool_specs
        payload["active_backbone"] = getattr(model.deterministic, "active_name", None)
    if stage in {"residual", "joint_residual"}:
        payload["diffusion"] = model.diffusion.state_dict()
        payload["protocol"]["diffusion"] = model.diffusion_config
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="PhyRD registered training/pilot runner")
    parser.add_argument("--config", default="configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="resume from a PhyRD checkpoint_last.pt (restores model, optimizer, step, and epoch)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    resume_path = Path(args.resume).expanduser().resolve() if args.resume else None
    if resume_path is not None and not resume_path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    # Reuse the original run directory when resuming, so checkpoints and logs
    # continue in-place instead of creating a new timestamped experiment.
    if resume_path is not None:
        resume_run_dir = resume_path.parent.parent
        config["artifacts"] = {
            **dict(config.get("artifacts", {})),
            "directory": str(resume_run_dir),
            "allow_existing": True,
        }
    seed = int(config["seed"])
    seed_everything(seed)
    device, rank, local_rank, world_size = setup_runtime(config.get("device", "cuda:0"))
    is_main = rank == 0
    data_config = dict(config["data"])
    if args.data_root is not None:
        data_config["root"] = args.data_root
    data_format = str(data_config.get("format", "catalog"))
    dataset = build_dataset(
        data_config,
        split=str(data_config["split"]),
        max_samples=data_config.get("max_samples"),
    )
    num_workers = int(data_config["num_workers"])
    loader, sampler = build_loader(
        dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=num_workers,
        device=device,
        rank=rank,
        world_size=world_size,
        seed=seed,
        training=True,
    )
    validation_config = dict(config.get("validation", {}))
    validation_enabled = bool(validation_config.get("enabled", False))
    validation_dataset = None
    validation_loader = None
    if validation_enabled:
        default_validation_split = "valid" if data_format == "diffcast_h5" else "val_model"
        validation_dataset = build_dataset(
            data_config,
            split=str(validation_config.get("split", default_validation_split)),
            max_samples=validation_config.get("max_samples"),
        )
        validation_loader, _ = build_loader(
            validation_dataset,
            batch_size=int(validation_config.get("batch_size", data_config["batch_size"])),
            num_workers=int(validation_config.get("num_workers", num_workers)),
            device=device,
            rank=rank,
            world_size=world_size,
            seed=seed,
            training=False,
        )
    model_config = config["model"]
    stage = str(config.get("stage", "deterministic"))
    if stage not in {"deterministic", "residual", "joint_residual"}:
        raise ValueError(
            "stage must be 'deterministic', 'residual', or 'joint_residual'"
        )
    model = build_composite_from_config(
        config,
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
    ).to(device)
    deterministic_checkpoint = model_config.get("deterministic_checkpoint")
    uses_backbone_pool = hasattr(model.deterministic, "select_for_step")
    if stage in {"residual", "joint_residual"}:
        if stage == "joint_residual" and uses_backbone_pool:
            raise ValueError("joint_residual requires one trainable deterministic backbone")
        if uses_backbone_pool:
            model.deterministic.requires_grad_(False)
            model.deterministic.eval()
            model.diffusion.requires_grad_(True)
            model.freeze_deterministic = True
        elif not deterministic_checkpoint:
            raise ValueError("residual stage requires model.deterministic_checkpoint")
        else:
            checkpoint_path = Path(deterministic_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"deterministic checkpoint not found: {checkpoint_path}")
            external_loader = getattr(model.deterministic, "load_external_checkpoint", None)
            if callable(external_loader):
                external_loader(checkpoint_path)
                payload = None
            else:
                payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload is not None and "deterministic" not in payload:
                raise KeyError("checkpoint does not contain a deterministic state dict")
            checkpoint_protocol = payload.get("protocol") if payload is not None else None
            expected_data_protocol = {
                "input_frames": dataset.input_frames,
                "output_frames": dataset.output_frames,
                "native_resolution": dataset.native_resolution,
                "model_resolution": dataset.model_resolution,
                "spatial_preprocess": dataset.spatial_preprocess,
            }
            checkpoint_data_protocol = {
                key: checkpoint_protocol.get(key) for key in expected_data_protocol
            } if isinstance(checkpoint_protocol, dict) else None
            checkpoint_deterministic = (
                checkpoint_backbone_spec(checkpoint_protocol)
                if isinstance(checkpoint_protocol, dict)
                else None
            )
            if payload is not None and (
                checkpoint_data_protocol != expected_data_protocol
                or checkpoint_deterministic != dict(model_config["deterministic"])
            ):
                raise ValueError(
                    "deterministic checkpoint protocol mismatch: "
                    f"checkpoint={checkpoint_protocol}, "
                    f"current_data={expected_data_protocol}, "
                    f"current_deterministic={model_config['deterministic']}"
                )
            if payload is not None:
                model.deterministic.load_state_dict(payload["deterministic"])
            model.deterministic.requires_grad_(stage == "joint_residual")
            if stage == "residual":
                model.deterministic.eval()
            model.diffusion.requires_grad_(True)
            model.freeze_deterministic = stage == "residual"
    else:
        model.deterministic.requires_grad_(True)
        model.diffusion.requires_grad_(False)
        model.freeze_deterministic = False
    resume_payload = None
    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if str(resume_payload.get("stage", stage)) != stage:
            raise ValueError(
                f"resume stage mismatch: checkpoint={resume_payload.get('stage')}, current={stage}"
            )
        if "deterministic" in resume_payload:
            model.deterministic.load_state_dict(resume_payload["deterministic"], strict=True)
        if stage in {"residual", "joint_residual"} and "diffusion" in resume_payload:
            model.diffusion.load_state_dict(resume_payload["diffusion"], strict=True)
    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=stage == "joint_residual",
        )
        # Only the residual stage needs independent diffusion noise on each rank.
        if stage in {"residual", "joint_residual"}:
            torch.manual_seed(seed + rank)
    optimization = config["optimization"]
    joint_config = dict(config.get("joint_training", {}))
    det_weight = float(joint_config.get("deterministic_weight", 0.5))
    diff_weight = float(joint_config.get("diffusion_weight", 0.5))
    if stage == "joint_residual" and not math.isclose(
        det_weight + diff_weight, 1.0, abs_tol=1e-6
    ):
        raise ValueError("joint_training loss weights must sum to one")
    model._joint_loss_weights = (det_weight, diff_weight)
    if stage == "joint_residual":
        optimizer_groups = [
            {
                "params": list(model.deterministic.parameters()),
                "lr": float(joint_config.get("deterministic_learning_rate", 1e-5)),
                "name": "deterministic",
            },
            {
                "params": list(model.diffusion.parameters()),
                "lr": float(
                    joint_config.get(
                        "diffusion_learning_rate", optimization["learning_rate"]
                    )
                ),
                "name": "diffusion",
            },
        ]
        parameters = [
            parameter for group in optimizer_groups for parameter in group["params"]
        ]
    else:
        parameters = [
            parameter
            for parameter in (
                model.deterministic.parameters()
                if stage == "deterministic"
                else model.diffusion.parameters()
            )
            if parameter.requires_grad
        ]
        optimizer_groups = parameters
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization.get("betas", (0.9, 0.999))),
        weight_decay=float(optimization["weight_decay"]),
    )
    if resume_payload is not None and "optimizer" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer"])
    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    elif optimization.get("max_epochs") is not None:
        max_steps = len(loader) * int(optimization["max_epochs"])
    else:
        max_steps = int(optimization["max_steps"])
    # The schedule is defined relative to the configured base LR.  Loading an
    # optimizer checkpoint restores its already-decayed LR, so using that value
    # as the new base would decay it a second time after every resumed step.
    if stage == "joint_residual":
        initial_group_lrs = [
            float(joint_config.get("deterministic_learning_rate", 1e-5)),
            float(joint_config.get("diffusion_learning_rate", optimization["learning_rate"])),
        ]
    else:
        initial_group_lrs = [float(optimization["learning_rate"])]
    if resume_payload is not None:
        for group, base_lr in zip(optimizer.param_groups, initial_group_lrs, strict=True):
            group["lr"] = base_lr
    precision = str(optimization.get("precision", "fp32"))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision == "fp16")
    artifacts_config = dict(config["artifacts"])
    allow_existing = bool(artifacts_config.get("allow_existing", False))
    configured_directory = artifacts_config.get("directory")
    if configured_directory:
        artifact_dir = Path(str(configured_directory))
    else:
        generated_directory = None
        if is_main:
            generated_directory = str(
                build_experiment_directory(
                    artifacts_config.get("root", "artifacts/experiments"),
                    str(artifacts_config.get("deterministic_name", model.deterministic_name)),
                    str(
                        artifacts_config.get(
                            "probabilistic_name",
                            model_config.get("probabilistic", {}).get("name", "residual_diffusion"),
                        )
                    ),
                )
            )
        if world_size > 1:
            generated_paths = [generated_directory]
            dist.broadcast_object_list(generated_paths, src=0)
            generated_directory = generated_paths[0]
        if not generated_directory:
            raise RuntimeError("failed to create an experiment directory")
        artifact_dir = Path(generated_directory)
    config["artifacts"] = {**artifacts_config, "directory": str(artifact_dir)}
    # Rank 0 owns run-output creation and the non-overwrite check.  Other DDP
    # ranks must attach to the same directory after the broadcast; otherwise
    # they race with rank 0's config snapshot and mistake a fresh run for an
    # existing artifact.
    checkpoint_manager = CheckpointManager(
        artifact_dir,
        allow_existing=allow_existing or not is_main,
    )
    if is_main and not allow_existing:
        checkpoint_manager.write_config_snapshot(config)
    if world_size > 1:
        dist.barrier()
    history_log: list[dict[str, float | int | str]] = []
    train_jsonl_path = checkpoint_manager.metrics_directory / "train_log.jsonl"
    if resume_payload is not None:
        existing_history = checkpoint_manager.metrics_directory / "train_log.json"
        if existing_history.is_file():
            try:
                loaded_history = json.loads(existing_history.read_text(encoding="utf-8"))
                if isinstance(loaded_history, list):
                    history_log = loaded_history
            except (OSError, json.JSONDecodeError):
                pass
        step = int(resume_payload.get("step", 0))
        epoch = int(resume_payload.get("epoch", 0))
        last_val_loss = resume_payload.get("val_loss")
    else:
        step = 0
        epoch = 0
        last_val_loss = None
    if resume_path is not None:
        best_path = checkpoint_manager.checkpoint_directory / "checkpoint_best.pt"
        if best_path.is_file():
            try:
                best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
                best_loss = best_payload.get("val_loss")
                if best_loss is not None:
                    checkpoint_manager.best_val_loss = float(best_loss)
            except (OSError, RuntimeError, EOFError):
                pass
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    try:
        while step < max_steps:
            if stage == "joint_residual":
                warmup_epochs = int(joint_config.get("warmup_epochs", 5))
                deterministic_trainable = epoch >= warmup_epochs
                model.deterministic.requires_grad_(deterministic_trainable)
                model.deterministic.train(deterministic_trainable)
                model.freeze_deterministic = not deterministic_trainable
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                lr_multiplier = learning_rate_multiplier(
                    optimization,
                    step=step,
                    max_steps=max_steps,
                    steps_per_epoch=len(loader),
                )
                for group, initial_lr in zip(
                    optimizer.param_groups,
                    initial_group_lrs,
                    strict=True,
                ):
                    group["lr"] = initial_lr * lr_multiplier
                history = batch["x"].to(device, non_blocking=True)
                target = batch["y"].to(device, non_blocking=True)
                cached_trend = batch.get("trend")
                if cached_trend is not None:
                    cached_trend = cached_trend.to(device, non_blocking=True)
                active_backbone = (
                    model.select_backbone_for_step(step, seed)
                    if stage == "residual" and uses_backbone_pool
                    else None
                )
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(device, precision):
                    if stage == "deterministic":
                        result = train_model(history, target, stage=stage)
                        loss_gen = result["loss_gen"]
                        physics_value = loss_gen.new_zeros(())
                        total = loss_gen
                    elif stage == "residual":
                        result = train_model(
                            history, target, stage=stage, trend=cached_trend
                        )
                        loss_gen = result["loss_gen"]
                        total = loss_gen
                        physics_value = total.new_zeros(())
                    else:
                        result = train_model(history, target, stage=stage)
                        loss_gen = result["loss_diff"]
                        loss_det = result["loss_det"]
                        total = det_weight * loss_det + diff_weight * loss_gen
                        physics_value = total.new_zeros(())
                if stage == "residual" and config["physics"]["enabled"]:
                    physics_timestep_max = int(
                        config["physics"].get(
                            "apply_below_timestep", model.diffusion.diffusion_steps - 1
                        )
                    )
                    physics_mask = result["timestep"] <= physics_timestep_max
                    physics_prediction = result["prediction_x0"][physics_mask]
                    physics_history = history[physics_mask]
                    if physics_prediction.shape[0] == 0:
                        physics_value = total.new_zeros(())
                    else:
                        fields = build_motion_fields(
                            physics_history.float(), output_frames=dataset.output_frames
                        )
                        physics_value, _ = weak_transport_loss(
                            physics_prediction.float(),
                            fields.flow.detach(),
                            fields.c_flow.detach(),
                            fields.m_nadv.detach(),
                            robust_scale=config["physics"]["robust_scale"],
                            tolerance=config["physics"]["tolerance"],
                            gamma_nadv=config["physics"]["gamma_nadv"],
                            pool_sizes=tuple(config["physics"]["pool_sizes"]),
                            alpha_mass=config["physics"]["alpha_mass"],
                        )
                    total = total + config["physics"]["lambda_train"] * physics_value
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                det_grad_norm = total.new_zeros(())
                diff_grad_norm = total.new_zeros(())
                if stage == "joint_residual":
                    det_grads = [
                        parameter.grad.detach().norm(2)
                        for parameter in model.deterministic.parameters()
                        if parameter.grad is not None
                    ]
                    diff_grads = [
                        parameter.grad.detach().norm(2)
                        for parameter in model.diffusion.parameters()
                        if parameter.grad is not None
                    ]
                    if det_grads:
                        det_grad_norm = torch.stack(det_grads).norm(2)
                    if diff_grads:
                        diff_grad_norm = torch.stack(diff_grads).norm(2)
                grad_clip = optimization.get("grad_clip")
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
                step += 1
                should_log = step % int(optimization["log_every"]) == 0 or step == 1
                if should_log:
                    record = {
                        "step": step,
                        "loss": distributed_mean(total, world_size),
                        "loss_gen": distributed_mean(loss_gen, world_size),
                        "loss_phys": distributed_mean(physics_value, world_size),
                        "lr": float(optimizer.param_groups[-1]["lr"]),
                    }
                    if active_backbone is not None:
                        record["backbone"] = active_backbone
                    if stage in {"deterministic", "residual"}:
                        for metric_name, metric_value in result.items():
                            if metric_name.startswith("loss_") and metric_name != "loss_gen":
                                record[metric_name] = distributed_mean(metric_value, world_size)
                    elif stage == "joint_residual":
                        record.update(
                            {
                                "loss_det": distributed_mean(result["loss_det"], world_size),
                                "loss_diff": distributed_mean(result["loss_diff"], world_size),
                                "grad_norm_det": distributed_mean(det_grad_norm, world_size),
                                "grad_norm_diff": distributed_mean(diff_grad_norm, world_size),
                                "lr_det": float(optimizer.param_groups[0]["lr"]),
                                "lr_diff": float(optimizer.param_groups[1]["lr"]),
                                "deterministic_trainable": bool(
                                    epoch >= int(joint_config.get("warmup_epochs", 5))
                                ),
                            }
                        )
                        # Joint residual architectures can expose different
                        # diagnostics. Log scalar losses generically and retain
                        # TrajRes-only fields when that model provides them.
                        for metric_name, metric_value in result.items():
                            if (
                                metric_name.startswith("loss_")
                                and metric_name not in record
                                and torch.is_tensor(metric_value)
                                and metric_value.numel() == 1
                            ):
                                record[metric_name] = distributed_mean(
                                    metric_value, world_size
                                )
                        for metric_name in ("segment_index", "prefix_mode"):
                            if metric_name in result:
                                metric_value = result[metric_name]
                                record[metric_name] = int(
                                    metric_value.item()
                                    if torch.is_tensor(metric_value)
                                    else metric_value
                                )
                        if "residual_abs_mean" in result:
                            record["residual_abs_mean"] = distributed_mean(
                                result["residual_abs_mean"], world_size
                            )
                    if is_main:
                        history_log.append(record)
                        with train_jsonl_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, sort_keys=True) + "\n")
                            handle.flush()
                        print(json.dumps(record, sort_keys=True), flush=True)
                if step >= max_steps:
                    break
            epoch += 1
            validation_due = validation_enabled and (
                epoch % int(validation_config.get("every_epochs", 1)) == 0
                or step >= max_steps
            )
            if validation_due:
                assert validation_loader is not None
                if stage == "residual" and uses_backbone_pool:
                    validation_backbone = str(
                        validation_config.get("backbone", model.deterministic.names[0])
                    )
                    model.select_backbone(validation_backbone)
                last_val_loss = validate(
                    train_model,
                    validation_loader,
                    device=device,
                    precision=precision,
                    stage=stage,
                    world_size=world_size,
                    max_batches=validation_config.get("max_batches"),
                )
                if is_main:
                    history_log.append({"step": step, "epoch": epoch, "val_loss": last_val_loss})
                    print(
                        json.dumps(
                            {"step": step, "epoch": epoch, "val_loss": last_val_loss},
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            payload = checkpoint_payload(
                stage=stage,
                model=model,
                optimizer=optimizer,
                dataset=dataset,
                data_config=data_config,
                world_size=world_size,
                precision=precision,
                step=step,
                epoch=epoch,
                val_loss=last_val_loss,
            )
            if is_main:
                checkpoint_manager.save(payload, val_loss=last_val_loss)
                write_json(checkpoint_manager.metrics_directory / "train_log.json", history_log)
            if world_size > 1:
                dist.barrier()
        if world_size > 1:
            dist.barrier()
        elapsed_seconds = time.time() - started
        peak_memory_gib = 0.0
        if device.type == "cuda":
            peak_memory = torch.tensor(
                torch.cuda.max_memory_allocated(device) / (1024**3), device=device
            )
            if world_size > 1:
                dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
            peak_memory_gib = float(peak_memory.item())
        if is_main:
            write_json(
                checkpoint_manager.metrics_directory / "run_summary.json",
                {
                    "status": "completed",
                    "stage": stage,
                    "steps": step,
                    "epochs": epoch,
                    "seconds": elapsed_seconds,
                    "data_format": data_format,
                    "data_source": str(
                        dataset.path if isinstance(dataset, DiffCastH5Dataset) else dataset.paths.data_root
                    ),
                    "model_resolution": dataset.model_resolution,
                    "spatial_preprocess": dataset.spatial_preprocess,
                    "world_size": world_size,
                    "per_rank_batch_size": int(data_config["batch_size"]),
                    "global_batch_size": int(data_config["batch_size"]) * world_size,
                    "global_samples_per_second": (
                        step * int(data_config["batch_size"]) * world_size / elapsed_seconds
                    ),
                    "max_rank_peak_memory_gib": peak_memory_gib,
                    "precision": precision,
                    "seed": seed,
                    # Use the resolved composer protocol rather than indexing the
                    # legacy single-backbone config.  A universal run can contain
                    # ``model.deterministic_pool`` instead of
                    # ``model.deterministic``.
                    "deterministic": {
                        "name": model.deterministic_name,
                        "params": model.deterministic_params,
                    },
                    "best_val_loss": (
                        None
                        if checkpoint_manager.best_val_loss == float("inf")
                        else checkpoint_manager.best_val_loss
                    ),
                    "last_val_loss": last_val_loss,
                },
            )
        post_test_config = dict(config.get("post_training_test", {}))
        # ``--max-steps`` is the explicit smoke/debug override.  It must never
        # consume a full report split or enter the formal result registry.
        if bool(post_test_config.get("enabled", False)) and args.max_steps is None:
            if not validation_enabled:
                raise ValueError(
                    "post_training_test requires validation so checkpoint_best.pt exists"
                )
            if world_size > 1:
                dist.barrier()
            best_checkpoint = (
                checkpoint_manager.checkpoint_directory / "checkpoint_best.pt"
            )
            if not best_checkpoint.is_file():
                raise FileNotFoundError(
                    f"best checkpoint was not created: {best_checkpoint}"
                )
            best_payload = torch.load(
                best_checkpoint, map_location="cpu", weights_only=False
            )
            if "deterministic" in best_payload:
                model.deterministic.load_state_dict(
                    best_payload["deterministic"], strict=True
                )
            if "diffusion" in best_payload:
                model.diffusion.load_state_dict(best_payload["diffusion"], strict=True)
            else:
                raise KeyError("best checkpoint does not contain diffusion weights")
            test_split = str(post_test_config.get("split", "report_test"))
            test_dataset = build_dataset(
                data_config,
                split=test_split,
                # A post-training test is always the complete registered split.
                max_samples=None,
            )
            try:
                metrics = evaluate_post_training(
                    model,
                    test_dataset,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    batch_size=int(
                        post_test_config.get("batch_size", data_config["batch_size"])
                    ),
                    num_workers=int(
                        post_test_config.get("num_workers", num_workers)
                    ),
                    ensemble_size=int(
                        post_test_config.get("ensemble_size", 10)
                    ),
                    sampling_steps=int(
                        post_test_config.get("sampling_steps", 20)
                    ),
                    checkpoint_epoch=int(best_payload.get("epoch", -1)),
                    checkpoint_path=str(best_checkpoint.resolve()),
                    split=test_split,
                )
                if is_main:
                    assert metrics is not None
                    metrics["protocol"] = (
                        f"{dataset.input_frames}to{dataset.output_frames}"
                        f"@{dataset.model_resolution}"
                    )
                    metrics_path = (
                        checkpoint_manager.metrics_directory
                        / f"{test_split}_best_full.json"
                    )
                    write_json(metrics_path, metrics)
                    if bool(post_test_config.get("register_result", True)):
                        probabilistic_name = str(
                            model_config.get("probabilistic", {}).get(
                                "name", "residual_diffusion"
                            )
                        )
                        result_id = append_full_test_result(
                            post_test_config.get(
                                "registry_path", "RESULTS_REGISTRY.csv"
                            ),
                            metrics=metrics,
                            experiment=str(
                                post_test_config.get(
                                    "experiment_name",
                                    f"{model.deterministic_name}+{probabilistic_name}",
                                )
                            ),
                            deterministic_backbone=model.deterministic_name,
                            probabilistic_module=probabilistic_name,
                            evidence=str(metrics_path.resolve()),
                        )
                        metrics["registry_result_id"] = result_id
                        write_json(metrics_path, metrics)
                    print(json.dumps({"post_training_test": metrics}, sort_keys=True))
            finally:
                test_dataset.close()
            if world_size > 1:
                dist.barrier()
    finally:
        dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
