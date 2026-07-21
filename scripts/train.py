from __future__ import annotations

import argparse
import json
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
from phyrd.data import DiffCastH5Dataset, SEVIRDataset
from phyrd.models import build_composite_from_config, checkpoint_backbone_spec
from phyrd.models.composer import ForecastComposer
from phyrd.motion import build_motion_fields
from phyrd.physics import weak_transport_loss
from phyrd.train import CheckpointManager, build_experiment_directory
from phyrd.utils import seed_everything, write_json


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
        return SEVIRDataset(data_config["root"], split, **dataset_kwargs)
    if data_format == "diffcast_h5":
        return DiffCastH5Dataset(data_config["root"], split, **dataset_kwargs)
    raise ValueError("data.format must be 'catalog' or 'diffcast_h5'")


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
        with autocast_context(device, precision):
            if stage == "deterministic":
                prediction = train_model(history, stage="deterministic")
                batch_loss = torch.nn.functional.l1_loss(prediction, target)
            else:
                result = train_model(history, target, stage="residual")
                batch_loss = result["loss_gen"]
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
    if stage == "residual":
        payload["diffusion"] = model.diffusion.state_dict()
        payload["protocol"]["diffusion"] = model.diffusion_config
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="PhyRD registered training/pilot runner")
    parser.add_argument("--config", default="configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
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
    if stage not in {"deterministic", "residual"}:
        raise ValueError("stage must be 'deterministic' or 'residual'")
    model = build_composite_from_config(
        config,
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
    ).to(device)
    deterministic_checkpoint = model_config.get("deterministic_checkpoint")
    uses_backbone_pool = hasattr(model.deterministic, "select_for_step")
    if stage == "residual":
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
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "deterministic" not in payload:
                raise KeyError("checkpoint does not contain a deterministic state dict")
            checkpoint_protocol = payload.get("protocol")
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
            if (
                checkpoint_data_protocol != expected_data_protocol
                or checkpoint_deterministic != dict(model_config["deterministic"])
            ):
                raise ValueError(
                    "deterministic checkpoint protocol mismatch: "
                    f"checkpoint={checkpoint_protocol}, "
                    f"current_data={expected_data_protocol}, "
                    f"current_deterministic={model_config['deterministic']}"
                )
            model.deterministic.load_state_dict(payload["deterministic"])
            model.deterministic.requires_grad_(False)
            model.deterministic.eval()
            model.diffusion.requires_grad_(True)
            model.freeze_deterministic = True
    else:
        model.deterministic.requires_grad_(True)
        model.diffusion.requires_grad_(False)
        model.freeze_deterministic = False
    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        # Only the residual stage needs independent diffusion noise on each rank.
        if stage == "residual":
            torch.manual_seed(seed + rank)
    optimization = config["optimization"]
    parameters = [
        parameter
        for parameter in (
            model.deterministic.parameters() if stage == "deterministic" else model.diffusion.parameters()
        )
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization.get("betas", (0.9, 0.999))),
        weight_decay=float(optimization["weight_decay"]),
    )
    if args.max_steps is not None:
        max_steps = int(args.max_steps)
    elif optimization.get("max_epochs") is not None:
        max_steps = len(loader) * int(optimization["max_epochs"])
    else:
        max_steps = int(optimization["max_steps"])
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
    checkpoint_manager = CheckpointManager(artifact_dir, allow_existing=allow_existing)
    if is_main and not allow_existing:
        checkpoint_manager.write_config_snapshot(config)
    if world_size > 1:
        dist.barrier()
    history_log: list[dict[str, float | int | str]] = []
    step = 0
    epoch = 0
    last_val_loss: float | None = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    try:
        while step < max_steps:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                history = batch["x"].to(device, non_blocking=True)
                target = batch["y"].to(device, non_blocking=True)
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
                    else:
                        result = train_model(history, target, stage=stage)
                        loss_gen = result["loss_gen"]
                        total = loss_gen
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
                    }
                    if active_backbone is not None:
                        record["backbone"] = active_backbone
                    if stage == "deterministic":
                        for metric_name, metric_value in result.items():
                            if metric_name.startswith("loss_") and metric_name != "loss_gen":
                                record[metric_name] = distributed_mean(metric_value, world_size)
                    if is_main:
                        history_log.append(record)
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
                    "deterministic": dict(model_config["deterministic"]),
                    "best_val_loss": (
                        None
                        if checkpoint_manager.best_val_loss == float("inf")
                        else checkpoint_manager.best_val_loss
                    ),
                    "last_val_loss": last_val_loss,
                },
            )
    finally:
        dataset.close()
        if validation_dataset is not None:
            validation_dataset.close()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
