"""Shared, model-agnostic evaluation for the registered SEVIR protocols.

The evaluator deliberately owns the metric implementation.  Model adapters only
load a checkpoint and return ``[B, T_out, 1, H, W]`` predictions.  This prevents
model-specific test scripts from silently changing CSI aggregation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

THRESHOLDS = (16.0, 74.0, 133.0, 160.0, 181.0, 219.0)
POOL_SIZES = (1, 4, 16)


class RankStridedSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length, self.rank, self.world_size = int(length), int(rank), int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        return max(0, (self.length - 1 - self.rank) // self.world_size + 1)


def distributed_context(device_arg: str) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        return True, dist.get_rank(), world_size, torch.device("cuda", local_rank)
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return False, 0, 1, torch.device("cpu")
    return False, 0, 1, torch.device(device_arg)


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _pool(sequence: torch.Tensor, size: int) -> torch.Tensor:
    if size == 1:
        return sequence
    b, t, c, h, w = sequence.shape
    if h % size or w % size:
        raise ValueError(f"pool={size} does not divide spatial shape {(h, w)}")
    pooled = F.max_pool2d(sequence.reshape(b * t, c, h, w), size, stride=size)
    return pooled.reshape(b, t, c, h // size, w // size)


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return torch.where(den > 0, num / den, torch.zeros_like(num))


def contingency(prediction_vil: torch.Tensor, target_vil: torch.Tensor) -> torch.Tensor:
    """Return [pool, threshold, hit/miss/falarm/correct-negative] global counts."""
    rows = []
    for pool in POOL_SIZES:
        pred, target = _pool(prediction_vil, pool), _pool(target_vil, pool)
        threshold_rows = []
        for threshold in THRESHOLDS:
            p, y = pred >= threshold, target >= threshold
            threshold_rows.append(
                torch.stack(
                    (
                        (p & y).sum(dtype=torch.float64),
                        ((~p) & y).sum(dtype=torch.float64),
                        (p & (~y)).sum(dtype=torch.float64),
                        ((~p) & (~y)).sum(dtype=torch.float64),
                    )
                )
            )
        rows.append(torch.stack(threshold_rows))
    return torch.stack(rows)


def categorical_metrics(counts: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for pool_index, pool in enumerate(POOL_SIZES):
        hits, misses, false_alarms, correct_negatives = counts[pool_index].unbind(dim=1)
        csi = _safe_ratio(hits, hits + misses + false_alarms)
        name = "CSI" if pool == 1 else f"CSI_pool{pool}"
        metrics[name] = float(csi.mean().item())
        for threshold, value in zip(THRESHOLDS, csi):
            metrics[f"{name}_{int(threshold)}"] = float(value.item())
        if pool == 1:
            numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
            denominator = (hits + misses) * (misses + correct_negatives) + (
                hits + false_alarms
            ) * (false_alarms + correct_negatives)
            hss = _safe_ratio(numerator, denominator)
            metrics["HSS"] = float(hss.mean().item())
            for threshold, value in zip(THRESHOLDS, hss):
                metrics[f"HSS_{int(threshold)}"] = float(value.item())
    metrics["CSI_global"] = metrics["CSI"]
    return metrics


def _continuous_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    diff = pred - target
    return (
        diff.abs().sum(dtype=torch.float64),
        diff.square().sum(dtype=torch.float64),
        torch.tensor(float(diff.numel()), dtype=torch.float64, device=pred.device),
        torch.tensor(float(pred.shape[0]), dtype=torch.float64, device=pred.device),
    )


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """A dependency-free frame-wise SSIM mean in [0,1]."""
    b, t, c, h, w = pred.shape
    x, y = pred.reshape(b * t, c, h, w), target.reshape(b * t, c, h, w)
    kernel_1d = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel_1d = torch.exp(-(kernel_1d.square()) / (2 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    window = (kernel_1d[:, None] * kernel_1d[None, :]).reshape(1, 1, 11, 11)
    window = window.repeat(c, 1, 1, 1)
    conv = lambda z: F.conv2d(z, window, padding=5, groups=c)
    mux, muy = conv(x), conv(y)
    vx, vy = conv(x.square()) - mux.square(), conv(y.square()) - muy.square()
    cov = conv(x * y) - mux * muy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mux * muy + c1) * (2 * cov + c2)) / (
        (mux.square() + muy.square() + c1) * (vx + vy + c2).clamp_min(1e-8)
    )
    return score.mean()


_LPIPS_MODELS: dict[str, Any] = {}


def _lpips(pred: torch.Tensor, target: torch.Tensor, chunk: int) -> torch.Tensor:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS is required; install it or pass --skip-lpips") from exc
    key = str(pred.device)
    if key not in _LPIPS_MODELS:
        _LPIPS_MODELS[key] = lpips.LPIPS(net="alex", verbose=False).eval().to(pred.device)
        for p in _LPIPS_MODELS[key].parameters():
            p.requires_grad_(False)
    model = _LPIPS_MODELS[key]
    b, t, c, h, w = pred.shape
    x = pred.reshape(b * t, c, h, w).repeat(1, 3, 1, 1) * 2 - 1
    y = target.reshape(b * t, c, h, w).repeat(1, 3, 1, 1) * 2 - 1
    values = [model(x[i : i + chunk], y[i : i + chunk]) for i in range(0, x.shape[0], chunk)]
    return torch.cat(values).mean()


def _batch_xy(batch: Any, input_frames: int, output_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and batch[0].ndim == 5:
        return batch[0], batch[1]
    if isinstance(batch, torch.Tensor) and batch.ndim == 4:
        return batch[:input_frames], batch[input_frames : input_frames + output_frames]
    if not isinstance(batch, torch.Tensor) or batch.ndim != 5:
        raise TypeError(f"unsupported dataset batch type: {type(batch)!r}")
    return batch[:, :input_frames], batch[:, input_frames : input_frames + output_frames]


def _load_dataset(args: argparse.Namespace, protocol: str, model_name: str):
    if model_name == "phydnet":
        root = Path(args.phydnet_root).resolve()
        sys.path.insert(0, str(root))
        from data.sevir import SEVIRH5Dataset, SEVIRNowcastH5Dataset

        if protocol == "5to20":
            return SEVIRH5Dataset(
                args.data_path,
                split=args.split,
                img_size=args.img_size,
                seq_len=args.input_frames + args.output_frames,
                multi_channel=False,
                input_channels=1,
                dem_norm="minmax",
                lucc_as_index=False,
            )
        return SEVIRNowcastH5Dataset(args.data_path, img_size=args.img_size)
    root = Path(args.phyrd_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from phyrd.data import DiffCastH5Dataset, SEVIRDataset

    if protocol == "5to20":
        return DiffCastH5Dataset(
            args.data_path,
            args.split,
            input_frames=args.input_frames,
            output_frames=args.output_frames,
            window_start_index=0,
            model_resolution=args.img_size,
            spatial_preprocess=args.spatial_preprocess,
        )
    sevir_split = {
        "test": "report_test",
        "valid": "val_model",
    }.get(args.split, args.split)
    return SEVIRDataset(
        args.data_path,
        sevir_split,
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        window_start_index=12,
        model_resolution=args.img_size,
        spatial_preprocess=args.spatial_preprocess,
    )


def _load_model(args: argparse.Namespace, device: torch.device, protocol: str):
    if args.model == "phydnet":
        root = Path(args.phydnet_root).resolve()
        sys.path.insert(0, str(root))
        from models.phydnet_sevir import get_model

        model = get_model(
            in_shape=(1, args.img_size, args.img_size),
            T_in=args.input_frames,
            T_out=args.output_frames,
            device=device,
            lucc_embed_dim=0,
            lucc_mask=0,
        ).to(device)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(payload.get("model", payload), strict=True)
        return model.eval(), int(payload.get("epoch", -1))
    root = Path(args.phyrd_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from phyrd.config import load_config
    from phyrd.models import build_backbone

    config = load_config(args.config)
    deterministic = dict(config["model"]["deterministic"])
    model = build_backbone(
        str(deterministic["name"]),
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        params=dict(deterministic.get("params", {})),
    ).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("deterministic", payload.get("model", payload))
    model.load_state_dict(state, strict=True)
    return model.eval(), int(payload.get("epoch", payload.get("step", -1)))


def _predict(model: torch.nn.Module, history: torch.Tensor, model_name: str) -> torch.Tensor:
    if model_name == "phydnet":
        return model.inference(history)
    return model(history)


def _vil_rgb(values: np.ndarray) -> np.ndarray:
    colors = np.array(
        [[0, 0, 0], [77, 77, 77], [40, 190, 40], [25, 150, 25], [10, 105, 10],
         [10, 75, 10], [245, 245, 0], [237, 172, 0], [240, 110, 0], [160, 0, 0],
         [231, 0, 255]], dtype=np.uint8
    )
    bounds = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256])
    bins = np.digitize(np.clip(values, 0, 255), bounds[1:-1], right=False)
    return colors[np.rint(bins * 10 / 9).astype(np.int64)]


@torch.no_grad()
def save_visualization(
    model,
    dataset,
    index: int,
    path: Path,
    device: torch.device,
    input_frames: int,
    output_frames: int,
    model_name: str,
) -> None:
    sample = dataset[index]
    history, target = _batch_xy(sample, input_frames, output_frames)
    prediction = _predict(model, history.unsqueeze(0).to(device), model_name).clamp(0, 1)[0].cpu()
    history, target = history.cpu(), target.cpu()
    sequences = [(history[:, 0] * 255).numpy(), (prediction[:, 0] * 255).numpy(), (target[:, 0] * 255).numpy()]
    residual = sequences[1] - sequences[2]
    cell, label = 64, 110
    columns = max(input_frames, output_frames)
    canvas = Image.new("RGB", (label + columns * cell, 36 + 4 * (cell + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    names = ["History", "Prediction", "Ground Truth", "Pred - GT"]
    for row, seq in enumerate((*sequences, residual)):
        y = 36 + row * (cell + 20)
        draw.text((8, y + 24), names[row], fill="black")
        for col in range(min(columns, len(seq))):
            tile = seq[col]
            if row < 3:
                rgb = _vil_rgb(tile)
            else:
                normalized = np.clip(tile / max(16.0, float(np.abs(residual).max())), -1, 1)
                rgb = np.stack([255 * (1 + normalized.clip(max=0)), 255 * (1 - np.abs(normalized)), 255 * (1 - normalized.clip(min=0))], axis=-1).clip(0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb).resize((cell, cell), Image.Resampling.NEAREST), (label + col * cell, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run(protocol: str, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=f"Unified SEVIR evaluator ({protocol})")
    parser.add_argument("--model", choices=("phydnet", "sdir", "phyrd"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", help="PhyRD/SDIR YAML config")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split", default="test", choices=("train", "valid", "val_model", "test", "report_test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--visualization-dir")
    parser.add_argument("--visualization-samples", default="0,1,2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--input-frames", type=int, default=5 if protocol == "5to20" else 13)
    parser.add_argument("--output-frames", type=int, default=20 if protocol == "5to20" else 12)
    parser.add_argument("--spatial-preprocess", default="diffcast_bilinear")
    parser.add_argument("--phyrd-root", default="/test1/wzq/PhyRD")
    parser.add_argument("--phydnet-root", default="/test1/wzq/Weather/PhyDNet")
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-chunk-size", type=int, default=32)
    args = parser.parse_args(argv)
    if args.model in {"sdir", "phyrd"} and not args.config:
        parser.error("--config is required for --model sdir/phyrd")
    if protocol == "5to20" and (args.input_frames, args.output_frames) != (5, 20):
        parser.error("5to20 requires --input-frames 5 --output-frames 20")
    if protocol == "13to12" and (args.input_frames, args.output_frames) != (13, 12):
        parser.error("13to12 requires --input-frames 13 --output-frames 12")

    distributed, rank, world_size, device = distributed_context(args.device)
    started = time.perf_counter()
    dataset = _load_dataset(args, protocol, "phydnet" if args.model == "phydnet" else "phyrd")
    sampler = RankStridedSampler(len(dataset), rank, world_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    model, epoch = _load_model(args, device, protocol)
    sums = torch.zeros(7, dtype=torch.float64, device=device)
    counts = torch.zeros(len(POOL_SIZES), len(THRESHOLDS), 4, dtype=torch.float64, device=device)
    lead_abs = torch.zeros(args.output_frames, dtype=torch.float64, device=device)
    lead_n = torch.zeros(args.output_frames, dtype=torch.float64, device=device)
    with torch.inference_mode():
        for batch in loader:
            history, target = _batch_xy(batch, args.input_frames, args.output_frames)
            prediction = _predict(
                model, history.to(device, non_blocking=True), args.model
            ).clamp(0, 1).float()
            target = target.to(device, non_blocking=True).float()
            abs_sum, sq_sum, elements, batch_n = _continuous_metrics(prediction, target)
            sums[0] += abs_sum; sums[1] += sq_sum; sums[2] += elements; sums[3] += batch_n
            counts += contingency(prediction * 255, target * 255)
            sums[4] += _ssim(prediction, target).double() * prediction.shape[0] * args.output_frames
            sums[5] += torch.tensor(float(prediction.shape[0] * args.output_frames), device=device, dtype=torch.float64)
            if not args.skip_lpips:
                sums[6] += _lpips(prediction, target, args.lpips_chunk_size).double() * prediction.shape[0] * args.output_frames
            lead_abs += (prediction - target).abs().sum(dim=(0, 2, 3, 4), dtype=torch.float64)
            lead_n += prediction.shape[0] * prediction.shape[2] * prediction.shape[3] * prediction.shape[4]
    reduce_sum(sums, distributed); reduce_sum(counts, distributed); reduce_sum(lead_abs, distributed); reduce_sum(lead_n, distributed)
    if rank == 0:
        metrics: dict[str, Any] = {
            "status": "completed", "protocol": protocol, "model": args.model,
            "checkpoint": str(Path(args.checkpoint).resolve()), "data_path": str(Path(args.data_path).resolve()),
            "split": args.split, "samples": int(sums[3].item()), "frames_in": args.input_frames,
            "frames_out": args.output_frames, "img_size": args.img_size, "world_size": world_size,
            "per_rank_batch_size": args.batch_size, "epoch_or_step": epoch,
            "MAE": (sums[0] / sums[2]).item(), "MSE": (sums[1] / sums[2]).item(),
            "CRPS": (sums[0] / sums[2] * 255).item(), "CRPS_note": "deterministic K=1; CRPS=MAE in VIL [0,255]",
            "SSIM": (sums[4] / sums[5]).item(), "LPIPS": "SKIPPED" if args.skip_lpips else (sums[6] / sums[5]).item(),
            "thresholds": list(THRESHOLDS), "pool_sizes": list(POOL_SIZES),
            "seconds": time.perf_counter() - started,
            "lead_mae_vil": (lead_abs / lead_n.clamp_min(1) * 255).tolist(),
        }
        metrics.update(categorical_metrics(counts))
        metrics["visualizations"] = []
        if args.visualization_dir:
            for item in (x.strip() for x in args.visualization_samples.split(",")):
                if item:
                    index = int(item)
                    path = Path(args.visualization_dir) / f"sample_{index:05d}.png"
                    save_visualization(
                        model,
                        dataset,
                        index,
                        path,
                        device,
                        args.input_frames,
                        args.output_frames,
                        args.model,
                    )
                    metrics["visualizations"].append(str(path))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(metrics, indent=2, sort_keys=True))
    if hasattr(dataset, "close"):
        dataset.close()
    if distributed:
        dist.barrier(); dist.destroy_process_group()
