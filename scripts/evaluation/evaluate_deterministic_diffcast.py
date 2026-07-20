from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from phyrd.config import load_config
from phyrd.data import DiffCastH5Dataset
from phyrd.evaluation.categorical import SEVIR_THRESHOLDS
from phyrd.evaluation.continuous import lpips_distance, ssim
from phyrd.models import PhyRDModel, checkpoint_backbone_spec
from phyrd.utils import write_json


POOL_SIZES = (1, 4, 16)


class RankStridedSampler(Sampler[int]):
    """Partition a split across ranks without padding or duplicating samples."""

    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.length = int(length)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.length, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.length:
            return 0
        return (self.length - 1 - self.rank) // self.world_size + 1


def _pool(sequence: torch.Tensor, size: int) -> torch.Tensor:
    if size == 1:
        return sequence
    batch, frames, channels, height, width = sequence.shape
    flat = sequence.reshape(batch * frames, channels, height, width)
    pooled = F.max_pool2d(flat, kernel_size=size, stride=size)
    return pooled.reshape(batch, frames, channels, height // size, width // size)


def contingency_counts(
    prediction_vil: torch.Tensor,
    target_vil: torch.Tensor,
    *,
    pool_sizes: Sequence[int] = POOL_SIZES,
) -> torch.Tensor:
    """Return [pool, threshold, hit/miss/false-alarm/correct-negative] counts."""

    rows = []
    for pool_size in pool_sizes:
        prediction = _pool(prediction_vil, pool_size)
        target = _pool(target_vil, pool_size)
        threshold_rows = []
        for threshold in SEVIR_THRESHOLDS:
            predicted = prediction >= threshold
            observed = target >= threshold
            threshold_rows.append(
                torch.stack(
                    (
                        (predicted & observed).sum(dtype=torch.float64),
                        ((~predicted) & observed).sum(dtype=torch.float64),
                        (predicted & (~observed)).sum(dtype=torch.float64),
                        ((~predicted) & (~observed)).sum(dtype=torch.float64),
                    )
                )
            )
        rows.append(torch.stack(threshold_rows))
    return torch.stack(rows)


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator > 0, numerator / denominator, numerator.new_zeros(()))


def scores_from_counts(counts: torch.Tensor) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for pool_index, pool_size in enumerate(POOL_SIZES):
        pool_counts = counts[pool_index]
        hits, misses, false_alarms, correct_negatives = pool_counts.unbind(dim=1)
        csi = _safe_ratio(hits, hits + misses + false_alarms)
        name = "CSI" if pool_size == 1 else f"CSI_pool{pool_size}"
        metrics[name] = float(csi.mean().item())
        for threshold, value in zip(SEVIR_THRESHOLDS, csi, strict=True):
            metrics[f"{name}_{int(threshold)}"] = float(value.item())
        if pool_size == 1:
            numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
            denominator = (hits + misses) * (misses + correct_negatives) + (
                hits + false_alarms
            ) * (false_alarms + correct_negatives)
            hss = _safe_ratio(numerator, denominator)
            metrics["HSS"] = float(hss.mean().item())
            for threshold, value in zip(SEVIR_THRESHOLDS, hss, strict=True):
                metrics[f"HSS_{int(threshold)}"] = float(value.item())
    return metrics


def _distributed_context(device_argument: str) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, dist.get_rank(), world_size, torch.device("cuda", local_rank)
    return False, 0, 1, torch.device(device_argument)


def _reduce_sum(tensor: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _parse_sample_indices(value: str, length: int) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        return []
    for index in indices:
        if index < 0 or index >= length:
            raise ValueError(f"visualization sample {index} outside [0, {length})")
    return indices


def save_visualizations(
    deterministic: torch.nn.Module,
    dataset: DiffCastH5Dataset,
    indices: Sequence[int],
    output_dir: Path,
    device: torch.device,
    precision: str,
) -> list[str]:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    vil_colors = np.array(
        [
            [0, 0, 0],
            [77, 77, 77],
            [40, 190, 40],
            [25, 150, 25],
            [10, 105, 10],
            [10, 75, 10],
            [245, 245, 0],
            [237, 172, 0],
            [240, 110, 0],
            [160, 0, 0],
            [231, 0, 255],
        ],
        dtype=np.uint8,
    )
    vil_bounds = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 256])

    def load_font(size: int) -> ImageFont.ImageFont:
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    title_font = load_font(18)
    label_font = load_font(14)
    small_font = load_font(11)

    def vil_rgb(values: np.ndarray) -> np.ndarray:
        encoded = np.clip(values.astype(np.float32), 0.0, 255.0)
        bins = np.digitize(encoded, vil_bounds[1:-1], right=False)
        color_indices = np.rint(bins * (len(vil_colors) - 1) / (len(vil_bounds) - 2))
        return vil_colors[color_indices.astype(np.int64)]

    def residual_rgb(values: np.ndarray, limit: float) -> np.ndarray:
        normalized = np.clip(values.astype(np.float32) / limit, -1.0, 1.0)
        rgb = np.full((*normalized.shape, 3), 245.0, dtype=np.float32)
        positive = normalized >= 0
        negative = ~positive
        rgb[positive, 1] *= 1.0 - normalized[positive]
        rgb[positive, 2] *= 1.0 - normalized[positive]
        rgb[negative, 0] *= 1.0 + normalized[negative]
        rgb[negative, 1] *= 1.0 + normalized[negative]
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def colorbar(kind: str, height: int, limit: float) -> Image.Image:
        if kind == "vil":
            values = np.linspace(255.0, 0.0, height, dtype=np.float32)[:, None]
            rgb = vil_rgb(values)
        else:
            values = np.linspace(limit, -limit, height, dtype=np.float32)[:, None]
            rgb = residual_rgb(values, limit)
        return Image.fromarray(np.repeat(rgb, 18, axis=1), mode="RGB")

    with torch.inference_mode():
        for index in indices:
            sample = dataset[index]
            history = sample["x"].unsqueeze(0).to(device)
            target = sample["y"].float()
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                prediction = deterministic(history).clamp(0.0, 1.0)
            prediction = prediction[0].float().cpu()
            history = history[0].float().cpu()
            target = target.float().cpu()
            history_vil = history[:, 0].numpy() * 255.0
            prediction_vil = prediction[:, 0].numpy() * 255.0
            target_vil = target[:, 0].numpy() * 255.0
            residual_vil = prediction_vil - target_vil
            residual_limit = max(float(np.max(np.abs(residual_vil))), 16.0)
            normalized_mae = float(np.mean(np.abs(residual_vil)) / 255.0)

            columns = dataset.output_frames
            cell_size = dataset.model_resolution
            title_height = 22
            row_gap = 10
            label_width = 128
            right_width = 118
            top_height = 48
            bottom_height = 32
            canvas_width = label_width + columns * cell_size + right_width
            canvas_height = (
                top_height
                + 4 * (title_height + cell_size)
                + 3 * row_gap
                + bottom_height
            )
            canvas = Image.new("RGB", (canvas_width, canvas_height), color=(250, 250, 250))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (label_width, 12),
                f"SDIR 5-to-20 deterministic forecast | test sample {index:05d}",
                fill=(25, 25, 25),
                font=title_font,
            )

            history_titles = [f"t-{dataset.input_frames - i - 1}" for i in range(dataset.input_frames)]
            history_titles[-1] = "t0"
            future_titles = [f"t+{i + 1}" for i in range(dataset.output_frames)]

            def draw_row(
                row_index: int,
                sequence: np.ndarray,
                label: str,
                titles: Sequence[str],
                kind: str,
            ) -> None:
                y0 = top_height + row_index * (title_height + cell_size + row_gap)
                draw.multiline_text(
                    (10, y0 + title_height + cell_size // 2 - 12),
                    label,
                    fill=(30, 30, 30),
                    font=label_font,
                    spacing=2,
                )
                for column in range(columns):
                    x0 = label_width + column * cell_size
                    if column >= len(sequence):
                        continue
                    title = titles[column] if column < len(titles) else ""
                    draw.text((x0 + 4, y0 + 4), title, fill=(55, 55, 55), font=small_font)
                    values = sequence[column]
                    rgb = vil_rgb(values) if kind == "vil" else residual_rgb(values, residual_limit)
                    canvas.paste(Image.fromarray(rgb, mode="RGB"), (x0, y0 + title_height))

            draw_row(0, history_vil, "History", history_titles, "vil")
            draw_row(1, prediction_vil, "Prediction", future_titles, "vil")
            draw_row(2, target_vil, "Ground Truth", future_titles, "vil")
            draw_row(3, residual_vil, "Residual\n(Pred - GT)", future_titles, "residual")

            colorbar_x = label_width + columns * cell_size + 22
            intensity_y = top_height + title_height
            intensity_height = 3 * cell_size + 2 * (title_height + row_gap)
            canvas.paste(colorbar("vil", intensity_height, residual_limit), (colorbar_x, intensity_y))
            draw.text((colorbar_x + 25, intensity_y - 2), "255", fill=(40, 40, 40), font=small_font)
            draw.text(
                (colorbar_x + 25, intensity_y + intensity_height // 2 - 6),
                "VIL",
                fill=(40, 40, 40),
                font=small_font,
            )
            draw.text(
                (colorbar_x + 25, intensity_y + intensity_height - 12),
                "0",
                fill=(40, 40, 40),
                font=small_font,
            )
            residual_y = top_height + 3 * (title_height + cell_size + row_gap) + title_height
            canvas.paste(colorbar("residual", cell_size, residual_limit), (colorbar_x, residual_y))
            draw.text(
                (colorbar_x + 25, residual_y - 2),
                f"+{residual_limit:.0f}",
                fill=(40, 40, 40),
                font=small_font,
            )
            draw.text(
                (colorbar_x + 25, residual_y + cell_size // 2 - 6),
                "0",
                fill=(40, 40, 40),
                font=small_font,
            )
            draw.text(
                (colorbar_x + 25, residual_y + cell_size - 12),
                f"-{residual_limit:.0f}",
                fill=(40, 40, 40),
                font=small_font,
            )
            draw.text(
                (label_width, canvas_height - bottom_height + 8),
                f"DiffCast/SEVIR palette | VIL [0,255] | FP32 | normalized MAE={normalized_mae:.4f}",
                fill=(80, 80, 80),
                font=small_font,
            )
            path = output_dir / f"test_sample_{index:05d}.png"
            canvas.save(path)
            saved.append(str(path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic SDIR on DiffCast test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument("--lpips-chunk-size", type=int, default=32)
    parser.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    parser.add_argument("--visualization-dir")
    parser.add_argument("--visualization-samples", default="0,1,2")
    args = parser.parse_args()

    distributed, rank, world_size, device = _distributed_context(args.device)
    started = time.perf_counter()
    config = load_config(args.config)
    data = config["data"]
    dataset = DiffCastH5Dataset(
        data["root"],
        args.split,
        input_frames=int(data["input_frames"]),
        output_frames=int(data["output_frames"]),
        window_start_index=int(data["window_start_index"]),
        model_resolution=int(data["model_resolution"]),
        spatial_preprocess=str(data["spatial_preprocess"]),
        max_samples=args.max_samples,
    )
    sampler = RankStridedSampler(len(dataset), rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model_config = config["model"]
    model = PhyRDModel(
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
        base_channels=int(model_config["base_channels"]),
        diffusion_steps=int(model_config["diffusion_steps"]),
        freeze_deterministic=False,
        deterministic=dict(model_config["deterministic"]),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    protocol = checkpoint.get("protocol", {})
    checkpoint_deterministic = checkpoint_backbone_spec(protocol)
    if checkpoint_deterministic != dict(model_config["deterministic"]):
        raise ValueError("checkpoint is not an SDIR deterministic checkpoint")
    model.deterministic.load_state_dict(checkpoint["deterministic"])
    deterministic = model.deterministic.eval().to(device)

    scalar_sums = torch.zeros(8, dtype=torch.float64, device=device)
    global_counts = torch.zeros(
        len(POOL_SIZES), len(SEVIR_THRESHOLDS), 4, dtype=torch.float64, device=device
    )
    lead_absolute_sums = torch.zeros(dataset.output_frames, dtype=torch.float64, device=device)
    lead_elements = torch.zeros(dataset.output_frames, dtype=torch.float64, device=device)
    try:
        with torch.inference_mode():
            iterator = tqdm(loader, desc=args.split, dynamic_ncols=True, disable=rank != 0)
            for batch in iterator:
                history = batch["x"].to(device, non_blocking=True)
                target = batch["y"].to(device, non_blocking=True).float()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=args.precision == "bf16",
                ):
                    prediction = deterministic(history).clamp(0.0, 1.0)
                prediction = prediction.float()
                difference = prediction - target
                batch_size = int(history.shape[0])
                elements = difference.numel()
                absolute_sum = difference.abs().sum(dtype=torch.float64)
                squared_sum = difference.square().sum(dtype=torch.float64)
                counts = contingency_counts(prediction * 255.0, target * 255.0)
                hits, misses, false_alarms = counts[0, :, :3].unbind(dim=1)
                batch_csi = _safe_ratio(
                    hits, hits + misses + false_alarms
                ).mean()
                batch_ssim = ssim(prediction, target)
                if args.skip_lpips:
                    batch_lpips = prediction.new_zeros(())
                else:
                    batch_lpips = lpips_distance(
                        prediction,
                        target,
                        net="alex",
                        chunk_size=args.lpips_chunk_size,
                    )
                scalar_sums += torch.stack(
                    (
                        absolute_sum,
                        squared_sum,
                        absolute_sum.new_tensor(elements),
                        absolute_sum.new_tensor(batch_size),
                        batch_csi.to(torch.float64) * batch_size,
                        batch_ssim.to(torch.float64) * batch_size,
                        batch_lpips.to(torch.float64) * batch_size,
                        absolute_sum.new_ones(()),
                    )
                )
                global_counts += counts
                lead_absolute_sums += difference.abs().sum(dim=(0, 2, 3, 4), dtype=torch.float64)
                per_lead_elements = batch_size * target.shape[2] * target.shape[3] * target.shape[4]
                lead_elements += per_lead_elements
    finally:
        dataset.close()

    _reduce_sum(scalar_sums, distributed)
    _reduce_sum(global_counts, distributed)
    _reduce_sum(lead_absolute_sums, distributed)
    _reduce_sum(lead_elements, distributed)

    if rank == 0:
        absolute_sum, squared_sum, elements, samples = scalar_sums[:4].tolist()
        metrics: dict[str, object] = {
            "status": "completed",
            "data_path": str(dataset.path),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "split": args.split,
            "samples": int(samples),
            "frames_in": dataset.input_frames,
            "frames_out": dataset.output_frames,
            "img_size": dataset.model_resolution,
            "world_size": world_size,
            "per_rank_batch_size": args.batch_size,
            "inference_precision": args.precision,
            "test_csi": scalar_sums[4].item() / samples,
            "test_mae": absolute_sum / elements,
            "test_mae_vil": absolute_sum / elements * 255.0,
            "test_mse": squared_sum / elements,
            "SSIM": scalar_sums[5].item() / samples,
            "LPIPS": "SKIPPED" if args.skip_lpips else scalar_sums[6].item() / samples,
            "CRPS": absolute_sum / elements * 255.0,
            "CRPS_note": "K=1 deterministic forecast; CRPS equals MAE in encoded VIL [0,255]",
            "metric_domain": "CSI/HSS/MAE/CRPS: VIL [0,255]; LPIPS/SSIM: [0,1]",
            "seconds": time.perf_counter() - started,
        }
        metrics.update(scores_from_counts(global_counts))
        metrics["test_csi_global"] = metrics["CSI"]
        metrics["lead_mae_vil"] = [
            float(value)
            for value in (lead_absolute_sums / lead_elements.clamp_min(1)).mul(255.0).tolist()
        ]
        visualization_paths: list[str] = []
        metrics["visualizations"] = visualization_paths
        write_json(Path(args.output), metrics)
        if args.visualization_dir:
            indices = _parse_sample_indices(args.visualization_samples, len(dataset))
            visualization_paths = save_visualizations(
                deterministic,
                dataset,
                indices,
                Path(args.visualization_dir),
                device,
                args.precision,
            )
        metrics["visualizations"] = visualization_paths
        write_json(Path(args.output), metrics)
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
