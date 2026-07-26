from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from phyrd.config import load_config
from phyrd.data import DiffCastH5Dataset
from phyrd.models import build_composite_from_config


def color_norm() -> BoundaryNorm:
    colors = [
        "#000000",
        "#32cd32",
        "#16a516",
        "#087208",
        "#005500",
        "#ffff00",
        "#f5b000",
        "#f07800",
        "#e04400",
        "#b00000",
        "#e000e0",
    ]
    boundaries = [0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255, 256]
    return BoundaryNorm(boundaries, len(colors)), ListedColormap(colors)


def choose_sample(dataset: DiffCastH5Dataset, requested: int) -> tuple[int, dict]:
    if requested >= 0:
        return requested, dataset[requested]
    best_score = -1.0
    best_index = 0
    best_item = dataset[0]
    for index in range(min(128, len(dataset))):
        item = dataset[index]
        score = float((item["y"] > 0.5).float().mean())
        if score > best_score:
            best_score, best_index, best_item = score, index, item
    return best_index, best_item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deterministic-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=-1)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--method-label", default="Temporal Residual DiT")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    data_config = dict(config["data"])
    dataset = DiffCastH5Dataset(
        args.data,
        "report_test",
        input_frames=int(data_config["input_frames"]),
        output_frames=int(data_config["output_frames"]),
        window_start_index=0,
        model_resolution=int(data_config["model_resolution"]),
        spatial_preprocess=str(data_config["spatial_preprocess"]),
    )
    sample_index, item = choose_sample(dataset, args.sample_index)
    history = item["x"].unsqueeze(0).to(device)
    target = item["y"].unsqueeze(0).to(device)

    model = build_composite_from_config(
        config,
        input_frames=dataset.input_frames,
        output_frames=dataset.output_frames,
    ).to(device).eval()
    deterministic_loader = getattr(model.deterministic, "load_external_checkpoint", None)
    if callable(deterministic_loader):
        deterministic_loader(args.deterministic_checkpoint)
    else:
        deterministic_payload = torch.load(
            args.deterministic_checkpoint, map_location="cpu", weights_only=False
        )
        model.deterministic.load_state_dict(
            deterministic_payload.get("deterministic", deterministic_payload), strict=True
        )
    with torch.inference_mode():
        pure_phydnet = model.predict_trend(history).clamp(0, 1)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "deterministic" in payload:
        model.deterministic.load_state_dict(payload["deterministic"], strict=True)
    model.diffusion.load_state_dict(payload["diffusion"], strict=True)

    with torch.inference_mode():
        trend = model.predict_trend(history).clamp(0, 1)
        ensemble = model.sample(
            history,
            ensemble_size=args.ensemble_size,
            sampling_steps=args.sampling_steps,
        ).clamp(0, 1)
        prediction = ensemble.mean(dim=1)

    input_frames = history[0, :, 0].cpu().numpy() * 255.0
    truth = target[0, :, 0].cpu().numpy() * 255.0
    deterministic = pure_phydnet[0, :, 0].cpu().numpy() * 255.0
    stochastic = prediction[0, :, 0].cpu().numpy() * 255.0
    norm, cmap = color_norm()
    selected = list(range(1, dataset.output_frames, 2))
    fig = plt.figure(figsize=(24, 10), dpi=180)
    grid = fig.add_gridspec(
        4, 12, width_ratios=[1.2] + [1] * 10 + [0.35], hspace=0.16, wspace=0.035
    )
    rows = [
        ("Input x", input_frames, list(range(dataset.input_frames)), 5),
        ("Ground Truth y", truth, selected, 10),
        ("PhyDNet", deterministic, selected, 10),
        (f"{args.method_label}\n(ensemble mean)", stochastic, selected, 10),
    ]
    for row_index, (label, frames, frame_indices, count) in enumerate(rows):
        axis_label = fig.add_subplot(grid[row_index, 0])
        axis_label.axis("off")
        axis_label.text(0.95, 0.5, label, ha="right", va="center", fontsize=14)
        for column, frame_index in enumerate(frame_indices):
            ax = fig.add_subplot(grid[row_index, column + 1])
            ax.imshow(frames[frame_index], cmap=cmap, norm=norm, interpolation="nearest")
            ax.axis("off")
            if row_index == 0:
                ax.set_title(f"-{(4-frame_index)*5} min" if frame_index < 4 else "0 min", fontsize=11)
            elif row_index == 1:
                ax.set_title(f"{(frame_index+1)*5} min", fontsize=11)
    colorbar_axis = fig.add_subplot(grid[:, 11])
    colorbar_axis.axis("off")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar, ax=colorbar_axis, fraction=0.75, pad=0.05)
    colorbar.set_ticks([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255])
    colorbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        f"PhyDNet vs {args.method_label} | report_test sample {sample_index} ({item['sample_id']}) | "
        f"K={args.ensemble_size} mean",
        fontsize=16,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    dataset.close()
    print(output)


if __name__ == "__main__":
    main()
