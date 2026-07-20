from __future__ import annotations

import argparse
import json
import time

import torch

from phyrd.models import PhyRDModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Production-width SDIR CUDA validation")
    parser.add_argument("--input-frames", type=int, default=5)
    parser.add_argument("--output-frames", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--frequency-stride", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("SDIR CUDA validation requires a CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    model = PhyRDModel(
        input_frames=args.input_frames,
        output_frames=args.output_frames,
        base_channels=64,
        diffusion_steps=20,
        freeze_deterministic=False,
        deterministic={
            "name": "sdir_official",
            "params": {
                "patch_size": args.patch_size,
                "hidden_size": 512,
                "num_heads": 4,
                "depth": 8,
                "frequency_stride": args.frequency_stride,
                "curriculum_alpha": 1.0,
                "curriculum_beta": 3.0,
                "pcpsd_weight": 0.01,
                "model_resolution": args.resolution,
            },
        },
    ).to(device)
    model.diffusion.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.deterministic.parameters(), lr=3e-4)
    history = torch.rand(
        1, args.input_frames, 1, args.resolution, args.resolution, device=device
    )
    target = torch.rand(
        1, args.output_frames, 1, args.resolution, args.resolution, device=device
    )

    model.train()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(history, target, stage="deterministic")
    if not torch.isfinite(result["loss_gen"]):
        raise FloatingPointError("non-finite SDIR training loss")
    result["loss_gen"].backward()
    sfg_gradient = sum(
        float(parameter.grad.float().abs().sum())
        for parameter in model.deterministic.network.sfg_former.parameters()
        if parameter.grad is not None
    )
    refiner_gradient = sum(
        float(parameter.grad.float().abs().sum())
        for parameter in model.deterministic.network.fr_refiner.parameters()
        if parameter.grad is not None
    )
    if sfg_gradient <= 0 or refiner_gradient <= 0:
        raise AssertionError("SDIR gradient did not reach both deterministic paths")
    optimizer.step()
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started

    optimizer.zero_grad(set_to_none=True)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = model(history, stage="deterministic")
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started
    expected_shape = (1, args.output_frames, 1, args.resolution, args.resolution)
    if tuple(prediction.shape) != expected_shape or not torch.isfinite(prediction).all():
        raise AssertionError(
            f"invalid SDIR inference: shape={tuple(prediction.shape)}, "
            f"finite={bool(torch.isfinite(prediction).all())}"
        )

    report = {
        "status": "passed",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "protocol": f"{args.input_frames}to{args.output_frames}@{args.resolution}",
        "deterministic_parameters": sum(
            parameter.numel() for parameter in model.deterministic.parameters()
        ),
        "loss": float(result["loss_gen"]),
        "loss_skeleton": float(result["loss_skeleton"]),
        "loss_residual": float(result["loss_residual"]),
        "loss_pcpsd": float(result["loss_pcpsd"]),
        "sfg_gradient_l1": sfg_gradient,
        "refiner_gradient_l1": refiner_gradient,
        "train_step_seconds": train_seconds,
        "frequency_unlocking_seconds": inference_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
