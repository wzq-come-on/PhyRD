from __future__ import annotations

import copy
import json

import torch

from phyrd.config import load_config
from phyrd.models import available_probabilistic_models
from phyrd.models.probabilistic.rescasformer import ResidualCasFormerModel
from phyrd.models.probabilistic.rescasformer.blocks import (
    patchify_pixels,
    unpatchify_pixels,
)
from phyrd.train import learning_rate_multiplier


def main() -> None:
    image = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(
        2,
        3,
        16,
        16,
    )
    patches = patchify_pixels(image, 4)
    recovered = unpatchify_pixels(
        patches,
        patch_size=4,
        channels=3,
        grid_height=4,
        grid_width=4,
    )
    torch.testing.assert_close(recovered, image)

    model = ResidualCasFormerModel(
        2,
        4,
        image_size=16,
        patch_size=4,
        frame_hidden_size=16,
        frame_depth=2,
        frame_heads=4,
        sequence_hidden_size=32,
        sequence_depth=2,
        sequence_heads=4,
        diffusion_steps=8,
        residual_scale=[0.1, 0.2, 0.3, 0.4],
        gradient_checkpointing=True,
    )
    history = torch.rand(1, 2, 1, 16, 16)
    trend = torch.rand(1, 4, 1, 16, 16)
    target = torch.rand_like(trend)
    result = model.training_loss(history, target, trend)
    if not torch.isfinite(result["loss_gen"]):
        raise AssertionError("training loss is not finite")
    result["loss_gen"].backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("backward did not produce gradients")
    ensemble = model.sample(
        history,
        trend,
        ensemble_size=2,
        sampling_steps=2,
    )
    if ensemble.shape != (1, 2, 4, 1, 16, 16):
        raise AssertionError(f"unexpected sample shape: {tuple(ensemble.shape)}")
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict(), strict=True)

    formal_config = load_config(
        "configs/active/5to20/"
        "train_ddp8_phydnet_rescasformer_5to20_v14_seed42.yaml"
    )
    if formal_config["optimization"]["max_steps"] != 200_000:
        raise AssertionError("formal config must train for 200,000 optimizer steps")
    if "rescasformer" not in available_probabilistic_models():
        raise AssertionError("rescasformer was not registered")
    schedule = formal_config["optimization"]
    lr_start = learning_rate_multiplier(
        schedule,
        step=0,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    lr_peak = learning_rate_multiplier(
        schedule,
        step=100,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    lr_end = learning_rate_multiplier(
        schedule,
        step=200_000,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    if not (
        abs(lr_start - 0.02) < 1e-8
        and abs(lr_peak - 1.0) < 1e-8
        and abs(lr_end - 0.02) < 1e-8
    ):
        raise AssertionError("cosine learning-rate schedule endpoints are wrong")

    print(
        json.dumps(
            {
                "status": "PASS",
                "loss": float(result["loss_gen"]),
                "sample_shape": list(ensemble.shape),
                "tiny_parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "lr_multiplier": {
                    "start": lr_start,
                    "peak": lr_peak,
                    "end": lr_end,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

