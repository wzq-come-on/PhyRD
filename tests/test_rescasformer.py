from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from phyrd.models.composer import ForecastComposer
from phyrd.models.probabilistic.rescasformer import (
    ResidualCasFormerDenoiser,
    ResidualCasFormerModel,
)
from phyrd.models.probabilistic.rescasformer.blocks import (
    patchify_pixels,
    unpatchify_pixels,
)
from phyrd.train import learning_rate_multiplier


class TinyBackbone(nn.Module):
    def __init__(self, output_frames: int, gain: float) -> None:
        super().__init__()
        self.output_frames = output_frames
        self.gain = nn.Parameter(torch.tensor(gain))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return (
            history[:, -1:]
            .expand(-1, self.output_frames, -1, -1, -1)
            .mul(self.gain)
            .clamp(0, 1)
        )


def tiny_probability() -> ResidualCasFormerModel:
    return ResidualCasFormerModel(
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
        prediction_type="v",
        residual_center=[0.0, 0.01, -0.01, 0.0],
        residual_scale=[0.1, 0.2, 0.3, 0.4],
        gradient_checkpointing=True,
    )


def test_pixel_patch_round_trip() -> None:
    image = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).reshape(2, 3, 16, 16)
    patches = patchify_pixels(image, 4)
    recovered = unpatchify_pixels(
        patches,
        patch_size=4,
        channels=3,
        grid_height=4,
        grid_width=4,
    )
    torch.testing.assert_close(recovered, image)


def test_rescasformer_denoiser_shape_and_zero_initialized_output() -> None:
    denoiser = ResidualCasFormerDenoiser(
        2,
        4,
        image_size=16,
        patch_size=4,
        frame_hidden_size=16,
        frame_depth=1,
        frame_heads=4,
        sequence_hidden_size=32,
        sequence_depth=1,
        sequence_heads=4,
        gradient_checkpointing=False,
    )
    noisy = torch.randn(2, 4, 1, 16, 16)
    history = torch.rand(2, 2, 1, 16, 16)
    trend = torch.rand(2, 4, 1, 16, 16)
    output = denoiser(noisy, torch.tensor([0, 7]), history, trend)
    assert output.shape == noisy.shape
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_rescasformer_training_backward_and_sampling() -> None:
    model = tiny_probability()
    history = torch.rand(1, 2, 1, 16, 16)
    trend = torch.rand(1, 4, 1, 16, 16)
    target = torch.rand_like(trend)
    result = model.training_loss(history, target, trend)
    assert torch.isfinite(result["loss_gen"])
    assert result["prediction_x0"].shape == target.shape
    result["loss_gen"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    ensemble = model.sample(
        history,
        trend,
        ensemble_size=2,
        sampling_steps=2,
    )
    assert ensemble.shape == (1, 2, 4, 1, 16, 16)
    assert ensemble.min() >= 0
    assert ensemble.max() <= 1


def test_residual_statistics_round_trip_and_terminal_noise() -> None:
    model = tiny_probability()
    diffusion = model.diffusion
    residual = torch.randn(2, 4, 1, 8, 8) * 0.1
    normalized = diffusion.normalize_residual(residual)
    recovered = diffusion.denormalize_residual(normalized)
    torch.testing.assert_close(recovered, residual)

    formal = ResidualCasFormerModel(
        2,
        4,
        image_size=16,
        patch_size=4,
        frame_hidden_size=16,
        frame_depth=1,
        frame_heads=4,
        sequence_hidden_size=32,
        sequence_depth=1,
        sequence_heads=4,
        diffusion_steps=1000,
        gradient_checkpointing=False,
    )
    assert formal.diffusion.alphas_cumprod[-1].item() < 1e-6


def test_checkpoint_round_trip_and_backbone_swap() -> None:
    source = tiny_probability()
    state = copy.deepcopy(source.state_dict())
    restored = tiny_probability()
    restored.load_state_dict(state, strict=True)
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])

    history = torch.rand(1, 2, 1, 16, 16)
    phydnet_like = ForecastComposer(
        TinyBackbone(4, 0.8),
        source,
        freeze_deterministic=True,
        deterministic_name="phydnet_like",
    )
    sdir_like = ForecastComposer(
        TinyBackbone(4, 0.6),
        restored,
        freeze_deterministic=True,
        deterministic_name="sdir_like",
    )
    first = phydnet_like.sample(history, ensemble_size=1, sampling_steps=1)
    second = sdir_like.sample(history, ensemble_size=1, sampling_steps=1)
    assert first.shape == second.shape == (1, 1, 4, 1, 16, 16)


def test_cosine_learning_rate_schedule() -> None:
    optimization = {
        "learning_rate": 5e-4,
        "scheduler": "cosine",
        "warmup_lr": 1e-5,
        "min_lr": 1e-5,
        "warmup_epochs": 0.1,
    }
    start = learning_rate_multiplier(
        optimization,
        step=0,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    peak = learning_rate_multiplier(
        optimization,
        step=100,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    finish = learning_rate_multiplier(
        optimization,
        step=200_000,
        max_steps=200_000,
        steps_per_epoch=1000,
    )
    assert start == pytest.approx(0.02)
    assert peak == pytest.approx(1.0)
    assert finish == pytest.approx(0.02)
