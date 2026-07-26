from __future__ import annotations

import torch
from torch import nn

from phyrd.models.probabilistic import (
    available_probabilistic_models,
    build_probabilistic,
)
from phyrd.models.probabilistic.temporal_residual_dit import (
    TemporalResidualDenoiser,
    TemporalResidualDiffusionModel,
)


def tiny_denoiser() -> TemporalResidualDenoiser:
    return TemporalResidualDenoiser(
        2,
        4,
        image_size=16,
        patch_size=4,
        hidden_size=16,
        depth=2,
        num_heads=4,
        high_frequency_channels=8,
        gradient_checkpointing=False,
    )


def tiny_model() -> TemporalResidualDiffusionModel:
    return TemporalResidualDiffusionModel(
        2,
        4,
        image_size=16,
        patch_size=4,
        hidden_size=16,
        depth=2,
        num_heads=4,
        high_frequency_channels=8,
        gradient_checkpointing=False,
        diffusion_steps=8,
        prediction_type="v",
        residual_center=[0.0, 0.01, -0.01, 0.0],
        residual_scale=[0.1, 0.2, 0.3, 0.4],
        high_frequency_loss_weight=0.1,
        intensity_loss_weight=0.05,
    )


def test_temporal_denoiser_shape_and_explicit_leads() -> None:
    denoiser = tiny_denoiser()
    noisy = torch.randn(2, 4, 1, 16, 16)
    history = torch.rand(2, 2, 1, 16, 16)
    trend = torch.rand(2, 4, 1, 16, 16)
    output = denoiser(noisy, torch.tensor([0, 7]), history, trend)
    assert output.shape == noisy.shape
    assert denoiser.future_lead_embedding.shape == (1, 4, 16)
    assert len(denoiser.blocks) == 2
    assert all(hasattr(block, "temporal") for block in denoiser.blocks)
    assert all(hasattr(block, "context") for block in denoiser.blocks)


def test_history_changes_prediction_through_cross_attention() -> None:
    torch.manual_seed(3)
    denoiser = tiny_denoiser().eval()
    # AdaLN-Zero starts as an identity path. Open only the context gates so
    # this test proves that observed history is in the actual computation.
    for block in denoiser.blocks:
        linear = block.context.modulation[-1]
        assert isinstance(linear, nn.Linear)
        nn.init.normal_(linear.weight, std=0.05)
        nn.init.normal_(linear.bias, std=0.05)
    noisy = torch.randn(1, 4, 1, 16, 16)
    trend = torch.rand(1, 4, 1, 16, 16)
    first_history = torch.zeros(1, 2, 1, 16, 16)
    second_history = torch.ones_like(first_history)
    first = denoiser(noisy, torch.tensor([4]), first_history, trend)
    second = denoiser(noisy, torch.tensor([4]), second_history, trend)
    assert not torch.allclose(first, second)


def test_temporal_residual_training_backward_and_sampling() -> None:
    model = tiny_model()
    history = torch.rand(1, 2, 1, 16, 16)
    trend = torch.rand(1, 4, 1, 16, 16)
    target = torch.rand_like(trend)
    result = model.training_loss(history, target, trend)
    for name in (
        "loss_gen",
        "loss_diffusion",
        "loss_high_frequency",
        "loss_intensity",
    ):
        assert torch.isfinite(result[name])
    assert result["prediction_x0"].shape == target.shape
    result["loss_gen"].backward()
    assert model.diffusion.denoiser.future_lead_embedding.grad is not None
    ensemble = model.sample(
        history,
        trend,
        ensemble_size=2,
        sampling_steps=2,
    )
    assert ensemble.shape == (1, 2, 4, 1, 16, 16)
    assert ensemble.min() >= 0
    assert ensemble.max() <= 1


def test_temporal_residual_registry_entry() -> None:
    assert "temporal_residual_dit" in available_probabilistic_models()
    model = build_probabilistic(
        "temporal_residual_dit",
        input_frames=2,
        output_frames=4,
        params={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 16,
            "depth": 1,
            "num_heads": 4,
            "high_frequency_channels": 8,
            "gradient_checkpointing": False,
            "diffusion_steps": 8,
        },
    )
    assert isinstance(model, TemporalResidualDiffusionModel)
