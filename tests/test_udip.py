from __future__ import annotations

import torch

from phyrd.models.probabilistic import available_probabilistic_models, build_probabilistic
from phyrd.models.probabilistic.udip import (
    DecompositionTargetBuilder,
    UDIPModel,
    reconstruct,
    warp_video,
)


def test_zero_deformation_is_identity() -> None:
    video = torch.rand(2, 3, 1, 16, 20)
    deformation = torch.zeros(2, 3, 2, 4, 5)
    assert torch.allclose(warp_video(video, deformation), video, atol=3e-6)


def test_decomposition_targets_reconstruct_exactly() -> None:
    trend = torch.rand(1, 3, 1, 16, 16)
    target = torch.rand_like(trend)
    builder = DecompositionTargetBuilder(steps=1, downsample_factor=4)
    targets = builder(trend, target)
    recovered = reconstruct(trend, targets.deformation, targets.intensity)
    assert targets.deformation.shape == (1, 3, 2, 4, 4)
    assert targets.confidence.shape == (1, 3, 1, 4, 4)
    assert torch.allclose(recovered, target, atol=1e-5)
    assert torch.all((0 <= targets.confidence) & (targets.confidence <= 1))


def test_registration_reduces_a_synthetic_translation_error() -> None:
    trend = torch.zeros(1, 2, 1, 32, 32)
    trend[:, :, :, 10:18, 8:16] = 1.0
    target = torch.roll(trend, shifts=3, dims=-1)
    builder = DecompositionTargetBuilder(
        steps=8, learning_rate=0.25, downsample_factor=4
    )
    targets = builder(trend, target)
    before = (trend - target).square().mean()
    after = (targets.warped_trend - target).square().mean()
    assert after < before * 0.75


def test_udip_training_and_sampling_contract() -> None:
    model = UDIPModel(
        input_frames=2,
        output_frames=3,
        base_channels=8,
        diffusion_steps=8,
        registration_steps=0,
        reconstruction_timestep_max=7,
        history_guidance_weight=0.1,
        anchor_guidance_weight=0.1,
    )
    history = torch.rand(1, 2, 1, 16, 16)
    trend = torch.rand(1, 3, 1, 16, 16)
    target = torch.rand_like(trend)
    result = model.training_loss(history, target, trend)
    assert result["prediction_x0"].shape == target.shape
    assert result["clean_deformation"].shape == (1, 3, 2, 4, 4)
    assert result["frame_timestep"].shape == (1, 3)
    assert torch.isfinite(result["loss_gen"])
    result["loss_gen"].backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.diffusion.parameters()
    )
    ensemble = model.sample(history, trend, ensemble_size=2, sampling_steps=4)
    assert ensemble.shape == (1, 2, 3, 1, 16, 16)
    assert torch.isfinite(ensemble).all()
    assert torch.all((0 <= ensemble) & (ensemble <= 1))


def test_udip_is_registered() -> None:
    assert "udip" in available_probabilistic_models()
    model = build_probabilistic(
        "udip",
        input_frames=2,
        output_frames=3,
        params={
            "base_channels": 8,
            "diffusion_steps": 8,
            "registration_steps": 0,
        },
    )
    assert isinstance(model, UDIPModel)
