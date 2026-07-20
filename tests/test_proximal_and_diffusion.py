from __future__ import annotations

import torch

from phyrd.models import GaussianResidualDiffusion, ResidualDenoiser
from phyrd.physics import proximal_correct


def test_sampler_x0_noise_round_trip() -> None:
    denoiser = ResidualDenoiser(13, 12, base_channels=8)
    diffusion = GaussianResidualDiffusion(denoiser, diffusion_steps=20)
    clean = torch.randn(2, 12, 1, 16, 16)
    timestep = torch.tensor([3, 11])
    noisy, _ = diffusion.q_sample(clean, timestep)
    error = diffusion.reparameterization_error(noisy, clean, timestep)
    assert error < 1e-4


def test_v_prediction_is_stable_at_terminal_cosine_timestep() -> None:
    denoiser = ResidualDenoiser(2, 3, base_channels=8)
    diffusion = GaussianResidualDiffusion(
        denoiser,
        diffusion_steps=100,
        prediction_type="v",
        residual_center=[-0.01, 0.0, 0.01],
        residual_scale=[0.04, 0.05, 0.06],
    )
    raw = torch.randn(2, 3, 1, 8, 8) * 0.04
    clean = diffusion.normalize_residual(raw)
    timestep = torch.full((2,), 99, dtype=torch.long)
    noisy, noise = diffusion.q_sample(clean, timestep)
    alpha = diffusion._extract(diffusion.sqrt_alphas_cumprod, timestep, clean.ndim)
    sigma = diffusion._extract(diffusion.sqrt_one_minus_alphas_cumprod, timestep, clean.ndim)
    velocity = alpha * noise - sigma * clean
    recovered = diffusion.predict_x0(noisy, velocity, timestep)
    assert torch.isfinite(recovered).all()
    assert (recovered - clean).abs().max() < 1e-4
    assert torch.allclose(diffusion.denormalize_residual(clean), raw, atol=1e-6)


def test_v_training_returns_raw_domain_clean_prediction() -> None:
    denoiser = ResidualDenoiser(2, 3, base_channels=8)
    diffusion = GaussianResidualDiffusion(
        denoiser,
        diffusion_steps=20,
        residual_center=0.01,
        residual_scale=0.05,
        x0_clip=4.0,
    )
    residual = torch.randn(2, 3, 1, 8, 8) * 0.05
    history = torch.rand(2, 2, 1, 8, 8)
    trend = torch.rand(2, 3, 1, 8, 8)
    result = diffusion.training_loss(
        residual,
        history,
        trend,
        timestep=torch.tensor([0, 19]),
    )
    assert torch.isfinite(result["loss_gen"])
    assert torch.isfinite(result["clean_prediction"]).all()
    normalized = diffusion.normalize_residual(result["clean_prediction"])
    assert normalized.abs().max() <= 4.0 + 1e-6


def test_proximal_energy_is_monotone() -> None:
    trend = torch.rand(1, 4, 1, 16, 16)
    residual = torch.randn_like(trend) * 0.2
    flow = torch.zeros(1, 3, 2, 16, 16)
    confidence = torch.ones(1, 3, 16, 16)
    nonadvective = torch.zeros_like(confidence)
    result = proximal_correct(
        residual,
        trend,
        flow,
        confidence,
        nonadvective,
        step_size=1.0,
        robust_scale=0.1,
        tolerance=0.0,
        pool_sizes=(4, 8),
    )
    assert result.energy_after <= result.energy_before + 1e-8
