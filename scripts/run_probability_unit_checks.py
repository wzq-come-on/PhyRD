from __future__ import annotations

import torch

from phyrd.models import GaussianResidualDiffusion, ResidualDenoiser
from phyrd.physics import proximal_correct


def main() -> None:
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
    error = float((recovered - clean).abs().max().item())
    if not torch.isfinite(recovered).all() or error >= 1e-4:
        raise AssertionError(f"terminal v-prediction round trip failed: {error}")
    if not torch.allclose(diffusion.denormalize_residual(clean), raw, atol=1e-6):
        raise AssertionError("per-lead residual normalization did not round trip")

    history = torch.rand(2, 2, 1, 8, 8)
    trend = torch.rand(2, 3, 1, 8, 8)
    result = diffusion.training_loss(raw, history, trend, timestep=timestep)
    if not torch.isfinite(result["loss_gen"]):
        raise AssertionError("v-prediction training loss is non-finite")
    if result["clean_prediction_normalized"].abs().max() > 5.0 + 1e-6:
        raise AssertionError("standardized x0 clipping failed")

    residual = torch.randn_like(trend) * 0.2
    flow = torch.zeros(2, 2, 2, 8, 8)
    confidence = torch.ones(2, 2, 8, 8)
    nonadvective = torch.zeros_like(confidence)
    proximal = proximal_correct(
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
    if proximal.energy_after > proximal.energy_before + 1e-8:
        raise AssertionError("proximal correction increased transport energy")
    print(
        {
            "status": "ok",
            "terminal_v_round_trip_max_error": error,
            "training_loss": float(result["loss_gen"].item()),
            "proximal_energy_before": proximal.energy_before,
            "proximal_energy_after": proximal.energy_after,
        }
    )


if __name__ == "__main__":
    main()
