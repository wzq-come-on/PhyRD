# Probabilistic models

This directory contains the stochastic forecast side of PhyRD. Each model has
its own package so its denoiser, scheduler, sampler, calibration and optional
physics hooks stay together.

Every model registered in `registry.py` exposes:

- `training_loss(history, target, trend)` returning a mapping with `loss_gen`;
- `sample(history, trend, ensemble_size=..., sampling_steps=...)` returning
  `[B,E,T,1,H,W]` samples.

The deterministic trend is passed explicitly by `models/composer.py`. Adding a
new probabilistic model therefore requires a new package and one registry entry,
not a new model-specific branch in `scripts/train.py`.
