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

`universal_residual_diffusion` is the backbone-agnostic DiffCast-style option. It is
trained with a frozen deterministic pool and fixed `target - trend` residual coordinates;
it must not receive a `residual_stats_path` derived from a single backbone.

`udip` is the v12 deformation/intensity probability adapter. It consumes only
normalized history and a deterministic trend, builds an operational low-resolution
deformation plus full-resolution intensity target, and jointly samples both coordinates
with an explicit space-time denoiser. The deformation is a reconstruction coordinate,
not a physical wind estimate. Keep `physics.enabled: false` for U-DIP experiments.

`rescasformer` is the performance-first DiffCast-style residual model. It keeps
the common `target - trend` probability interface and replaces the 2D U-Net
denoiser with a clean-room CasFormer topology: matching noisy-residual and
deterministic leads are encoded frame-wise, aggregated at corresponding spatial
patches, and denoised jointly by sequence-wise DiT blocks. The formal P1 recipe
uses a 1000-step cosine schedule with v-prediction, DDIM-20 inference, and a
frozen deterministic backbone. Switching PhyDNet to SDIR must require only
configuration and backbone-specific residual statistics, never a model-specific
branch in the trainer.
