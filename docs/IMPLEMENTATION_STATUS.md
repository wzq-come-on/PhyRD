# v10.3 Implementation Status

> The active research protocol is `5→20@128`. Earlier `13→12` entries below are retained as interface-validation history rather than the current main-task definition.

## Implemented in the research-ready MVP

- Explicit SEVIR catalog/HDF5 path resolution and fixed event-center `13→12` window.
- Canonical `none@384`, DiffCast-compatible non-antialiased bilinear `@128`, and conservative area-pooling `@128` spatial protocols.
- Four event-time splits (`train`, `val_model`, `val_calib`, `report_test`).
- Two-stage deterministic-trend then residual-diffusion training contract.
- SDIR deterministic backbone: SFG-Former, Fourier Residual Refiner, Beta frequency curriculum, PCPSD loss, and frequency-unlocking inference.
- Compact 2D U-Net residual noise predictor with x0/epsilon round-trip and DDIM sampling.
- Farneback motion, forward/backward confidence, lead-time confidence decay, and history-only non-advection evidence.
- Differentiable weak transport plus multi-scale regional budget loss.
- Backtracked clean-residual proximal correction and violation-feedback state.
- CSI, CSI-pool4, CSI-pool16, HSS, LPIPS, SSIM, CRPS, and MAE.
- Unit tests and a real-SEVIR GPU smoke runner.

## Previously verified on `weather-30828` (legacy v10.1 backbone)

- `ruff` and 14 unit tests pass in conda environment `PhyRD`.
- Real-SEVIR `13→12@384` and `13→12@128` forward/backward smoke tests pass on an NVIDIA H800.
- Both smoke tests execute every required metric, including pretrained-AlexNet LPIPS.
- The four event-time splits contain 12,038 / 3,078 / 2,625 / 1,466 events and have zero pairwise overlap.
- A one-step 128 deterministic-to-residual training chain passes with checkpoint protocol validation enabled.

The emitted smoke scores come from untrained networks. They are retained only to prove that metric and sampling code paths execute and must not be reported as model quality.

The SDIR replacement requires a fresh server smoke/DDP probe; the legacy U-Net verification does not validate SDIR.

## Locally verified for v10.2 SDIR

- `5→20` iterative inference returns `[B,20,1,H,W]` for dynamic batch sizes.
- Native SDIR skeleton/residual/PCPSD objective is finite and backpropagates through both SFG-Former and FR-Refiner.
- PCPSD returns zero for identical fields; CPU bf16 autocast forward/backward passes.
- The production-width `hidden=512, heads=4, depth=8, FR=32×8` model completes a small-grid forward/backward and has 33.83M deterministic parameters.
- Python compilation and all 18 YAML/config parses pass.
- CUDA and 8-rank DDP were subsequently verified on `weather-30537` inside `tmux:wzq/sdir-v102`.

## Verified on `weather-30537` for v10.2 SDIR

- Deployment archive SHA256 matched: `fceb48a91ef405796b85a757ce2f58e1ada554de37df99ba61f7bbc198571ea7`.
- Five SDIR model tests passed remotely in 8.62 s.
- Production-width `5→20@128` CUDA bf16 validation passed on H800: 33,828,628 deterministic parameters, 3.620 s training step, 0.256 s frequency-unlocking inference, 1.707 GiB peak at batch 1.
- Gradient L1 was nonzero for both SFG-Former (`2771.515`) and FR-Refiner (`366.256`).
- A real catalog SEVIR `13→12@128` single-GPU step completed with finite loss `1.288655` and 1.709 GiB peak.
- A real DiffCast/PhyDNet-matched HDF5 `5→20@128` same-seed 8-rank DDP step completed with finite loss `1.250592`, `world_size=8`, global batch 8, and 1.839 GiB maximum rank peak at probe batch 1.
- Validation logs contain no Traceback, RuntimeError, NCCL error, NaN, or Inf; all eight H800s returned to 0 MiB after completion.
- Validation artifacts are under `/test1/wzq/PhyRD/artifacts/validation_sdir_20260717_094406` and `/test1/wzq/PhyRD/artifacts/probes/sdir_*`.
- Residual diffusion was not launched.

## Intentionally deferred to evidence phases

- Full official DiffCast reproduction/checkpoint evaluation.
- Trained neural motion alternatives and calibrated robust scales.
- Patch-level risk calibration, external radar datasets, and object metrics.
- Tier A/B baseline training, multi-seed statistics, and report-test evaluation.
- Any claim that PhyRD improves accuracy, probability quality, or physical consistency.

These items are deferred because the user requested a debugged codebase before moving formal training to another server. Their interfaces and Gates remain documented, but no placeholder result is presented as evidence.
