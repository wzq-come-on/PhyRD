# Server Validation Record — legacy v10.1

> v10.3 update (2026-07-20): the `13→12` smoke records below are historical engineering checks. Current formal runs use the frozen DiffCast HDF5 `5→20@128` protocol and its registered `valid → val_model/val_calib` partition.

> The deterministic path validated here predates the SDIR replacement; v10.2 requires fresh CUDA and DDP validation.

## v10.2 SDIR addendum — weather-30537

The fresh SDIR validation was run inside `tmux:wzq`, window `sdir-v102`, on eight NVIDIA H800 GPUs. The deployment hash matched the local archive. Model tests, production-width CUDA bf16 forward/backward, frequency-unlocking inference, one real catalog-SEVIR step, and one real DiffCast-HDF5 eight-rank synchronized DDP step all passed. The DDP checkpoint/run summary records `deterministic_backbone: sdir`, the full SDIR configuration, `seed: 42`, and `world_size: 8`. No residual diffusion command was launched.

Authoritative artifact root:

```text
/test1/wzq/PhyRD/artifacts/validation_sdir_20260717_094406
/test1/wzq/PhyRD/artifacts/probes/sdir_deterministic_13to12_ddp8
/test1/wzq/PhyRD/artifacts/probes/sdir_diffcast_5to20_ddp8_seed42
```

## Scope

This record covers engineering readiness only. Every command was run on `weather-30828` inside tmux session `wzq`. No formal training or baseline comparison was performed, and no smoke score is evidence of model quality.

## Environment

- Project: `/test1/wzq/PhyRD`
- Conda environment: `PhyRD`
- Python: 3.11.15
- PyTorch: 2.4.1+cu121
- CUDA runtime: 12.1
- Hardware: 4 × NVIDIA H800 80 GB; smoke used one GPU
- Install sources: Tsinghua conda and PyPI mirrors
- SEVIR catalog: `/test1/wzq/data/sevir/CATALOG.csv`
- SEVIR HDF5 root: `/test1/wzq/data/sevir/OpenScienceLab___SEVIR/raw/SEVIR/data`

The environment history, pip freeze and hardware inventory are stored in `artifacts/server/`.

## Passed checks

1. `ruff` passes and all 14 unit tests pass.
2. Native 384 smoke reads event `R19010201527419` and produces `[1,13,1,384,384] → [1,12,1,384,384]`.
3. DiffCast-spatial 128 smoke keeps the same 13→12 time protocol and produces `[1,13,1,128,128] → [1,12,1,128,128]` using non-antialiased bilinear resize after VIL normalization.
4. Both smokes pass backward gradients, weak-transport loss, proximal energy descent, DDIM x0/epsilon round-trip, and all requested metrics: CSI, CSI-pool4, CSI-pool16, HSS, LPIPS, SSIM, CRPS and MAE.
5. The 128 two-stage pilot passes one deterministic optimizer step and one residual optimizer step using the frozen deterministic checkpoint.
6. Checkpoint loading verifies the serialized protocol, preventing accidental 128/384 mixing.

The 128 smoke used 0.1123 GiB peak GPU memory and completed in 3.42 seconds. Its proximal diagnostic decreased from 1.75091 to 1.74844, and the DDIM round-trip maximum error was `4.77e-7`. These are implementation diagnostics, not research results.

## Split audit

| Split | Events | First UTC event | Last UTC event |
|---|---:|---|---|
| train | 12,038 | 2017-06-13 | 2018-12-30 |
| val_model | 3,078 | 2019-01-02 | 2019-05-31 |
| val_calib | 2,625 | 2019-06-01 | 2019-09-30 |
| report_test | 1,466 | 2019-10-01 | 2019-11-30 |

All six pairwise overlap counts are zero. The machine-readable audit is `artifacts/server/sevir_split_audit.json`.

## Deferred evidence

Formal multi-seed training, official DiffCast reproduction, baseline comparison, calibration, ablation, statistical testing and frozen report-test evaluation remain under Gates G-00 through G-08. They should run on the user's later training server without changing the frozen temporal, spatial, split or metric contracts.
