# PhyRD v10.3

PhyRD is organized as a deterministic backbone plus a config-selected probabilistic model. Deterministic adapters live under `src/phyrd/models/deterministic/`; probabilistic models and their components live under `src/phyrd/models/probabilistic/`. The pair is constructed through the shared factory/composer, so the trainer does not need a branch for every new model. Upstream source trees stay under `third_party/`.

The deterministic model selection contract is:

```yaml
model:
  deterministic:
    name: sdir_official
    params:
      patch_size: 4
      model_resolution: 128
```

New deterministic backbones are added to the registry described in `src/phyrd/models/deterministic/README.md`; training and evaluation code do not need backbone-specific branches.

The corresponding probabilistic contract is:

```yaml
model:
  deterministic:
    name: sdir_official
    params: {model_resolution: 128}
  probabilistic:
    name: residual_diffusion
    params:
      prediction_type: v
    extensions: []
```

Each model is a package when it has multiple components. For example,
`probabilistic/residual_diffusion/` contains the denoiser, diffusion schedule,
sampler, and model adapter. Physics and calibration hooks belong inside the
probabilistic model that uses them; they are optional extensions, not global
assumptions.

Official SDIR additionally requires a CUDA/torch-matched `flash-attn` wheel. Server wheels belong under `vendor/wheels/`, not in the project root.

## Data contract

The primary protocol uses the frozen DiffCast SEVIR HDF5: five historical frames predict twenty future frames (`5→20@128`, 100 min horizon). Model tensors are `[B,T,1,H,W]` in `[0,1]`; categorical scores are computed after conversion to official VIL encoding `[0,255]`.

Spatial preprocessing is explicit. `diffcast_bilinear@128` is the primary protocol and matches DiffCast's effective SEVIR wrapper (`VIL/255` followed by `transforms.Resize`). `area@128` is a conservative pooling ablation; `13→12@384` is a separately registered extension. Metrics from these protocols must never be mixed in one comparison table.

The loader accepts the user-facing root and separately resolves the catalog and HDF5 tree. On `weather-30828`:

```text
catalog: /test1/wzq/data/sevir/CATALOG.csv
HDF5:    /test1/wzq/data/sevir/OpenScienceLab___SEVIR/raw/SEVIR/data
```

## Environment

All server commands are run inside `tmux:wzq`, window `phyrd-codex`. The reproducible bootstrap command is:

```bash
bash scripts/bootstrap_env.sh
```

The script creates `PhyRD` only when it does not already exist and uses the Tsinghua conda/PyPI mirrors for dependency installation. It never deletes an environment or file. The active remote B1 launcher uses the separately pinned Python 3.10 environment described in `environment-sdir.yml`; `environment.yml` is the Python 3.11 package environment for the main project.

The local-to-GitHub-to-server synchronization contract is documented in
`docs/GIT_WORKFLOW.md`. Git tracks code and experiment definitions only;
server data, checkpoints, and run outputs remain local to each server.

## Primary protocol validation

```bash
bash scripts/validate_sdir_remote.sh
```

For the formal primary runs, use `configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml` followed by `configs/active/5to20/train_ddp8_residual_diffcast_5to20_vpred_b1_formal_split_seed42.yaml`. Diagnostic probes and smoke configs live under `configs/diagnostics/`; superseded experiments live under `configs/archive/`.

This runs the primary `5→20@128` chain, including model forward/backward, motion/reliability, weak-transport and proximal-gradient paths, and evaluates CSI, CSI-pool4, CSI-pool16, HSS, LPIPS, SSIM, CRPS, and MAE. It is a code-readiness check, not a benchmark.

The SDIR-specific remote gate must run inside `tmux:wzq`:

```bash
bash scripts/validate_sdir_remote.sh
```

On an eight-GPU host, use `RUN_DDP8=1`; the script still runs deterministic SDIR only.

The primary `5→20@128` server validation and one-step two-stage training chain have passed. The valid-group `val_model/val_calib` partition is a separately registered protocol requirement. Exact engineering evidence is recorded in `docs/SERVER_VALIDATION.md`; smoke metric values are deliberately not presented as performance results.

## Training and evaluation

```bash
conda run -n PhyRD python scripts/train.py --config configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml
conda run -n PhyRD python scripts/train.py --config configs/active/5to20/train_ddp8_residual_diffcast_5to20_vpred_b1_formal_split_seed42.yaml
conda run -n PhyRD python scripts/evaluate.py --predictions predictions.npz --require-lpips

# HDF5 protocol evaluation (the same entry point dispatches all evaluators)
python -m scripts.evaluate --mode protocol --protocol 5to20 --model sdir \
  --checkpoint /path/to/checkpoint.pt --config /path/to/config.yaml \
  --data-path /path/to/sevir.h5 --output /path/to/report.json
```

Training is explicitly two-stage: first train the deterministic trend, then point the residual config at that frozen checkpoint. The residual stage fails loudly if the deterministic checkpoint is absent, preventing accidental diffusion training around a random frozen trend. Formal training must freeze the HDF5 valid-group partition, run multiple seeds, and keep `val_model`, `val_calib`, and `report_test` isolated as specified in `PROTOCOL.yaml`.

Formal training writes `checkpoints/checkpoint_last.pt` and
`checkpoints/checkpoint_best.pt`; the latter is updated only when the configured
validation loss improves. Every new run also stores a full
`config_snapshot.yaml`. New experiment paths should use
`artifacts/experiments/<deterministic>_<probabilistic>/YYYYMMDD_HHMMSS/`.
Existing run directories are rejected by default, preventing a new launch from
silently overwriting an earlier experiment. Old v10/v10.3 artifact layouts are
kept unchanged for reproducibility.

## Multi-GPU DDP training

The production launcher runs one seed with eight synchronous PyTorch DDP ranks. `DistributedSampler` gives each rank a disjoint shard, NCCL averages gradients, and only rank 0 writes logs and checkpoints. It runs SDIR deterministic training only by default; residual diffusion requires an explicit `RUN_RESIDUAL=1` opt-in.

```bash
conda run --no-capture-output -n PhyRD bash scripts/launch_ddp8.sh
```

The registered SDIR run uses seed 42 across GPUs 0–7 and starts with batch 8 per rank (global batch 64) for 50 epochs. This batch is provisional until the new SDIR CUDA memory probe passes; legacy U-Net memory measurements must not be reused. Artifacts are under `artifacts/experiments/phyrd_sdir_13to12_ddp8_seed42/`.

`scripts/launch_multigpu.py` remains available only for intentionally independent multi-seed experiments; it is not the production DDP launcher.

## Licensing

The main package is an independent implementation. Official DiffCast is retained under `external_baselines/DiffCast` as an isolated GPL-3.0 baseline and is neither imported nor copied into `src/phyrd`.
