# Changelog

## 2026-07-21 - repository layout and environment cleanup

- Split configurations into `configs/active/5to20/`, `configs/diagnostics/`, and `configs/archive/`.
- Consolidated evaluation entry points under `scripts/evaluation/` with one shared protocol evaluator.
- Pinned the validated `PhyRD` and production `sdir` environment versions in `environment.yml` and `environment-sdir.yml`.
- Marked superseded registry runs E-005 and E-009 as stopped.

## 2026-07-20 — v10.3 matched DiffCast primary protocol

- Promoted frozen DiffCast HDF5 `5→20@128` to the active PhyRD protocol; retained `13→12@384` only as a separate extension.
- Registered `train`, deterministic `valid → val_model/val_calib`, and `test → report_test` partitions in `PROTOCOL.yaml`.
- Added HDF5 loader support for `val_model`, `val_calib`, and `report_test` logical splits.
- Added deployment-feature risk artifacts and independent `P_err` logistic calibration on `val_calib`.
- Kept `prediction_type=v` as the formal residual-diffusion parameterization and synchronized the v10 pseudocode.
- Started the matched `5→20@128` B1 weak-transport training run on `weather-30842`; it is development evidence until the frozen split and full evaluation complete.

## 2026-07-19 — deterministic registry and best/last checkpoints

- Consolidated deterministic adapters under `src/phyrd/models/deterministic/` with lazy config-driven registration.
- Removed the retired native SDIR implementation from the active package; official SDIR is now `sdir_official` only.
- Replaced periodic step checkpoints with atomic `checkpoint_last.pt` and validation-selected `checkpoint_best.pt`.
- Added validation splits to formal configs and reject non-empty artifact directories by default to prevent silent overwrite.
- Archived duplicate remote root entry points and deployment files under `legacy/`; moved the CUDA wheel under `vendor/wheels/`.

## 2026-07-17 — v10.2 SDIR deterministic backbone

- Replaced the legacy compact 2D U-Net deterministic forecast with full SDIR.
- Added SFG-Former, scale-conditioned Fourier residual refinement, PCPSD loss, Beta frequency curriculum, and iterative frequency-unlocking inference.
- Updated deterministic training from Smooth-L1 to the native three-term SDIR objective.
- Added checkpoint protocol identity and retired legacy deterministic checkpoints.
- Updated all active configurations and the three v10 design/requirements documents.
- Added tmux-enforced CUDA/real-SEVIR/DDP validation scripts and a deterministic-only launch guard.
- Passed H800 bf16, real-SEVIR single-step, and same-seed 8-rank DDP SDIR validation on weather-30537; residual diffusion remained disabled.

## 2026-07-15 — v10.1 engineering freeze

- 选择 Route B clean port；隔离保留官方 DiffCast GPL 基线。
- 冻结 SEVIR `13→12@384` 中心窗口与四段 event-time split。
- 明确 CSI-pool4/16、LPIPS、SSIM、CRPS、MAE 的实现契约。
- 将 code-ready smoke Gate 与论文 evidence Gate 分离。
- 增加“所有删除操作先询问用户”的强制约束。
- 将空间协议拆为 canonical `none@384`、DiffCast-compatible `diffcast_bilinear@128` 和保守消融 `area@128`；三者禁止混表。
- 在 `weather-30828` 的 `tmux:wzq` 内完成真实 SEVIR 的 384/128 GPU smoke、切分泄漏审计和 128 两阶段单步训练链。
- 固化 `PhyRD` conda 环境（Python 3.11、PyTorch 2.4.1+cu121），安装过程使用清华 conda/PyPI 镜像。
- C-00 至 C-04 全部通过；论文证据 Gate 保持未开始。
- 在 `weather-30537` 使用清华镜像新建 `PhyRD` 环境，并新增启动前 GPU 温度/占用筛选的 8 GPU × 8 seed 两阶段训练编排。
- 根据用户澄清，将正式运行更正为同一 seed 的 8-rank NCCL DDP；增加 `DistributedSampler`、rank-0 checkpoint、BF16、epoch 训练长度、CUDA 峰值与吞吐记录。
- DDP 实测冻结 deterministic 每卡 batch 64、residual 每卡 batch 32；正式运行每卡约保留 10.8–11.0 GiB，计算利用率峰值 84%。
