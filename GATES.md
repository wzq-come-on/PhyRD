# PhyRD v10.3 Gates

## G-SDIR — deterministic backbone replacement

- Native SDIR training loss is finite and reaches SFG-Former and FR-Refiner parameters.
- Frequency-unlocking inference starts from an all-zero future condition for the primary `5→20@128` protocol.
- PCPSD is zero for identical fields and finite under bf16 training.
- Legacy deterministic U-Net checkpoints fail protocol validation.
- Batch=1 CUDA forward/backward and same-seed 8-rank DDP probes pass before formal training.
- Standalone SDIR is evaluated on the exact fair PhyDNet/DiffCast `5→20` test set.

Current status: local CPU model tests pass; CUDA/DDP items are pending because the configured weather servers refused SSH connections on 2026-07-17.

Gate 状态分成 `code-ready` 与 `evidence`，两者不得混写。

| Gate | 类型 | 当前状态 | 验收条件 |
|---|---|---|---|
| C-00 | code-ready | passed | `ruff`、14 项单元测试、配置解析与包导入均通过 |
| C-01 | code-ready | passed | 冻结 DiffCast HDF5 `5→20@128` 的读取、VIL 域与空间预处理已验证；valid 二次划分需单独登记 |
| C-02 | code-ready | passed | forward/backward、物理梯度、proximal 回溯与 DDIM 重参数化通过 |
| C-03 | code-ready | passed | CSI/pooled CSI/HSS/LPIPS/SSIM/CRPS/MAE 均在 GPU smoke 中执行 |
| C-04 | code-ready | passed | `weather-30828` 的 `tmux:wzq` 内两种分辨率 smoke 均返回 0 |
| G-00…G-08 | evidence | not_started | 按 MASTER 文档完成训练、消融、统计和冻结测试 |

`C-*` 通过只表示工程调用链已调通，不能作为任何模型优于基线的证据。

验收环境、切分统计和 smoke 诊断详见 `docs/SERVER_VALIDATION.md`。`G-00…G-08` 必须等正式训练迁移到目标训练服务器后再推进。
