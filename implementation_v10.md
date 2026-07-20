# Implementation Guide — PhyRD-v10

> v10.3 工程决策：主包采用 Route B clean port，确定性 `Pθ` 锁定为 SDIR；当前主协议为冻结 DiffCast HDF5 的 `5→20@128`。官方 DiffCast 与 SDIR 仓库仅放在隔离目录作参考/基线。原 compact 2D U-Net 只保留给 residual diffusion denoiser，不再承担确定性预测。当前 smoke 不等同完整训练或论文 Gate 通过。

> 配套总控：`idea_report_master_v10.md`  
> 配套约束：`user_requirements_v10.md`  
> 状态：设计规范；任何“已跑通”结论必须来自实际日志  
> 原则：先复现并锁定无物理母基线，再逐层增加 motion、训练约束、采样修正和风险校准。

---

# 0. 代码路线与仓库

## 0.1 第一仓库

```bash
git clone https://github.com/DeminYu98/DiffCast.git
cd DiffCast
git checkout -b audit/phyrd-v10
```

官方 DiffCast 是直接母基线，但是否直接作为最终高分辨率主仓，由 Phase 0 的 `ADR-001_CODEBASE_STRATEGY.md` 决定。

## 0.2 Route A / Route B 决策指标

必须实测：

- 官方 smoke test 是否可复现；
- 5→20@128 单 batch 前向、反向和一次采样峰值内存；
- 需要重写的核心文件比例；
- 原仓采样器、backbone 和数据接口的耦合度；
- GPL 发布计划；
- 在 128 协议下 official 与 standardized port 的行为一致性。

不得仅凭“DiffCast 是 baseline”就默认大规模改旧仓。

## 0.3 外部仓库

```text
DiffCast      https://github.com/DeminYu98/DiffCast.git
Earthformer   https://github.com/amazon-science/earth-forecasting-transformer.git
CasCast       https://github.com/OpenEarthLab/CasCast.git
PreDiff       https://github.com/gaozhihan/PreDiff.git
FlowCast      https://github.com/b-rbmp/FlowCast.git
FREUD         https://github.com/CompVis/weather-rf.git
PySTEPS       https://github.com/pySTEPS/pysteps.git
HKO-7         https://github.com/sxjscience/HKO-7.git
```

外部基线独立环境运行，通过 prediction artifact 交换，不相互 import。

---

# 1. 强制控制文件

项目根目录必须包含：

```text
PROTOCOL.yaml
DECISIONS.yaml
BASELINE_REGISTRY.yaml
EXPERIMENT_REGISTRY.csv
GATES.md
CHANGELOG.md
LICENSE_LEDGER.md
```

## 1.1 `PROTOCOL.yaml`

至少记录：

```yaml
version: phyrd-v10.3
dataset: SEVIR
input_frames: 5
output_frames: 20
frame_minutes: 5
resolution: [128, 128]
variable: VIL
split_manifest_sha256: TBD
calibration_split: val_calib
ensemble_size_primary: 10
primary_endpoints:
  - CSI_M
  - CSI_high
  - CRPS
```

## 1.2 `EXPERIMENT_REGISTRY.csv`

字段至少为：

```text
experiment_id,status,phase,git_commit,config_sha,dataset_manifest,
seed,checkpoint_sha,hardware,start_time,end_time,exit_code,
primary_metrics,artifact_path,notes
```

## 1.3 `BASELINE_REGISTRY.yaml`

每个外部方法记录：repo、commit、license、checkpoint、protocol、output domain、ensemble K、NFE、适配脚本。

---

# 2. 推荐目录

```text
project/
├── src/
│   ├── data/
│   │   ├── sevir_dataset.py
│   │   ├── meteonet_dataset.py
│   │   ├── hko7_dataset.py
│   │   ├── synthetic_transport.py
│   │   ├── split_audit.py
│   │   └── manifests.py
│   ├── models/
│   │   ├── deterministic/         # registered deterministic adapters
│   │   │   ├── registry.py
│   │   │   └── sdir_official.py   # official SDIR interface adapter
│   │   ├── residual_diffusion.py
│   │   ├── residual_cfm.py          # 扩展实验
│   │   ├── source_model.py
│   │   └── phyrd.py
│   ├── motion/
│   │   ├── tvl1.py
│   │   ├── rainymotion_adapter.py
│   │   ├── neural_flow_adapter.py
│   │   ├── extrapolation.py
│   │   ├── flow_reliability.py
│   │   └── nonadvective_evidence.py
│   ├── physics/
│   │   ├── vil_proxy.py
│   │   ├── warp.py
│   │   ├── weak_transport.py
│   │   ├── robust_scale.py
│   │   ├── constraint_budget.py
│   │   └── proximal_guidance.py
│   ├── calibration/
│   │   ├── feature_audit.py
│   │   ├── targets.py
│   │   ├── calibrators.py
│   │   └── crossfit.py
│   ├── evaluation/
│   │   ├── deterministic.py
│   │   ├── probabilistic.py
│   │   ├── object_metrics.py
│   │   ├── risk_metrics.py
│   │   ├── bootstrap.py
│   │   └── evaluator.py
│   └── train/
├── tests/
├── configs/
├── manifests/
├── external_baselines/              # gitignored
├── artifacts/
└── docs/
```

---

# 3. 数据契约

## 3.1 张量

```text
x          [B,Tin,1,H,W]
y          [B,Tout,1,H,W]
mu         [B,Tout,1,H,W]
r0         [B,Tout,1,H,W]
flow       [B,Tout-1,2,H,W]
c_flow     [B,Tout-1,H,W]
m_nadv     [B,Tout-1,H,W]
valid_mask [B,Tout-1,H,W]
```

所有模块内部统一 `[B,T,C,H,W]`。空间 flow 顺序明确为 `(dx,dy)` 或 `(vx,vy)`，必须在 `PROTOCOL.yaml` 固定。

## 3.2 时间对齐

若只约束未来内部相邻帧：

```text
q[:, :-1] ↔ q[:, 1:]
flow.shape[1] = Tout−1
```

若纳入“最后观测 → 第一预测”：

```text
q_all = concat(q_last_observed, q_pred)
flow.shape[1] = Tout
```

禁止隐式广播。形状不一致必须立即抛错。

## 3.3 数据域

- SEVIR 主变量命名为 `vil`，不命名为 `dbz`；
- 阈值指标在官方 VIL 编码域计算；
- CRPS/Brier 的域必须在结果表脚注中写明；
- MeteoNet/HKO-7 使用各自单位、阈值和归一化；
- 所有转换函数需要 round-trip 或单调性单测。

## 3.4 切分

主协议最少四组：

```text
train
val_model
val_calib
report_test
```

当前 HDF5 只有 `train/valid/test` 三个源 group，因此 `valid` 必须按冻结的 `sample_id` 奇偶二次划分；不得在看过指标后重新划分：

```text
train       HDF5 group=train
val_model   HDF5 group=valid, sample_id % 2 = 0
val_calib   HDF5 group=valid, sample_id % 2 = 1
report_test HDF5 group=test
```

catalog 与 HDF5 路径必须分别解析；禁止假定二者总在同一个 `root/data` 布局下。

空间预处理不得与时间抽样混写。实现必须支持：

```text
diffcast_bilinear@128     主协议：VIL/255 后 bilinear resize，匹配 DiffCast 有效 wrapper
area@128                  weak-budget 消融
none@384                  后续独立扩展
```

冻结 DiffCast HDF5 的 25 帧样本是当前 SEVIR 主路径：前 5 帧输入、后 20 帧输出。其有效 wrapper 使用 VIL/255 与 `transforms.Resize`；`avg_pool2d` helper 因 `downsample_dict=None` 默认未启用。

---

# 4. 生成基线

## 4.1 确定性分支

```python
train_result = sdir.training_loss(x, y)
# L = L_skeleton + L_residual + 0.01 * L_PCPSD

mu = sdir(x)  # inference: zero condition + frequency unlocking
mu_frozen = mu.detach()
```

实现约束：

- `SFGFormer`：历史帧 token 与未来 coarse-condition token 拼接，使用 scale-adaptive Transformer 和时间/空间 3-D RoPE，输出低频骨架；
- `FourierResidualRefiner`：编码器/解码器中间使用 scale-conditioned FNO，预测高频修正而非最终场；
- `PCPSDLoss`：对 Hann-window 后的径向 PSD 做动态频率遮罩，权重随课程尺度变化；
- 训练尺度 `s~Beta(1,3)`，训练 condition 可由 `Y` 构造，但推理 condition 必须从零开始；
- 推理按 `range(0,W,frequency_stride)` 迭代，每轮输出经低分辨率投影成为下一轮 condition；
- 模型 I/O 固定为 `[B,T,1,H,W]`、`[0,1]`，保持 residual diffusion 接口不变；
- 主配置：`patch=4, hidden=512, heads=4, depth=8, FR channels=32, FR depth=8, stride=16, resolution=128`；
- `13→12@384` 扩展必须单独注册 patch/stride 与实测资源，不能沿用主协议 checkpoint。
- `scripts/launch_ddp8.sh` 默认只启动 SDIR；残差阶段必须显式设置 `RUN_RESIDUAL=1`，当前调试阶段禁止设置。

远端验收统一在 `tmux:wzq` 内执行：

```bash
bash scripts/validate_sdir_remote.sh
# 仅在八卡主机显式启用：
RUN_DDP8=1 bash scripts/validate_sdir_remote.sh
```

脚本会拒绝非 `tmux:wzq` 环境，依次执行模型测试、正式宽度 CUDA bf16 前反向、`5→20@128` frequency-unlocking 推理、真实 SEVIR 单步训练；八卡模式再执行同 seed 的一步 DDP。整个流程不启动 residual diffusion。

2026-07-17 实测：上述流程已在 `weather-30537` 的 `tmux:wzq/sdir-v102` 全部通过。八卡 `5→20@128` 探针记录 `world_size=8, seed=42, global_batch=8, max_rank_peak_memory=1.839 GiB`；这是 batch=1/rank 的正确性 Gate，不是正式训练显存配置。正式配置仍为 batch=8/rank、global batch=64，并需独立记录实际峰值。

固定 deterministic checkpoint、配置 hash 和完整的 `model.deterministic.name/params` protocol。所有物理消融共享。任何旧 backbone checkpoint 必须因 protocol/state-dict 不匹配而失败，不允许静默部分加载。训练目录只保留验证集最优的 `checkpoint_best.pt` 与最新的 `checkpoint_last.pt`。

## 4.2 残差扩散

```python
r0 = y - mu_frozen
rt, noise, t = q_sample(r0)
model_out = denoiser(rt, t, cond=(x, mu_frozen))
r0_pred = predict_x0(rt, model_out, t)
y0_pred = mu_frozen + r0_pred
```

当前正式配置使用 `prediction_type=v`；`epsilon` 仅保留为兼容选项。伪代码中的 `model_out` 由具体参数化决定，不能默认解释为 epsilon。

必须同时支持：

```text
physics_enabled=false
train_physics_only
sampling_guidance_only
train_and_sampling
```

## 4.3 高分辨率扩展可行性

Phase 0/2 必须记录：

- batch=1 峰值显存；
- AMP、gradient checkpointing、activation offload 的收益；
- 直接 pixel-space 与 latent/downsampled residual 方案；
- 训练和采样 wall time。

若启动 384 扩展而 pixel diffusion 无法稳定运行，使用 ADR 记录转向 latent residual；不得把扩展结果写进 `5→20@128` 主表。

---

# 5. Motion 与约束输入

## 5.1 `flow_reliability.py`

```python
c_flow = valid_mask * torch.exp(
    -a * robust_norm(fb_error)
    -b * robust_norm(aperture_error)
    -c * robust_norm(boundary_penalty)
)
```

禁止默认把强度 warping error 直接作为 flow failure，因为它包含真实生消。

## 5.2 `nonadvective_evidence.py`

候选输入：

- 多个 flow 候选下的稳健 warping residual；
- 历史局地强度趋势；
- object area/peak change；
- 历史形态分裂合并证据。

输出：

```python
m_nadv = clamp(score, 0, 1)
```

MVP 可用非学习规则；学习模型只能用历史观测训练，不得使用测试未来。

## 5.3 未来 flow

至少支持：

```text
constant_last
linear_extrapolation
learned_extrapolation
```

首版用 `constant_last`，但必须同时输出 `c_extrap[:,τ]`：

```python
c_flow_future = c_obs * c_extrap_by_lead
```

`c_extrap_by_lead` 由 `val_model` 的历史回代/backtest、多外推器 disagreement 或 motion ensemble 校准，禁止把观测期 `c_obs` 原样复制到所有未来 lead。所有方法在主消融中共享相同 future-flow 设定。

---

## 5.4 `C_flow/M_nadv` 可辨识性接口

D0 数据生成器必须返回：

```python
flow_gt, source_gt, flow_corruption_mask, nonadvective_mask
```

评估器必须分别计算：

```text
AUROC/AUPRC(C_flow vs flow-error mask)
AUROC/AUPRC(M_nadv vs non-advection mask)
cross-AUROC(C_flow vs source mask, M_nadv vs flow-error mask)
lead-time calibration(C_extrap)
```

真实数据的独立佐证接口预留：`motion_disagreement`、`object_track_velocity`、`doppler_or_nwp_wind`、`manual_birth_death_labels`。

# 6. 弱输运实现

## 6.1 MVP

```python
q_adv = warp(q_t, flow_t)
r_local = q_tp1 - q_adv
r_mass = sum_pool(q_tp1, m) - sum_pool(q_adv, m)
```

默认 `kappa=0, source=None`。

## 6.2 容忍预算

```python
tol = tol_by_lead * (1.0 + gamma_nadv * m_nadv)
violation = relu(abs(r_norm) - tol)
```

`tol_by_lead` 和 `gamma_nadv` 只由 `val_model` 选择。

## 6.3 归一化

`MAD` 需按训练集冻结，可按 lead/intensity bucket。部署时 intensity bucket 必须由输入或预测强度决定；评估分层可用真值，但不能喂给模型。

## 6.4 损失

```python
loss_local = masked_huber(violation_local, weight=c_flow)
loss_mass = sum(masked_huber(violation_mass[m], weight=pool(c_flow,m)))
loss_phys = loss_local + alpha_mass * loss_mass
```

## 6.5 `kappa/source` Gate

只有以下条件满足才开放：

- MVP 独立物理指标改善；
- 高值技巧没有系统性下降；
- D0 能恢复已知 source/diffusion；
- source guard 测试通过。

---

# 7. 训练期梯度

训练伪代码：

```python
mu = sdir(x).detach()  # frozen frequency-unlocking inference
r0 = y - mu
rt, noise, t = diffusion.q_sample(r0)
model_out = diffusion.denoise(rt, t, x, mu)
r0_pred = diffusion.predict_x0(rt, model_out, t)
y0_pred = mu + r0_pred

with torch.no_grad():
    flow, c_flow, m_nadv = motion_pipeline(x)

loss_gen = mse(eps_pred, noise)
loss_phys, diag = transport_loss(
    y0_pred, flow.detach(), c_flow.detach(), m_nadv.detach()
)
loss = loss_gen + lambda_schedule(step) * loss_phys
```

调试时必须分别执行：

```python
loss_phys.backward(retain_graph=True)
assert nonzero_grad(diffusion.parameters())
assert zero_grad(motion.parameters())
```

禁止使用：

```python
y0_pred = mu + (y - mu)  # 等于真值，物理损失绕过生成器
```

---

# 8. 推理期近端修正

## 8.1 DDIM/DDPM 通用接口

```python
r0_hat, param_hat = model.predict_clean_and_param(rt, t, cond)
energy = constraint_energy(mu + r0_hat, flow, c_flow, m_nadv)

grad = autograd.grad(energy, r0_hat)[0]
r0_corr = r0_hat - eta_t * lambda_map * clip_norm(grad)

param_corr = sampler.parameter_from_x0(rt, r0_corr, t)
rt_prev = sampler.step(rt, param_corr, t)
```

不能直接修改一个与能量计算图不相连的 `rt_prev`。

## 8.2 Violation-feedback update（默认名称）

```python
# 可作局部 dual-inspired 解释，但默认不宣称全局 primal-dual 收敛
lambda_map = clamp(
    lambda_map + rho * c_flow * violation.detach(),
    0, lambda_max
)
```

## 8.3 Backtracking

若：

- 约束能量上升；
- 修正幅度超过阈值；
- predicted intensity 超出合法域；

则缩小 `eta_t` 或回滚该步。

## 8.4 计算策略

- 只在后 M 步；
- 每 s 步一次；
- 可在 192/96 分辨率计算能量，但最终独立指标在原分辨率评估；
- 报 physics gradient evaluations，不能只报 denoising NFE。

---

# 9. 风险校准

## 9.1 主粒度

默认 patch-level：

```text
16×16 或 32×32 patch × lead time
```

像素级作为次结果。事件级风险可由 patch 风险聚合。

## 9.2 特征审计

`feature_audit.py` 必须把每个特征标为：

```text
input-derived / prediction-derived / target-derived
```

正式校准器只允许前两类。

## 9.3 交叉拟合

```text
val_model: 选生成模型/超参
val_calib: 拟合校准器
```

或按 event 做 K-fold out-of-fold 预测。模型选择和校准评估不可使用同一 in-sample 输出。

## 9.4 类别不平衡

强回波 miss/false alarm 使用 AUPRC、Brier、分层 reliability；报告 prevalence。不能只报 AUROC。

---

# 9.5 最近邻基线的强制实现

必须在同一 backbone、同一 residual representation、同一 sampler 预算下实现：

```text
B0 residual diffusion
B1 train-time fixed weak-transport loss
B2 PreDiff-style knowledge alignment
B3 physics-conditioning-only / Nowcast3D-like 2D control
B4 fixed inference guidance
B5 C_flow-only
B6 M_nadv-only
B7 full PhyRD
```

`B2` 使用相同 transport energy，但不使用 reliability gating 或 non-advection tolerance；`B3` 只把 motion/physics deterministic forecast 作为 condition，采样过程不回传物理梯度。所有组记录实际 NFE、physics gradient evaluations 和 latency。

# 10. 基线适配

统一 artifact：

```text
predictions.zarr
metadata.json
metrics.json
run.log
manifest.yaml
```

`metadata.json` 至少包含：

```json
{
  "model": "FlowCast",
  "commit": "...",
  "checkpoint_sha256": "...",
  "dataset_manifest_sha256": "...",
  "input_frames": 5,
  "output_frames": 20,
  "resolution": [128, 128],
  "ensemble_size": 10,
  "nfe": 10,
  "output_domain": "SEVIR_VIL"
}
```

转换脚本只能做确定性格式、尺寸、单位转换；不得针对某模型调平滑或阈值。

---

# 11. 实验配置

至少：

```text
E001_diffcast_official.yaml
E002_sevir_protocol_baselines.yaml
E010_diffcast_like.yaml
E020_motion_synthetic.yaml
E030_train_physics.yaml
E040_fixed_guidance.yaml
E041_cflow_guidance.yaml
E042_mnadv_tolerance.yaml
E043_dual_guidance.yaml
E050_risk_calibration.yaml
E060_external_dataset.yaml
E070_engine_transfer_cfm.yaml
E080_sevir_3h.yaml
```

核心控制：

```text
compute_matched
random_energy
shuffled_flow
reverse_flow
smoothness_only
oracle_flow
constraint_target_mu_r_full
```

---

# 12. 指标与统计

核心实现必须提供并在统一 VIL `[0,255]` 域报告：`CSI↑`、`CSI-pool4↑`、`CSI-pool16↑`、`HSS↑`、`LPIPS↓`、`SSIM↑`、`CRPS↓`、`MAE↓`。其中 pooled CSI 使用非重叠 max-pool（`kernel=stride=4/16`），CRPS 输入必须含显式 ensemble 轴；单成员 CRPS 会退化为 MAE，应标注为 deterministic sanity check，不能冒充概率评估。

## 12.1 主终点

- CSI-M；
- 高阈值 CSI；
- CRPS 非劣/改善。

## 12.2 独立物理

- centroid displacement；
- component track；
- FSS；
- regional budget；
- D0 true flow/source error。

## 12.3 风险

- AUPRC、Brier、coverage-risk；
- event/patch block bootstrap。

## 12.4 seeds

核心模型和关键消融至少 3 seeds。大型外部单 checkpoint 需明确标记。

---

# 13. 单元测试

```text
test_shapes_and_units.py
test_split_no_leakage.py
test_vil_proxy_monotonic.py
test_warp_translation.py
test_flow_reliability_vs_nadv.py
test_transport_analytic.py
test_physics_gradient.py
test_proximal_guidance_monotonic.py
test_sampler_reparameterization.py
test_source_guard.py
test_calibration_feature_leakage.py
test_metrics_reference.py
test_flow_confidence_identifiability.py
test_nonadvective_identifiability.py
test_future_flow_confidence_decay.py
test_convective_regime_split.py
test_preddiff_style_control_equivalence.py
test_physics_conditioning_only_control.py
```

关键断言：

1. correct flow residual < wrong flow residual；
2. intensity growth with correct flow 应提高 `M_nadv`，但不必强制降低 `C_flow`；
3. proximal correction 后 energy 不增；
4. `r0_corr → parameter_corr → x0_recovered` 一致；
5. `L_phys` 对生成器梯度非零；
6. 任何 target-derived risk feature 使正式运行失败。

---

# 14. Gate

新增不可跳过 Gate：

- **G4a Identifiability**：D0 上 `C_flow` 与 `M_nadv` 对各自目标有显著优于随机的 AUROC/AUPRC，且交叉混淆可解释；
- **G4b Future reliability**：`C_extrap` 随 lead time 的可靠性曲线经 `val_model` 校准；
- **G6a Nearest-neighbor attribution**：B2/B3/B4 在相同计算预算下完成，PhyRD 的增益不能仅由一般 knowledge guidance 或 physics conditioning 解释；
- **G8a Convective negative transfer**：新生/消散子集不得出现超过预注册容忍界的高阈值召回或 CRPS 退化。


- **G-00**：官方仓可运行，ADR 路线有实测依据；
- **G-01**：数据、split、指标、四组数据隔离正确；
- **G-02**：无物理 residual baseline 稳定且 ensemble 不塌缩；
- **G-03**：D0 与 motion 解耦测试通过；
- **G-04**：训练约束改善独立物理指标且主终点不灾难退化；
- **G-05**：近端修正单调、稳定，收益不只来自更多计算；
- **G-06**：风险增量相对 `U_ens` 成立，否则降级；
- **G-07**：SEVIR + 一个外部域完成统计闭环；
- **G-08**：论文主张、表格和代码注册表一致。

未通过 evidence Gate 不得写性能结论。未完整训练时允许为接口联调继续实现下一阶段，但必须将结果标记为 `code-ready only`，不得把 smoke test 记为 evidence Gate 通过。
