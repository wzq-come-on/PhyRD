# 用户需求与约束记录 — PhyRD-v10

> 本文件只保留当前有效决策。与旧 v5/v6/v7 冲突时，以本文件和 `idea_report_master_v10.md` 为准。

> 2026-07-15 补充：当前交付先完成 research-ready MVP 与服务器真实 SEVIR 单批调试，不在本阶段进行正式训练。任何删除文件/目录、清理远端内容、移除或覆盖式重建 conda 环境的命令，必须事先询问用户并得到明确同意。
> 服务器 conda 与 pip 依赖安装使用清华镜像源。
> 当前主任务采用冻结 DiffCast HDF5 的 `5→20@128`：VIL/255 后 bilinear resize；`13→12@384` 改为后续扩展。两种协议结果禁止混表。
> 2026-07-17 变更：确定性 backbone 固定为完整 SDIR，替换临时 compact 2D U-Net；旧确定性 checkpoint 作废。SDIR 论文与官方仓为 `arXiv:2606.02661` / `RuntimeWarning/SDIR`。

## 1. 目标

- 雷达降水短临预报；
- 主任务：SEVIR VIL `5→20@128`，未来 100 min；
- 扩展：SEVIR `5→36@128` 或独立 `13→12@384`；
- 方法：SDIR 确定性趋势 `μ` + 残差生成 `r` + 可靠性受控弱输运约束；
- 目标：改善强回波、运动/空间结构和长 lead，同时不明显损害概率质量与效率。

## 1.1 创新边界（强制）

- 不宣称首次在 nowcasting diffusion 中加入知识/物理引导；
- 不把“物理确定性预测 + residual diffusion”本身作为 hero；
- 主创新锁定为：**uncertain physical operator 下的 reliability-gated guidance**；
- 必须与 PreDiff-style 同架构知识引导和 physics-conditioning-only/Nowcast3D-like 2D 对照比较；
- 若最近邻对照达到相同结果，必须收缩或放弃对应贡献。

## 2. 当前核心变量

| 变量 | 有效定义 | 禁止定义 |
|---|---|---|
| `U_ens` | ensemble spread | 不等同 epistemic/aleatoric 完整分解 |
| `R_phys` | 标准化弱输运约束违背 | 不称为概率 uncertainty |
| `C_flow` | `C_obs × C_extrap(lead)` 的 motion operator reliability | 不把观测期置信度无衰减复制到未来；不使用真实生消误差简单代替坏 flow |
| `M_nadv` | non-advection evidence | 不称为坏 flow 概率 |
| `P_err` | 独立校准切分上拟合的风险概率 | 不由测试标签拟合 |
| `λ` | 约束预算的 projected dual weight | 不使用无依据的 `R_phys/(R_phys+U_ens)` |

## 3. 物理边界

- VIL 不是 dBZ、雨强或守恒质量；
- 使用固定单调 VIL proxy；
- 主约束是可微 warping + 多尺度区域收支；
- MVP：`κ=0,S=0`；
- bounded `κ/S` 只能在 MVP 通过 Gate 后启用；
- 论文写 weak transport consistency，不写 exact conservation。

## 4. Motion 决策

- 优先雷达域 TV-L1/rainymotion 或雷达适配网络；
- natural-image RAFT 只作消融；
- `C_flow` 由 forward-backward、aperture/texture、boundary 等构成；
- warping intensity error 主要用于 `M_nadv`，避免把真实新生误判为坏 flow；
- future flow 默认常速，其他方式消融。

## 5. 训练与采样

### 5.0 SDIR 确定性骨干（强制）

- 必须包含 SFG-Former、FR-Refiner、frequency-scale conditioning 和 PCPSD loss；
- 必须使用原生 SDIR 三项训练目标，不得退回单一 Smooth-L1；
- 训练 target-derived coarse condition 不得进入推理；推理必须从零 condition 逐频率迭代；
- `5→20@128` 是当前唯一主协议，用于与 PhyDNet/DiffCast 同数据公平对比；`13→12@384` 是后续独立扩展；
- checkpoint 必须记录完整的 `model.deterministic.name/params`，当前唯一正式名称为 `sdir_official`；旧 2D-U-Net 与自建 SDIR checkpoint 禁止复用；
- residual diffusion 只能消费冻结 SDIR 输出，修改确定性骨干不得破坏后续 test 接口；
- 完整训练前先做 batch=1 前向、反向、PCPSD 有限值、DDP 和显存探针。
- 当前阶段允许按 B0→B7 顺序训练和测试 residual diffusion；每个阶段必须保留独立配置、checkpoint 和日志，不得跳过同协议消融。
- 2026-07-17 `weather-30537/tmux:wzq` 的单卡与八卡 SDIR code-ready 探针已通过；正式 5→20 训练和 report-test 指标仍须另行完成，禁止把一步 loss 写成效果提升。

- 物理训练损失必须作用于模型预测 `r̂0` 对应的 `μ+r̂0`；
- 禁止在 `μ+(Y−μ)=Y` 上计算假物理损失；
- 采样引导必须按 `r̂0 correction → sampler parameter recomputation → sampler step` 实现；
- 必须有 sampler reparameterization 和 energy monotonicity 测试；
- `λ` 和 motion 默认 stop-gradient。

## 6. 风险校准

- 主粒度为 patch-level；
- 风险特征只能来自输入或预测；
- 训练、模型验证、风险校准、报告测试四组隔离；
- 强回波风险以 AUPRC/Brier 为主，AUROC 为辅；
- 若 `R_phys/C_flow/M_nadv` 相对 `U_ens` 无增量，风险贡献降级。

## 7. 代码基线

第一步必须下载：

```bash
git clone https://github.com/DeminYu98/DiffCast.git
```

但 Phase 0 需通过实测决定：

- Route A：直接 DiffCast GPL fork；
- Route B：标准协议 port，同时独立保留 official DiffCast 复现。

当前主协议不依赖 384 可行性；若启动 `13→12@384` 扩展，必须先完成独立资源审计。

## 8. 外部仓库

```text
Earthformer: https://github.com/amazon-science/earth-forecasting-transformer.git
CasCast:     https://github.com/OpenEarthLab/CasCast.git
PreDiff:     https://github.com/gaozhihan/PreDiff.git
FlowCast:    https://github.com/b-rbmp/FlowCast.git
FREUD:       https://github.com/CompVis/weather-rf.git
PySTEPS:     https://github.com/pySTEPS/pysteps.git
HKO-7:       https://github.com/sxjscience/HKO-7.git
```

- DiffCast：GPL-3.0；
- Earthformer/PreDiff/FlowCast：按官方 Apache-2.0 归属要求；
- FREUD：个人与科研非商业许可证；
- CasCast 页面未显示明确 LICENSE 时，不复制核心源码；
- 外部方法独立环境运行。

## 9. 实验优先级

### 核心必须

1. D0 synthetic；
2. SEVIR 完整主表；
3. 一个独立真实雷达域，优先 MeteoNet，HKO-7 可替代；
4. Tier A 同协议基线；
5. 机制阶梯、负控制、概率质量、效率；
6. 3 seeds 与 event bootstrap；
7. D0 上 `C_flow/M_nadv` 可辨识性；
8. 新生/消散/增强/衰减等事件分层与负迁移检查；
9. PreDiff-style 与 physics-conditioning-only 最近邻对照。

### 增强项

- 第二外部真实域；
- residual CFM 引擎迁移；
- SEVIR-3h；
- 数据效率；
- 缺帧/噪声/遮挡。

## 10. 基线准入

### Tier A 主表必须

- Persistence；
- optical flow/rainymotion；
- PySTEPS；
- SimVP/Earthformer；
- SDIR standalone deterministic；
- backbone-matched DiffCast-like；
- compute-matched control；
- PreDiff-style same-backbone knowledge alignment；
- physics-conditioning-only / Nowcast3D-like 2D control；
- fixed physics-guided sampling；
- PhyRD。

### Tier B 尽量重跑

PreDiff、CasCast、FlowCast、FREUD、PostCast/后续核验通过的新增强模型。只有同协议或明确适配后进统一主表；否则进背景表。

## 11. 主指标

- CSI、CSI-pool4、CSI-pool16、HSS；
- LPIPS、SSIM、MAE；
- CSI-M；
- 高阈值 CSI；
- CRPS；
- FSS；
- centroid/component track；
- Brier/reliability/spread-skill；
- p50/p95 latency、NFE、physics gradient evaluations、memory。

测试前冻结 CRPS 非劣界和高阈值选择规则。

## 12. Codex 行为约束

- 严格按 Phase/Gate；
- 一次一个可审查 patch；
- 不实际运行不声称成功；
- 不伪造 checkpoint、数据和指标；
- 不混协议；
- 不使用测试集选参或校准；
- 不删除失败实验；
- 不把逐像素 violation-feedback 默认称为已证明收敛的 primal-dual algorithm；
- 不在没有独立标签/佐证时声称 `C_flow/M_nadv` 完全分离真实物理原因；
- 必须报告新生/消散子集，不能只展示平移型成功案例；
- 修改后同步更新机器可读注册表；
- 文档冲突时停止并报告；
- 实验证据否定假设时，允许降级贡献而不是强行维护故事。

## 13. 已废止旧决策

| 旧决策 | 状态 |
|---|---|
| `U_total=U_ens+βU_phys` | 废止 |
| `λ∝R_phys/(R_phys+U_ens)` | 废止 |
| warping error 全部代表坏 flow | 废止：拆为 `C_flow` 与 `M_nadv` |
| 默认 advection-diffusion-source 全开 | 废止：MVP `κ=0,S=0` |
| 直接修改 `r_{t-1}` 的含糊采样公式 | 废止：修正 `r̂0` 后重参数化 |
| 主风险默认像素级 | 改为 patch-level 主结果 |
| D0+SEVIR+MeteoNet+HKO-7+3h 全部才算完成 | 废止：改为核心/扩展证据梯度 |
| 所有 SOTA 必须塞入统一主表 | 改为 Tier A/B/C 准入 |
| 主开发路线无条件锁死 DiffCast fork | 改为 Phase 0 ADR 实测决策 |


## 14. 本轮新增废止决策

- 废止“PreDiff/physics-conditioning-only 只是可选 Tier B”的安排：它们现在是同架构 Tier A 归因基线；
- 废止把 `projected dual ascent` 作为默认方法名：改称 violation-feedback adaptive guidance，除非补足严格局部约束理论；
- 废止把历史 `C_flow` 直接复制到所有未来帧：必须乘 lead-dependent `C_extrap`；
- 废止只用整体测试集证明物理引导有效：必须报告新生、消散、增强、衰减等子集；
- 废止仅凭热图声称 `C_flow/M_nadv` 解耦：必须在 D0 上进行定量识别与交叉混淆实验。
