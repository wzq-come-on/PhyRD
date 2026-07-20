# PhyRD-v10 MASTER：置信度受控的弱输运残差生成用于可靠降水短临预报

> 英文暂名：**PhyRD — Reliability-aware Weak-Transport Guidance for Residual Generative Precipitation Nowcasting**  
> 文档级别：MASTER，总控文档；研究、实验、工程和 Codex 执行均以本文件为最高优先级  
> 当前状态：**设计冻结候选版，尚未通过任何实证 Gate，不得写成“方法已验证”**  
> 主协议：DiffCast-compatible SEVIR VIL，5 帧历史 → 20 帧未来，128×128，5 min/帧，未来 100 min  
> 核心边界：VIL 不是严格守恒质量；物理残差是约束违背分数，不是天然概率不确定性  
> 仓库与论文状态核验日期：2026-07-14

## v10.3 主协议变更（2026-07-20）

当前主任务正式切换为 **DiffCast-compatible SEVIR VIL `5→20@128`**。该协议直接使用冻结的 25 帧 HDF5 样本：前 5 帧为历史，后 20 帧为未来，VIL 先除以 255 再 bilinear resize 到 `128×128`。`13→12@384` 保留为后续高分辨率扩展，不进入当前主表，也不得与 `5→20@128` 数字混表。

主协议的数据隔离改为：`train` HDF5 group 用于训练；`valid` group 按冻结的 `sample_id` 奇偶划分为 `val_model` 与 `val_calib`；`test` group 仅作为 `report_test`。校准器只在 `val_calib` 拟合，所有模型选择、物理超参和采样超参只使用 `val_model`。

本节覆盖全文此前把 `13→12@384` 称为当前 canonical 的表述；历史段落仍保留其决策背景，但不再定义当前执行协议。

## v10.2 确定性骨干变更（2026-07-17）

本节覆盖 v10.1 中未锁定或由工程 MVP 临时采用的确定性网络方案：

1. `Pθ` 正式锁定为 **SDIR（Spectral-Decoupled Iterative Refinement）**，不再使用把时间维压为通道的 compact 2D U-Net 作为确定性趋势网络。
2. SDIR 由 **SFG-Former**（全局低频天气尺度骨架）、**FR-Refiner**（尺度条件 Fourier neural operator 高频残差）和 **PCPSD** 动态频谱损失构成；不得删去任一组件后仍把模型记为完整 SDIR。
3. 训练采用 target-derived coarse-frequency curriculum；该条件只用于训练构造监督课程。推理必须从全零未来条件开始，按注册的 `frequency_stride` 逐步解锁频率，禁止读取未来真值。
4. SDIR 输出成为冻结趋势 `μ=Pθ(X)`，后续 residual diffusion 仍建模 `r=Y-stopgrad(μ)`；因此更换确定性骨干不改变 PhyRD 的可靠性门控弱输运主创新边界。
5. v10.1 compact 2D U-Net checkpoint 与官方 SDIR 不兼容，禁止续训或用于 residual 阶段；新 checkpoint protocol 必须记录完整的 `model.deterministic.name/params`，当前 name 为 `sdir_official`。
6. 当前主验证与 PhyDNet/DiffCast 完全同协议，固定为 `5→20@128`；`13→12@384` 仅作为后续独立扩展，两种协议禁止混表。
7. 此变更是结构决策，不构成性能提升证据。必须重新训练并按 CSI、CSI-pool4、CSI-pool16、HSS、LPIPS、SSIM、CRPS、MAE 验证。

工程状态补充：2026-07-17 已在 `weather-30537/tmux:wzq` 通过 H800 bf16、真实 SEVIR 单步与同 seed 八卡 DDP code-ready Gate；该结果只证明实现与分布式链路可运行，不证明 SDIR 相对 PhyDNet 的性能提升。

论文与官方实现：`https://arxiv.org/abs/2606.02661`，`https://github.com/RuntimeWarning/SDIR`。

## v10.1 工程冻结补充（2026-07-15）

本节用于消除“论文完整证据链”和“当前可运行交付”之间的范围混淆；若与后文的执行措辞冲突，以本节为准。

1. 当前交付目标是 **research-ready MVP**：代码、数据契约、训练/采样调用链、指标和真实 SEVIR 单批 smoke test 可运行；不把未进行的完整训练写成已验证结果。
2. 代码路线选择 **Route B clean port**。官方 DiffCast 固定在 `external_baselines/DiffCast`，只作为隔离的 GPL-3.0 基线，不向主包复制源码。原因是官方协议为 `5→20@128`、环境为 Python 3.8/PyTorch 1.12，并将时间维压入 2D 通道；直接扩成 `13→12@384` 会同时改变协议、显存和维护边界。
3. Gate 分为两类：`code-ready` Gate 可由单元测试、合成数据和真实单批调试通过；`evidence` Gate 只有完整训练、多个 seed 和冻结测试评估后才能通过。前者允许继续搭建后续模块，后者未通过时禁止写论文性能结论。
4. 主协议固定为官方 DiffCast HDF5 的 25 帧样本，前 5 帧历史、后 20 帧未来；不得在主表中混入任意重切的滑窗或 `13→12` 样本。
5. 本地数据参数分成 `catalog_path` 与 `data_root`。当前服务器 catalog 为 `/test1/wzq/data/sevir/CATALOG.csv`，HDF5 根目录为 `/test1/wzq/data/sevir/OpenScienceLab___SEVIR/raw/SEVIR/data`，加载器必须解析并打印最终路径。
6. 所有删除文件、目录、远端内容或 conda 环境的命令，执行前必须取得用户明确确认。
7. 当前主表固定为 `5→20@128`：先 `VIL/255`、再 bilinear resize 到 `128×128`。`13→12@384` 与 `area@128` 都是单独注册的扩展/消融；任何结果必须同时写明 native resolution、model resolution、时间长度与 preprocess。

---

# 0. 本轮自检 Loop 的结论

本版经过五轮“提出主张 → 主动寻找反例 → 修订方案”审查：

1. **概念 Loop**：区分集合离散度、物理违背、运动算子可靠性和校准风险，废止把不同量纲变量直接相加的旧定义。
2. **物理 Loop**：发现历史帧 warping 误差既可能来自坏光流，也可能来自真实生消，因此将其拆为运动可靠性 `C_flow` 与非平流演化证据 `M_nadv`，避免把对流新生误判为“算子不可信”。
3. **算法 Loop**：将采样引导从含糊的“对下一步样本减梯度”改为明确的 **预测干净残差 `r̂0` 近端修正 → 重新换算噪声/速度参数 → 执行原采样器更新**。
4. **执行 Loop**：将过度庞大的实验清单分成核心证据链和扩展证据链；加入基线准入规则、专用校准切分、主终点、失败降级路径和代码路线决策 Gate。
5. **创新边界 Loop**：对照最近邻路线后，收缩新颖性主张为“**不可靠物理算子下的可靠性门控引导**”；新增 PreDiff-style 同架构知识引导、physics-conditioning-only 对照、`C_flow/M_nadv` 可辨识性、未来 motion 可靠性和对流新生/消散负迁移测试。

因此，本版的核心不再是“给 DiffCast 加一个 PDE loss”，而是：

> **在物理算子可能错误、未来运动可信度随 lead time 下降、降水又存在真实生消的条件下，研究何时应该相信、减弱或关闭弱输运引导，并验证可靠性门控是否优于固定知识引导和仅物理条件化。**

---

# Part 1 研究定位

## 1.1 问题与任务协议

给定历史雷达序列 `X`，预测未来序列 `Y` 的条件分布：

```text
X = {v_{t-4}, …, v_t},        5 frames
Y = {v_{t+1}, …, v_{t+20}},   20 frames
p(Y | X)
```

主任务固定为：

```text
SEVIR-DiffCast: 5 → 20, 128×128, 5 min/frame, forecast horizon = 100 min
```

扩展长时效任务：

```text
SEVIR-long: 5 → 36, 128×128, 5 min/frame, forecast horizon = 180 min
```

长时效任务必须单独训练和评估，不得与 `5→20` 主表混表排名。

## 1.2 研究矛盾

降水短临预报同时包含：

- 可外推的平移、旋转、形变等运动；
- 难以由纯平流描述的新生、增强、衰减、消散；
- 给定相同历史时存在多个合理未来的条件随机性；
- 雷达观测噪声、遮挡、编码饱和和边界缺失；
- 运动估计器本身可能错误。

DiffCast 式残差生成将预测分解为：

```text
Y = μ(X) + r
```

其中确定性分支 `μ` 给出可预测趋势，生成分支建模剩余随机残差 `r`。该结构适合本问题，但若直接施加固定物理权重，会出现两个风险：

1. 错误运动场把样本推向错误方向；
2. 真实生消被纯平流约束误判为物理违规，导致极端降水被压平。

## 1.3 可辩护的研究空白

在完成系统检索与方案定稿前不使用“首次”。当前可辩护的 gap 为：

1. 残差生成 nowcasting 通常没有显式区分：生成集合离散、预测场约束违背、运动算子可靠性、真实非平流演化；
2. 现有物理/知识引导常使用固定或全局权重，较少在不可靠运动场区域主动关闭错误引导；
3. 物理违背常被直接当作 uncertainty 使用，但缺少概率定义、独立校准切分和部署时风险验证；
4. 很多工作只证明“自己定义的物理损失下降”，没有用独立运动、对象和区域收支指标排除指标自洽；
5. 物理引导的收益常未与相同额外梯度次数、错误 flow、随机平滑和相同 NFE 的控制组比较。

## 1.3.1 最近邻工作与创新边界（方案中必须正面回答）

本工作不再把“降水扩散中加入知识/物理引导”或“物理确定性预测 + 残差生成”本身作为新颖性。最近邻路线至少包括：

- **PreDiff-style knowledge alignment**：在去噪过程中依据知识违背调整生成转移；
- **physics-conditioning / Nowcast3D-like 路线**：先得到物理或运动驱动的确定性预测，再由条件生成模型补充残差与集合；
- 固定权重的 physics-guided sampling 与通用 scientific diffusion guidance。

因此，PhyRD 的主张收缩为：

> **Reliability-gated weak-transport guidance under uncertain operators：当运动算子可能错误且真实场含非平流生消时，显式估计 motion reliability 与 non-advection evidence，避免固定物理引导造成负迁移。**

必须通过同架构最近邻对照证明收益来自“何时关闭/放宽错误约束”，而不是来自更多 NFE、额外梯度、输出平滑或一般知识引导。后续检索若发现完全同构工作，必须进一步收缩 claim。

## 1.4 研究问题

- **RQ1 — 技巧与概率质量**：弱输运训练正则和推理引导，能否改善强回波与长 lead 技巧，同时保持 CRPS、可靠性和 ensemble spread？
- **RQ2 — 可辨识性与负迁移**：`C_flow` 能否识别坏 motion、`M_nadv` 能否识别真实非平流演化，并在两者混淆时减少新生/消散区域的负迁移？
- **RQ3 — 最近邻归因**：可靠性门控与容忍预算是否优于 PreDiff-style 固定知识引导、固定采样引导，以及只把物理预测作为 condition 的方案？
- **RQ4 — 风险诊断（条件性贡献）**：`R_phys/C_flow/M_nadv` 是否在 `U_ens + intensity + lead time` 之外提供跨域可校准的错误风险增量？
- **RQ5 — 边界**：技巧、联合分布质量、对象新生召回、物理一致性与推理成本之间的 Pareto 边界在哪里？

## 1.5 贡献层级

### 主贡献 C1：可靠性与生消分离的弱输运约束

构造 VIL proxy 上的局地 warping 残差和多尺度区域收支残差；使用 `C_flow` 表示运动算子可靠性，使用 `M_nadv` 表示真实非平流演化证据。两者承担不同作用：

- `C_flow` 决定“是否相信这个运动方向”；
- `M_nadv` 决定“需要给纯平流约束多大容忍度”。

### 主贡献 C2：违背反馈式可靠性门控训练与采样

把物理引导实现为 **violation-feedback adaptive guidance + proximal clean-residual correction**。默认不把逐像素逐步更新包装成严格的全局 primal-dual 最优算法；只有在补全局部约束族、变量更新与稳定性分析后，理论附录才可使用 dual interpretation。贡献重点是可靠性门控、非平流容忍以及与最近邻方法的归因实验。

### 条件性贡献 C3：可校准的物理风险归因

分别保留：

- `U_ens`：生成集合离散度；
- `R_phys`：标准化约束违背分数；
- `C_flow`：运动算子可靠性；
- `M_nadv`：非平流演化证据；
- `P_err`：在独立校准切分上拟合的局地错误概率。

若 `R_phys/C_flow/M_nadv` 对 `U_ens` 没有稳定增量，C3 必须降级为附录诊断，论文仍以 C1–C2 为主，不能强行维持故事。

---

# Part 2 方法

## 2.1 符号与输出

```text
x        [B, Tin, 1, H, W]
y        [B, Tout,1, H, W]
μ        [B, Tout,1, H, W]
r0       [B, Tout,1, H, W]
ŷ(k)     [B, Tout,1, H, W]
flow     [B, Tout-1,2,H,W]
C_flow   [B, Tout-1,H,W]
M_nadv   [B, Tout-1,H,W]
R_phys   [B, Tout-1,H,W]
P_err    patch-level or pixel-level calibrated risk
```

默认风险输出以 `16×16` 或 `32×32` patch 为主，像素级结果作为附录；这样可降低干区类别失衡和空间相关造成的虚高统计显著性。

## 2.2 确定性趋势与残差生成

```text
μ = Pθ(X)
r0 = Y − stopgrad(μ)
pφ(r0 | X, μ)
ŷ(k) = μ + r̂(k)
```

其中 `Pθ` 不再是 2D U-Net，而是完整 SDIR：

```text
X + coarse-frequency condition
        ↓
SFG-Former / Scale-Adaptive Transformer / 3-D RoPE
        ↓
low-frequency synoptic skeleton μ_skel
        ↓
FR-Refiner / scale-conditioned FNO
        ↓
μ = clip(μ_skel + Δμ_fourier, 0, 1)
```

训练时从 `Beta(1,3)` 采样保留尺度 `s`，将 `Y` 双三次降采样再恢复为低频条件；目标为：

```text
L_SDIR = MAE(μ_skel,Y)
       + MAE(Δμ_fourier,Y−μ_skel)
       + 0.01·(s/W)^2·PCPSD(μ,Y;s)
```

推理时没有 `Y`：从零未来 condition 和 `s=0` 开始，按 `frequency_stride` 迭代预测，并把当前预测投影为下一尺度条件。`μ` checkpoint 独立训练、评估和冻结；所有生成/物理消融必须共享同一个 SDIR checkpoint。

执行原则：

1. `Pθ` 独立训练、评估并冻结；
2. 所有核心消融共享同一个 `μ` checkpoint；
3. 首版以 DiffCast-like denoising diffusion 为直接母基线；
4. 若预算允许，增加同架构 residual conditional flow matching 作为生成引擎迁移实验，验证物理机制是否依赖 DDPM/DDIM。

## 2.3 VIL proxy 与物理边界

SEVIR VIL 不能直接称为 dBZ、雨强或严格守恒质量。定义固定单调非负映射：

```text
q = g(v), q ≥ 0
```

MVP 使用训练前固定的归一化 VIL proxy。禁止学习一个可任意缩放从而规避约束的 `g`。

论文表述限定为：

- weak transport consistency；
- transport-consistency proxy；
- physics-inspired inductive bias。

禁止表述：

- exact mass conservation；
- complete atmospheric PDE；
- true microphysical diffusivity。

## 2.4 运动算子可靠性与非平流演化分离

### 2.4.1 运动场

从最后若干观测帧估计雷达域 motion：

- TV-L1 / rainymotion 类光流；
- 或在雷达序列上自监督适配的神经 motion model。

自然图像 RAFT 只作消融。未来 motion 默认从观测期末速度场常速外推，线性和学习式外推进入消融。

### 2.4.2 运动可靠性 `C_flow`

`C_flow` 只使用尽量与真实生消解耦的证据：

```text
E_fb       = forward-backward consistency error
E_texture  = local aperture / texture conditioning score
E_bound    = out-of-bound and invalid-mask indicator
```

```text
C_obs = valid_mask · exp(
    −a·Norm(E_fb)
    −b·Norm(E_texture)
    −c·Norm(E_bound)
)

C_flow(x,τ) = C_obs(x) · C_extrap(x,τ)
```

`C_extrap` 表示从观测期 motion 外推到未来 lead `τ` 的可信度，必须随 lead time 经验证集校准；可由多种外推器/flow ensemble 的 disagreement、历史 backtest 或学习式 motion uncertainty 得到。禁止默认把观测期高置信度无衰减复制到未来 60 min。历史 warping intensity error 不直接并入 `C_obs`，因为它也可能来自真实增强/衰减。

### 2.4.3 非平流演化证据 `M_nadv`

利用多个合理 flow 候选下仍然存在的稳健残差、局地强度趋势和形态变化，构造：

```text
M_nadv ∈ [0,1]
```

它不是“坏 flow 概率”，而是“纯平流不足的证据”。它用于放宽约束预算或激活受限源汇项：

```text
ε(x,τ) = ε_base(τ) · (1 + γ_nadv · M_nadv(x,τ))
```

这样，新生/消散区不会因为 warping error 大而被错误地强制回原位置。

### 2.4.4 可辨识性边界与校准要求

仅凭单通道雷达强度，坏 motion 与真实生消并非完全可辨识，因此 `C_flow/M_nadv` 只能称为**部署时证据或校准评分**，不能称为真实概率或已识别因果变量。

- D0 使用 `flow_gt/source_gt/corruption_mask` 分别监督和评估两者；
- 真实数据优先加入多 motion 方法一致性、对象追踪位移、可用时的 Doppler/NWP 风场或人工新生消散标注；
- 若真实域只能证明相关性，论文必须限制结论，不声称准确分离所有运动错误与微物理生消。

## 2.5 弱输运约束

### 2.5.1 MVP：advection + regional budget

首版默认：

```text
κ = 0
S = 0
```

原因：可微 warping 已承担平流；过早加入 Laplacian 可能与插值数值扩散重复并引入额外平滑。只有 MVP 成立后才加入 `κ` 和 `S`。

局地残差：

```text
R_local,τ = q̂_{τ+1} − Warp(q̂τ, ûτ)
```

多尺度区域收支：

```text
R_mass,τ(m) = SumPool_m(q̂_{τ+1})
            − SumPool_m(Warp(q̂τ,ûτ)),  m∈{8,16,32}
```

### 2.5.2 扩展：有效扩散与受限源汇

通过 Gate 后才启用：

```text
R_local,τ = q̂_{τ+1}
          − Warp(q̂τ,ûτ)
          − Δt·(κΔq̂τ + ŝτ)
```

约束：

- `κ≥0`，固定标量或低自由度 bounded field；
- `ŝ=sψ(X,τ)` 只能读取历史与 lead time；
- `ŝ` 低分辨率输出后上采样；
- `tanh` 限幅并加 L1、TV、temporal smoothness；
- 禁止读取 `Y`、当前 target residual 或当前预测误差。

### 2.5.3 稳健归一化

按训练集、lead-time 和强度桶冻结 robust scale：

```text
R̃ = R / (MAD_train,lead,intensity + ε)
```

不能使用测试集统计；不能用当前 batch 标准差让模型通过改变分布缩小损失。

## 2.6 局部约束视角与违背反馈更新

把生成过程写成带约束的条件生成：

```text
minφ  L_gen(φ)
subject to
E[Wτ(x) · ℓphys,τ(ŷ)] ≤ ετ(x),   τ=1…Tout−1
```

其中：

```text
Wτ = C_flow
ετ(x) = ε_base(τ) · (1 + γ_nadv M_nadv)
```

拉格朗日形式：

```text
L = L_gen + Στ λτ · (E[Wτℓphys,τ] − ετ)
λτ ≥ 0
```

训练首版可使用 warmup penalty；推理期默认称为 **violation-feedback update**。若论文采用 dual interpretation，必须把约束明确为每个空间单元与 lead time 的局部约束族 `g_{i,τ}≤ε_{i,τ}`，说明 `λ_{i,τ}` 的初始化、跨去噪步保留方式、更新次数和稳定性；否则只把该形式作为设计动机，不宣称 primal-dual 收敛或最优性。

## 2.7 训练期物理正则

噪声预测参数化下：

```text
rt = αt r0 + σt ε
ε̂ = εφ(rt,t,X,μ)
r̂0 = (rt − σt ε̂) / αt
ŷ0 = μ + r̂0
```

物理损失必须作用于 `ŷ0`：

```text
L_phys = E[C_flow · Huber(R̃_local; tolerance=ε)]
       + α_mass Σm E[Pool(C_flow,m) · Huber(R̃_mass(m); tolerance=εm)]
```

总损失：

```text
L_total = L_gen + λ_train(epoch) L_phys
```

必须监控：

- `∇φL_phys` 非零；
- `L_gen/L_phys` 梯度范数与夹角；
- 强回波区域是否因物理正则系统性减弱；
- `C_flow` 和 `M_nadv` 的分布。

## 2.8 推理期近端物理修正

当前版本不再直接写含糊的 `r_{j−1}←r_{j−1}−…`。每个被选中的采样步按以下流程执行：

1. 原生成器从当前状态 `rt` 预测干净残差 `r̂0`；
2. 在 `r̂0` 空间计算约束能量；
3. 对 `r̂0` 做一步或少量步近端梯度修正；
4. 将修正后的 `r̂0,corr` 重新换算成该采样器需要的噪声/速度参数；
5. 执行原 DDIM/DDPM/ODE 更新。

```text
Ephys(r̂0) = Στ C_flow · [R̃phys(μ+r̂0) − ετ(M_nadv)]_+

λ ← clip(λ + ρ·C_flow·[R̃phys−ετ]_+, 0, λmax)

r̂0,corr = r̂0 − ηt · λ · ∇r̂0 Ephys

εcorr = (rt − αt r̂0,corr) / σt
rt−1 = SamplerStep(rt, εcorr, t)
```

实现要求：

- `λ` 默认 detach 更新，避免无意构建二阶图；
- 对 `r̂0` 修正做 norm clipping 与可选 backtracking；
- 只在后若干步或间隔步骤执行；
- 通过合成数据测试“修正后约束残差单调下降”；
- 若修正导致 base sampler 的预测代理显著恶化，回滚该步。

## 2.9 风险估计

### 2.9.1 部署时可用特征

只能使用部署时可获得的信息：

```text
log(U_ens+ε)
log(R_phys+ε)
1−C_flow
M_nadv
predicted intensity / input intensity
lead time
predicted gradient / object size
```

禁止把真实未来强度、测试标签或由真值计算的分层变量作为校准器输入。

### 2.9.2 风险目标

分别定义，不混成单一含糊标签：

- 连续误差超过预注册阈值；
- 强回波 miss；
- 强回波 false alarm；
- patch-level CSI 或 FSS 低于阈值。

### 2.9.3 数据隔离

```text
train      → 训练预测模型
val-model  → 选 checkpoint、λ、motion 和约束超参
val_calib  → 拟合风险校准器
report_test→ 一次性冻结评估
```

若样本不足，使用按 weather event 分组的 nested cross-fitting。禁止在同一验证预测上既选特征又报告校准泛化。

---

# Part 3 实验设计

## 3.1 证据梯度：核心与扩展

### 核心证据链（必须）

```text
D0 可控合成机制数据
+ D1 SEVIR-1h 完整主实验
+ 一个独立真实雷达集（优先 MeteoNet；HKO-7 可替代）
+ 同协议核心基线
+ 关键消融、负控制、概率质量、效率和统计检验
```

### 扩展证据链（资源允许）

```text
第二个外部雷达集
+ SEVIR-3h
+ 1/10/50/100% 数据效率
+ residual flow-matching 引擎迁移
+ 缺帧/噪声/遮挡鲁棒性
```

不再把“D0+SEVIR+MeteoNet+HKO-7+3h 全部完成”设为唯一完成条件，以免项目被数据处理和外部基线拖垮。核心机制、两个真实域和统计验证构成主体证据，其余作为扩展验证。

## 3.2 数据集

### D0 合成输运数据

已知 `flow_gt/κ_gt/source_gt`，包含：平移、旋转、剪切、变速、扩散、生消、合并、分裂、遮挡、flow noise。用于：

- 解析验证；
- oracle 与 wrong-flow 对照；
- `C_flow/M_nadv` 解耦，并报告各自对 flow corruption/source mask 的 AUROC、AUPRC、校准与交叉混淆；
- 引导单调性；
- 源汇作弊测试；
- 条件分析：在固定 source 强度下评估 `C_flow`，在固定 flow error 下评估 `M_nadv`，检验两者是否只是同一 residual 的重命名。

### D1 SEVIR-DiffCast

主协议：`5→20@128`。所有主表模型共享同一冻结 HDF5、`train/valid/test` 源 group、由 `valid` 样本 ID 奇偶冻结出的 `val_model/val_calib`、阈值、指标实现和样本 ID。

### D2 MeteoNet

开放雷达数据，独立单位、阈值、切分与归一化。不得复用 SEVIR VIL 阈值。优先用一个区域训练、另一区域补充域偏移分析，具体方案在数据审计后冻结。

### D3 HKO-7

若数据访问与许可顺利，可作为第二外部域；若访问受限，则由另一个公开连续雷达集替代，并在方案中记录替代依据。

### D4 SEVIR-long

`5→36` 作为长时效扩展；必须独立构造冻结数据与切分，不得复用主协议 test 选参。

## 3.3 基线准入规则

主表不再写“必须塞入所有 SOTA”。每个基线先分级：

### Tier A：同协议、可公平重训，必须进主表

- Persistence；
- optical flow / rainymotion；
- PySTEPS；
- SimVP 或 Earthformer；
- **SDIR standalone deterministic**（与 PhyRD 使用同一 checkpoint，隔离衡量确定性骨干收益）；
- backbone-matched DiffCast-like；
- compute-matched DiffCast control；
- **PreDiff-style same-backbone knowledge alignment**：同一 weak-transport energy、无 `C_flow/M_nadv`；
- **physics-conditioning-only / Nowcast3D-like 2D control**：物理/运动确定性预测只作为 condition，不在采样中做物理修正；
- fixed physics-guided sampling；
- PhyRD。

### Tier B：官方代码可用，尽量同协议重跑

- PreDiff；
- CasCast；
- FlowCast；
- FREUD；
- PostCast 或后续核验通过的新增强模型。

只有完成同 split、同输入输出、同域指标或经过明确适配后才能进入统一主表。否则进入“官方协议背景表”。

### Tier C：协议不匹配或无法重跑

只能作 related work / literature reference，不能把论文数字与本方法直接加粗比较。

每个 baseline 必须有：

```text
repo URL, commit SHA, license snapshot, environment lock,
checkpoint SHA256, dataset manifest, protocol, NFE, ensemble K, hardware
```

## 3.4 主终点与成功判据

为减少多指标挑结果，测试前冻结两类共同主终点：

1. **强回波/确定性主终点**：CSI-M 与一个高阈值 CSI（SEVIR 优先 CSI-219；若极端样本过少则按预注册规则使用 CSI-181）；
2. **概率主终点**：CRPS，要求相对 DiffCast-like 达到预注册非劣界，并尽量改善。

独立物理指标和风险校准为关键次终点。

非劣界不在测试结果出来后制定，应根据验证集重复实验方差和业务尺度预先冻结。

## 3.5 指标

### 预测与空间结构

- CSI / CSI-M / 高阈值 CSI；
- CSI-pool4 / CSI-pool16：先对预测和真值做 `kernel=stride∈{4,16}` 的非重叠空间 max-pool，再按相同 VIL 阈值计算 CSI；
- HSS；
- FSS 多尺度；
- MAE；
- SSIM 仅辅助；
- LPIPS 仅作感知辅助指标，逐帧计算并记录所用 backbone/weights；不得用随机初始化网络冒充 LPIPS；
- centroid displacement；
- connected-component track error；
- 可选 SAL/object-based score。

### 概率质量

- CRPS；
- threshold Brier/BSS；
- reliability + sharpness；
- spread-skill；
- rank/PIT diagnostics；
- 可选 Energy/Variogram Score 或 ensemble-FSS，用于补充空间联合分布。

### 风险

- Brier；
- AUPRC 为主、AUROC 为辅；
- calibration error；
- coverage-risk / selective prediction；
- patch、lead time、强度和事件分层。

### 成本

- 生成 NFE；
- physics gradient evaluations；
- ensemble size；
- p50/p95 latency；
- peak GPU memory；
- samples/s；
- energy/GPU-hours 可选。

## 3.6 必做实验

### E1 同协议主结果

Tier A 全部完成；Tier B 只纳入可公平重跑项。

### E2 核心机制阶梯

```text
B0 DiffCast-like residual diffusion
B1 + train-time fixed weak-transport loss
B2 PreDiff-style same-backbone knowledge alignment
B3 physics-conditioning-only / Nowcast3D-like 2D control
B4 fixed inference physics guidance
B5 + C_flow gating only
B6 + M_nadv tolerance only
B7 full PhyRD: C_flow + M_nadv + violation-feedback guidance
```

### E3 负控制

- 相同额外梯度次数但随机/零目标；
- shuffled flow；
- reverse flow；
- random smoothness energy；
- oracle flow（D0）；
- wrong flow + without gating；
- 仅输出平滑后处理。

### E4 物理定义消融

- warping only；
- warping + multi-scale budget；
- `κ=0` vs bounded `κ`；
- `S=0` vs bounded source；
- 约束 `μ`、`r`、`μ+r`。

### E5 运动解耦消融

- `C_flow` 不使用 warping intensity error；
- 将 warping error 错误并入 `C_flow` 的旧版对照；
- 无 `M_nadv`；
- `M_nadv` 只调容忍度；
- `M_nadv` 激活 bounded source。

### E5b 对流演化分层与负迁移

按预注册规则将事件或对象分为：平移主导、增强、衰减、新生、消散、合并、分裂、边界进入。分别报告：

- CSI/POD/FAR 与高阈值召回；
- FSS、centroid/track error；
- object birth/death recall；
- peak intensity 与强回波面积误差；
- CRPS 与 ensemble spread。

必须重点检验 PhyRD 是否压制真实新生单体。若新生/消散子集显著退化，主张应限制为平移占优场景或重新设计容忍机制。

### E6 概率与多样性

- K∈{5,10,20,40}；
- NFE/physics steps 曲线；
- mode collapse/spread-skill；
- same-compute control。

### E7 风险增量

嵌套模型：

```text
U_ens
U_ens + R_phys
U_ens + R_phys + C_flow
U_ens + R_phys + C_flow + M_nadv
full deploy-time features
```

风险结果按 event-block 统计，避免把数百万相关像素当作独立样本。

### E7b `C_flow/M_nadv` 可辨识性

D0 必做定量指标：

- `C_flow` 对 flow corruption/error mask 的 AUROC/AUPRC；
- `M_nadv` 对 source/non-advection mask 的 AUROC/AUPRC；
- 两者交叉预测结果，检验是否混为同一个 residual score；
- oracle、wrong-flow、source-only、flow-only 和联合扰动四象限；
- lead-time calibration of `C_extrap`。

真实数据至少使用一种独立佐证：多 motion 方法 disagreement、对象追踪位移、Doppler/NWP 风场或人工新生/消散标注。若没有独立信息，只能报告“与错误相关”，不能声称真实分离。

### E8 外部域

在一个独立真实雷达数据集完成：Tier A 最小基线、DiffCast-like、PhyRD、主要指标和关键消融。第二外部域是增强项。

### E9 生成引擎迁移（扩展但高价值）

用相同 deterministic backbone 与 residual representation，对比：

```text
residual diffusion vs residual conditional flow matching
with and without weak-transport guidance
```

该实验用于回应“方法是否只适用于过时的 diffusion sampler”。不要求替代主母基线。

## 3.7 统计与案例选择

- 核心方法、DiffCast-like 和关键消融至少 3 seeds；
- 按 weather event 配对 bootstrap；
- 风险按 event/patch block bootstrap；
- 报绝对差、相对差、95% CI 和效应量；
- 共同主终点之外的多重比较做 Holm 校正或明确标为探索性；
- 案例选择规则在看测试结果前冻结，例如极端事件前 N、随机事件和失败事件各固定数量；
- 外部单 checkpoint 方法不能伪装成 3-seed 方差。

## 3.8 可证伪路径

- 若 B2/B3 已达到与 PhyRD 相同收益，则可靠性门控的独立贡献不成立；
- 若 `C_flow/M_nadv` 在 D0 上不能分别识别 flow error 与 source mask，则 C1 不成立；
- 若新生/消散对象召回显著下降，则弱输运引导存在关键负迁移，必须限制适用范围或返工；
- 若 `C_extrap` 不随 lead time 校准，禁止宣称长 lead 可靠性受控；

- 若 `L_phys` 下降但独立 track/FSS 不改善：说明约束指标自洽，C1 失败；
- 若固定 guidance 有益而自适应无益：保留固定引导，C2 降级；
- 若 `C_flow` 无法识别 wrong-flow：停止真实数据物理引导，先修 motion 模块；
- 若 `M_nadv` 导致约束完全关闭：限制其最大容忍增幅；
- 若风险特征无增量：C3 移至附录；
- 若高阈值提升但 CRPS 超过非劣界：不能宣称“可靠概率预报全面改善”；
- 若收益仅来自更多 NFE：方法不通过效率 Gate。

---

# Part 4 代码母体与仓库决策

## 4.1 先下载什么

第一优先下载并审计 DiffCast：

```bash
git clone https://github.com/DeminYu98/DiffCast.git
cd DiffCast
git checkout -b audit/phyrd-v10
```

原因：它是最直接的 residual diffusion 母基线，官方仓明确提供 SEVIR `5→20` 训练/推理代码，适合建立 `DiffCast → PhyRD` 的贡献链。

## 4.2 但不再盲目冻结“直接把旧仓改到 384”

Phase 0 必须生成 `ADR-001_CODEBASE_STRATEGY.md`，在以下两条路线中选择：

### Route A：直接 DiffCast fork

适用条件：

- 官方代码结构可维护；
- `13→12@384` 单 batch 前向/反向和采样在目标硬件上可接受；
- 数据、backbone 和评估改造不会导致大规模重写。

优点：最直接公平，缺点：GPL-3.0、旧代码、384 像素扩散成本高。

### Route B：标准协议 clean port

适用条件：

- 直接 fork 在 384 上不可运行或维护成本过高；
- 需要 Earthformer/现代数据管线；
- 可以通过固定接口重建 DiffCast-like residual diffusion 并在 128 协议上验证与官方行为一致。

Route B 不是规避对照：仍须独立跑官方 DiffCast，并提供 backbone-matched DiffCast-like。若以 DiffCast 源码为参考进行派生，许可证边界必须由实际代码来源决定，不能自行宣称 clean-room。

## 4.3 官方仓库清单（2026-07-14 核验）

| 角色 | 官方仓库 | 当前用途 | 关键边界 |
|---|---|---|---|
| 直接母基线 | `https://github.com/DeminYu98/DiffCast.git` | official 5→20 复现、残差扩散母体 | GPL-3.0 |
| 标准 SEVIR 协议 | `https://github.com/amazon-science/earth-forecasting-transformer.git` | 13→12@384 loader、Earthformer | Apache-2.0 |
| 高分辨率概率基线 | `https://github.com/OpenEarthLab/CasCast.git` | SEVIR split、权重、CasCast 独立运行 | 页面未显示明确 LICENSE 时不复制源码 |
| 物理/知识引导参考 | `https://github.com/gaozhihan/PreDiff.git` | KA 实现与合成实验参考 | Apache-2.0；原协议 7→6@128 |
| 当前强流模型 | `https://github.com/b-rbmp/FlowCast.git` | ICLR 2026 外部强基线、效率对照 | Apache-2.0 |
| 确定性主骨干 | `https://github.com/RuntimeWarning/SDIR.git` | SFG-Former、FR-Refiner、PCPSD 与 frequency-unlocking 参考 | README 声明 MIT；所检 commit 缺少 LICENSE 文件，发布前复核 |
| 当前强概率模型 | `https://github.com/CompVis/weather-rf.git` | FREUD 外部重跑 | 个人/科研非商业许可证 |
| 经典概率外推 | `https://github.com/pySTEPS/pysteps.git` | PySTEPS | 固定版本和配置 |
| HKO-7 工具 | `https://github.com/sxjscience/HKO-7.git` | HKO-7 协议与基线参考 | MIT code；数据使用另核验 |

## 4.4 项目机器可读控制文件

Codex 开始编码前必须创建：

```text
PROTOCOL.yaml
DECISIONS.yaml
BASELINE_REGISTRY.yaml
EXPERIMENT_REGISTRY.csv
DATA_MANIFESTS/
GATES.md
CHANGELOG.md
LICENSE_LEDGER.md
```

每个决策、实验和 Gate 使用稳定 ID：

```text
D-001 codebase route
D-002 SEVIR split
E-001 official DiffCast reproduction
E-010 physics train ablation
G-01 data contract
G-06 risk calibration
```

自然语言文档与机器可读文件冲突时，先停止执行并更新决策记录，不得静默选择一个版本。

---

# Part 5 Codex 执行阶段

## Phase 0：仓库审计与路线决策

交付：

- `BASELINE_AUDIT.md`；
- official commit、环境、入口、残差定义、采样参数化；
- official toy/eval smoke test；
- `5→20@128` 单 batch 前向、反向、采样内存与延迟记录；
- `ADR-001_CODEBASE_STRATEGY.md` 选择 Route A/B；
- 许可证账本。

禁止：此阶段不得实现 PhyRD。

## Phase 1：官方 DiffCast 复现

- 跑通官方 `5→20` 或官方权重；
- 复核输出域、指标、NFE 和 checkpoint；
- 保存原始预测样例与运行命令。

## Phase 2：标准协议与数据防泄漏

- 冻结 `5→20@128` HDF5 的 train/valid/test manifest 与 valid 二次划分；
- sample-id split 审计；
- persistence、optical flow、PySTEPS、确定性基线；
- CSI/CRPS/FSS 单元测试；
- 独立 `val_model/val_calib/report_test`。

## Phase 3：无物理 DiffCast-like

- 共享固定 `μ`；
- 建立无物理 residual diffusion；
- 验证 ensemble 不塌缩；
- 形成所有物理实验共同 checkpoint。

## Phase 4：D0、motion 可辨识性与未来可靠性

- 合成输运数据；
- `C_flow` 与 `M_nadv` 分离；
- correct/wrong/shuffled flow；
- future flow 外推；
- Gate：正确 flow 的独立误差必须显著更低。

## Phase 5：训练期弱约束

- advection + multi-scale budget MVP；
- `κ=0,S=0`；
- 物理梯度单测；
- same-compute 与 smoothness 控制；
- Gate：独立物理指标改善且主指标不灾难退化。

## Phase 6：推理期近端修正

- 实现 `r̂0 correction → ε/velocity recomputation → sampler step`；
- 固定权重 → `C_flow` → `M_nadv` 容忍 → dual update；
- 约束单调性、backtracking、成本曲线；
- Gate：收益不只是额外 NFE。

## Phase 7：风险校准

- patch-level 主风险；
- 部署时特征审计；
- val_calib 或 nested cross-fitting；
- 嵌套特征增量；
- Gate：相对 `U_ens` 有稳定增量，否则降级。

## Phase 8：最近邻同架构基线与核心真实数据论文实验

- SEVIR 主表；
- 一个外部真实数据集；
- 3 seeds、event bootstrap、主终点；
- 定性成功/失败案例；
- Tier B 基线按准入规则处理。

## Phase 9：扩展实验与方案冻结

按优先级：

1. 第二外部数据集；
2. residual CFM 引擎迁移；
3. SEVIR-3h；
4. 数据效率和缺帧鲁棒性；
5. 主实验冻结前重新检索并核验最新基线。

---

# Part 6 Codex 工程规则与总控提示词

## 6.1 强制工程规则

1. 一次只完成一个 Phase 和一个可审查 PR；
2. 修改前列出文件、调用链、shape、单位、梯度路径和测试；
3. 禁止整仓重写；优先最小 patch，保留一键退化到 baseline；
4. 不实际运行不得说成功；日志、命令和返回码必须保留；
5. 所有数据和预测由稳定 `sample_id/event_id/start_frame` 对齐；
6. 测试集只在冻结后评估，不参与选参、归一化和校准；
7. 所有部署风险特征不得读取未来真值；
8. 不同协议结果不得混表；
9. 新依赖先核验许可证、PyTorch/CUDA 和维护状态；
10. 失败结果写入实验注册表，不能删除；
11. 每个 Gate 通过后打 tag，未通过不得进入下一阶段；
12. 文档假设被实验否定时，先记录证据，再更新文档，不能让代码迎合故事。

## 6.2 可直接复制的总控提示词

```text
你负责执行 PhyRD-v10 项目。开始前完整阅读：
1. idea_report_master_v10.md（最高优先级）
2. implementation_v10.md
3. user_requirements_v10.md

严格按 Phase 0→9 执行，当前只完成我指定的 Phase。

每次开始前必须：
- 检查 git status、当前 commit、分支、环境和数据 manifest；
- 读取 PROTOCOL.yaml、DECISIONS.yaml、GATES.md 和 EXPERIMENT_REGISTRY.csv；
- 列出本阶段涉及的文件、函数、张量 shape、数据域、梯度路径、许可证来源和验收标准；
- 发现文档冲突时停止编码并提出最小决策修订。

每次结束必须交付：
- 实际修改的完整文件或 git diff；
- 可复制命令与真实日志；
- 单元测试和 smoke test；
- 与无修改 baseline 的对照；
- 资源消耗；
- 已知失败与下一 Gate；
- 更新 CHANGELOG、EXPERIMENT_REGISTRY、DECISIONS 和 LICENSE_LEDGER。

严禁：
- 在 μ+(Y−μ)=Y 上计算与生成网络无关的假物理损失；
- 把 warping error 同时当成坏 flow 与真实生消而不区分；
- 把 VIL 写成严格守恒质量；
- 把 R_phys 称为概率 uncertainty；
- 用测试集拟合校准器或选择阈值；
- 混用不同输入输出长度、分辨率、split 和 ensemble K；
- 未运行就声称性能提升；
- 未核验许可证就复制外部代码。
```

## 6.3 第一次执行指令

```text
先完成 Phase 0 审计并记录 Route A/B 决策；若当前任务明确要求工程交付，可继续完成 research-ready MVP，但必须把 code-ready 与 evidence Gate 分开，不得把 smoke test 写成性能验证。输出 `BASELINE_AUDIT.md`、`ADR-001_CODEBASE_STRATEGY.md`、真实运行日志和资源记录。
```

---

# Part 7 冻结与变更管理

当前冻结的是问题定义、实验公平原则和 Gate，不是未经实验验证的结论。

以下内容变化必须先更新 MASTER：

- 主任务和数据 split；
- DiffCast fork / clean port 路线；
- `C_flow/M_nadv/R_phys/P_err` 定义；
- 约束预算和采样参数化；
- 主终点和非劣界；
- 基线准入；
- 校准切分；
- 许可证和发布方式。

版本命名统一为 **PhyRD-v10.3**。v10.1 的 compact 2D U-Net 确定性实现仅作为历史实验，不再作为执行方案；旧文件中的 v6/v7/v8/v9 仅作为历史。
