# PhyRD v12：面向 CVPR 的热插拔概率短临预报设计方案

> 工作定位：把视频扩散中的时序生成、历史引导与高效注意力机制迁移到 precipitation nowcasting，构建可接在任意确定性 nowcaster 后面的概率模块。  
> 主协议：SEVIR VIL，`5→20@128`，5 min/帧，100 min horizon，与 DiffCast/PhyDNet 协议严格对齐。  
> 文档日期：2026-07-22。本文是研究与实施方案，不把尚未完成的实验写成性能结论。

---

## 0. 结论先行

建议不要把论文写成“SDIR 后面再接一个高频锐化扩散”，也不要继续把弱输运 physics guidance 当主线。更有机会的主线是：

> **一个以确定性预报为强锚点、显式建模形变与强度创新、可跨确定性主干训练和部署的时空概率适配器。**

本文暂称它为 **U-DIP：Universal Deformation–Intensity Probabilistic Adapter**。名字只是工作名，投稿前再做正式命名检索。

核心分解为：

```text
历史观测 X ──> 任意冻结的确定性 nowcaster P_b ──> 趋势 μ_b
       │                                      │
       └────────── history condition ─────────┤
                                              ▼
                            U-DIP probability adapter
                         p(d, a | X, μ_b, backbone-id/style)
                                              │
                                              ▼
                              Y_k = Warp(μ_b,d_k) + a_k
```

其中：

- `d` 是低分辨率 deformation correction，表达位置、轮廓与局地传播的不确定性；
- `a` 是高分辨率 intensity innovation，表达增强、衰减、新生和消散；
- 两者只在输出坐标上分开，在 denoiser 中通过共享时空表示联合建模；
- 模块只读标准化历史 `X` 和确定性输出 `μ_b`，不读 SDIR 的内部特征，因此接口层面热插拔；
- 通过多个冻结 backbone 的输出混合训练，争取做到同一概率 checkpoint 的权重级热插拔。

论文主问题不是“diffusion 能不能让图更锐”，而是：

> **像素加性残差把几何错位编码成高幅值正负尖峰，是否是尖锐降水系统的不良随机坐标？换成联合形变–强度坐标，并用视频扩散式 history anchoring，能否同时改善强回波技巧、轨迹分布和概率可靠性？**

这比“backbone-agnostic residual diffusion”更有辨识度，因为 DiffCast 已经宣称 residual diffusion 可以装备不同类型的时空模型；仅靠“能接不同 backbone”不足以成为 CVPR hero claim。

---

## 1. 当前工作审计：已经有什么、缺什么

### 1.1 本地代码已经具备的资产

当前仓库不是从零开始，已经具备：

- 冻结的 SDIR 官方实现与 checkpoint 接口；
- PhyDNet 外部 backbone 适配器；
- deterministic registry、backbone pool 和统一 `[B,T,1,H,W]` 输出契约；
- `ProbabilisticModel.training_loss()/sample()` 通用接口；
- `ForecastComposer` 的确定–概率组合；
- v-prediction、residual normalization、DDIM sampling；
- `5→20@128` DiffCast-compatible 数据协议；
- CSI、pooled CSI、HSS、SSIM、LPIPS、MAE、CRPS 等评估链；
- `val_model / val_calib / report_test` 隔离；
- official DiffCast 的 GPL 隔离基线登记；
- universal residual diffusion 与 SDIR/PhyDNet/backbone-pool 配置；
- JDIR 的初步设计与形变–强度分解论证。

这意味着迁移不应重写训练框架，而应新增一个概率模型包，并复用现有 composer、registry、数据与评估层。

### 1.2 已有结果告诉了我们什么

已完成的实验至少给出四个有效判断：

1. 早期 epsilon-prediction 失败是采样链坍塌，不是普通 loss 发散；v-prediction、标准化和裁剪修复了数值稳定性。
2. 弱输运 physics guidance 不是主要瓶颈。已有 B1 在无推理 guidance 时 K=8 得到 `CSI=0.29410, CRPS=5.9840, SSIM=0.7022`；加入 guidance 后 `CRPS=5.8417, SSIM=0.7297`，但 `CSI=0.29046`。它更像一个平滑/校准旋钮，而不是能提升强回波的核心机制。
3. 概率模块确实能改善 CRPS，但可能伤害确定性技巧。例如 PhyDNet 纯确定性为 `CSI=0.26979, CRPS=7.8243`；接 K=8 universal residual diffusion 后 `CRPS=6.3876`，但 `CSI=0.25678`、SSIM 也下降。这正是需要解决的 skill–reliability trade-off。
4. 当前 pixel residual denoiser 把 20 个 lead 当 2D 通道，缺少明确的 lead embedding、共享时序路径和逐帧噪声控制。视频扩散迁移的第一价值是修复这个建模缺口，而不是直接换一个大 DiT。

### 1.3 weather-30842 当前状态

远端真实路径为 `/test1/wzq/PhyRD`。截至审计时：

- 正在运行 `4×H800` 的 intensity-weighted v-pred residual diffusion；
- global batch 为 256，epoch 20 时 val denoising loss 为 `0.47545`；
- 尚无完整采样指标，不能用 loss 下降判断 CSI/CRPS 是否提升；
- 强度权重当前为 `w=1+4·Y` 后再归一化，它是连续像素加权，不是对 SEVIR 阈值、事件频率或生成样本的直接优化；
- 远端仓库中大量文件仍显示 untracked，并存在 `core.*` 与临时 patch/verify 文件。

因此当前 intensity run 应被登记为 **R1 诊断基线**：等它到预设评估点后做生成评估，但不因它仍在训练而推迟 U-DIP 的表示审计。正式新路线启动前要先冻结代码快照、config hash 和 checkpoint，不从远端临时 patch 状态直接分叉论文实验。

---

## 2. 投稿故事与贡献边界

### 2.1 推荐的论文故事

可以按以下逻辑写：

1. 强确定性 nowcaster 给出条件均值/主趋势，但真实未来在位置、形态和对流强度上是多模态的。
2. 现有 residual diffusion 通常在像素坐标建模 `Y-μ`。对尖锐回波，几像素位移就产生沿边缘成对的高幅正负残差，生成器要用“强度噪声”间接表达“位置不确定性”。
3. U-DIP 把概率修正改写为联合的低分辨率形变 `d` 与高分辨率强度创新 `a`，并用视频扩散式逐帧噪声与 history anchoring 保证整段预测一致。
4. 它使用统一输入输出协议，可挂在 SDIR、PhyDNet 以及另一种结构差异明显的确定性模型后；多主干训练使一个概率模块学习跨 backbone 的误差族，而不是记住 SDIR 的残差风格。
5. 最终同时检验 deterministic skill、ensemble probability quality、object trajectory 与跨主干迁移。

### 2.2 三个可投稿贡献

若实验支持，贡献可收敛为：

1. **随机坐标贡献**：指出并实证 additive pixel residual 对尖锐移动系统的病态性，提出 coupled deformation–intensity stochastic coordinate。
2. **视频扩散迁移贡献**：将 per-frame noise-as-mask、history-conditioned guidance 与 factorized space–time modeling 改造成适合 nowcasting 的 deterministic-anchor / observation-history 双条件机制。
3. **通用概率适配贡献**：同一接口、最好同一权重，在多个冻结确定性 nowcaster 上提升概率质量且不牺牲关键技巧，并给出严谨的 plug-and-play 评估协议。

### 2.3 不能单独当创新的内容

- deterministic + stochastic decomposition：DiffCast 已有；
- residual diffusion：DiffCast、CorrDiff、CoST 等路线已有；
- latent diffusion：PreDiff 及后续工作已有；
- temporal attention / DiT：只能算必要组件；
- intensity-weighted loss：只能算训练技巧；
- ControlNet 式条件注入：只能算实现选择；
- “支持任意 backbone”的软件接口：DiffCast 已有类似主张；
- physics loss 或推理期物理梯度：现有结果不支持作为主线。

### 2.4 两层“热插拔”必须分清

```text
Level A：接口热插拔
任意 backbone 只要输出统一 trend 张量即可接入；允许每个 backbone 单独训练 adapter。

Level B：权重热插拔
同一个训练好的 adapter checkpoint，不微调或仅校准少量归一化参数，就能换 backbone。
```

当前代码已经实现 Level A；Level B 只是研究假设。论文只有在 held-out-backbone 实验通过后才能宣称 universal one-checkpoint adapter。否则应诚实写成 backbone-agnostic interface + efficient adaptation。

---

## 3. U-DIP 方法设计

### 3.1 数据与确定性锚点

```text
X ∈ [0,1]^[B,5,1,128,128]
Y ∈ [0,1]^[B,20,1,128,128]
μ_b = stopgrad(P_b(X))
```

训练概率模块时确定性 backbone 全部冻结。所有核心对照必须复用同一份预计算 `μ_b`，避免因为重复前向、checkpoint 差异或随机预处理破坏归因。

建议离线建立 trend bank：

```text
trend_bank/{split}/{backbone}/{sample_id}.npz
  history
  target
  trend
  backbone_id
  checkpoint_sha256
  protocol_hash
```

这也能把 8 卡训练的显存和吞吐用于概率模块本身。

### 3.2 从像素残差改成联合形变–强度坐标

定义：

```text
d ∈ R^[B,T,2,32,32]
a ∈ R^[B,T,1,128,128]
Y_hat = clip(Warp(μ_b, Up(d)) + a, 0, 1)
```

设计原则：

- `d` 低分辨率、平滑、有界，只负责可被局地形变合理解释的误差；
- `a` 保留全分辨率和正负号，允许新生/消散，不强迫所有变化都服从平流；
- `d` 不称作真实风场，`a` 不称作真实微物理源汇；
- 模型学习联合分布 `p(d,a|X,μ_b)`，不是分别训练两个独立生成器。

### 3.3 形变目标的 MVP 与长期版本

真实数据没有 `d` 标签，首版不应直接把一个复杂外部光流网络变成黑盒前提。建议两阶段：

**MVP：优化式软配准目标**

```text
d* = argmin_d  ρ(Y, Warp(μ,d))
              + λ_s ||∇d||_1
              + λ_t ||d_t-d_{t-1}||_1
              + λ_m ||d||_1
a* = Y - Warp(μ,d*)
```

- `ρ` 同时含稳健强度、局地梯度和多尺度结构项；
- 生成置信图 `M_reg`，只降低不可辨识区域的形变监督，不把低置信区强制设为零运动；
- 对纯平移合成雨团做单元测试；
- 对新生/消散、边界出流、弱纹理区域单独审计。

**长期版本：amortized decomposition**

如果离线配准成为瓶颈，可训练一个只用于生成操作性坐标的 decomposition encoder。它仍需通过重建、平滑、置信和打乱对照，不能被包装成物理真值。

### 3.4 视频扩散机制如何迁移

#### A. DFoT 式逐帧噪声，而不是简单“加 history attention”

History-Guided Video Diffusion 的关键不是普通 temporal attention，而是对不同帧使用独立噪声等级，使噪声承担 mask 的角色，从而支持不同历史子集的条件分数。

迁移到 nowcasting：

- 历史观测 `X` 永远作为真实观测条件；
- 确定性未来 `μ` 是未来结构锚点；
- 对 `(d,a)` 的不同 future lead 采样独立或分组噪声等级；
- 训练时随机将部分近未来 lead 设为低噪声 pseudo-history，其余为高噪声待生成帧；
- 推理时组合 full-history、recent-history 与 deterministic-anchor 三种 conditional score。

建议的 guidance：

```text
s = s(X_recent, X_full, μ)
  + w_h [s(X_full, μ) - s(X_recent, μ)]
  + w_μ [s(X_full, μ) - s(X_full, μ_masked)]
```

这不是照搬公式，实际首版只实现一个 history guidance 和一个 anchor guidance，并在 `val_model` 上画 `w_h × w_μ` 的 skill/diversity Pareto 曲线。

#### B. factorized space–time denoiser

首版不建议直接上超大 DiT。`128×128×20` 上更稳妥的是 U-Net/DiT hybrid：

```text
spatial blocks at 128/64
        ↓
cross-scale d↔a coupling at 32/16
        ↓
temporal attention / temporal conv at bottleneck
        ↓
shared spatial weights + explicit lead embedding
```

要求：

- 时间维始终显式存在，不能只压成无序 channel；
- 空间权重跨 lead 共享；
- lead-time embedding 表达 5–100 min 的误差增长；
- deformation 与 intensity 分支在低分辨率交换特征；
- condition encoder 只读 `X,μ` 和可公开计算的 mask/lead embedding。

#### C. Sparse VideoGen 只作为后期加速

Sparse VideoGen 是训练后/无需重训的时空稀疏注意力加速思路。它适合在 U-DIP hero 通过后做 NFE/延迟优化，不适合首轮就与表示创新绑在一起。先确认 dense attention 的收益，再做 sparse head profiling 或窗口化近似。

#### D. 不建议首版使用 Astra 式 action adapter

nowcasting 当前没有动作输入。把 deterministic forecast 叫“action”会造成概念错位。将来加入 NWP、卫星、地形或环境场时，才可把多源外生量作为独立 modality adapter；这应是 extension，不是当前主线。

### 3.5 条件注入：强锚定但不复制底图

条件分两路：

```text
history encoder:       X -> multi-scale dynamic context
anchor encoder:        μ -> zero-init residual control features
```

使用 zero-initialized control projection 或 FiLM/cross-attention 注入，保证初始化时概率模块近似不破坏 `μ`。同时做两种 dropout：

- history subset/noise masking：支持 history guidance；
- anchor masking/style perturbation：避免模型只复制某个 backbone 的特定误差风格。

但 anchor dropout 不能过大，否则单样本会漂移。建议首轮 `p_anchor_drop∈{0.05,0.1}`，只在 `val_model` 选一个，不进行大网格搜索。

### 3.6 多 backbone 训练与 held-out 泛化

训练 backbone pool 至少包括：

- SDIR：强、sharp、频率细化型；
- PhyDNet：物理结构不同、误差谱不同；
- 第三个轻量 backbone：优先选择已可稳定复现的 ConvLSTM/UNet/Earthformer-like 之一。

训练采样：

```text
b ~ balanced categorical(backbones)
μ = trend_bank[b, sample]
loss = L_U-DIP(X,Y,μ,b)
```

不能按数据吞吐自然采样，否则最快/最弱的 backbone 可能主导训练。对每个 backbone 记录 residual scale、CSI、频谱和事件分层，并使用：

- 固定 `[0,1]` 输出域；
- per-lead 全局 scale + 轻量 backbone-style normalization；
- backbone ID 仅作为可选 style token；
- 训练中随机隐藏 backbone ID，检查模型是否真的依赖输出统计而非标签记忆。

三种评估：

1. **in-pool**：同一 checkpoint 接训练见过的 SDIR/PhyDNet；
2. **held-out backbone**：留一主干不参与 adapter 训练，直接接入；
3. **few-shot calibration**：只更新 normalization/LoRA/adapter，不更新主 denoiser。

只有第 2 类不退化，才能写“zero-shot hot-swappable”。若第 2 类失败但第 3 类成功，则写“parameter-efficient hot adaptation”。

### 3.7 训练目标

基础扩散目标使用已经验证稳定的 v-prediction：

```text
z_d = normalize(d*)
z_a = normalize(a*)
z_t = α_t z_0 + σ_t ε
v*  = α_t ε - σ_t z_0

L_v = L_vd + L_va
```

形变监督使用置信加权：

```text
m = m_floor + (1-m_floor) M_reg
L_vd = sum(m · ||v_d-v_d*||²) / sum(m)
```

只在中低噪声区增加小权重的可重建损失：

```text
Y_hat_0 = Warp(μ,d_hat_0) + a_hat_0
L_rec = Charbonnier(Y_hat_0,Y)
      + λ_grad Charbonnier(∇Y_hat_0,∇Y)
      + λ_pool Σ_{p∈{4,16}} Charbonnier(MaxPool_p(Y_hat_0),MaxPool_p(Y))
```

总目标：

```text
L = L_v + 1[t≤t_rec] λ_rec L_rec
```

强度加权不作为 hero。当前 `1+4Y` 实验完成后，若显示高阈值 recall 有收益，可将其改成事件频率校正后的分段权重，并做严格消融。不要直接对 CSI 做不可导 surrogate 主导训练，否则容易以 FAR/校准为代价刷阈值。

### 3.8 概率分数微调

只有基础模型完整采样通过后，再用 `K=4` 的短步可微采样做小学习率微调：

```text
L_prob = fair_CRPS_K(Y_1:K,Y)
       + λ_brier Σ_q Brier(P(Y>q), 1[Y>q])
```

训练时 ensemble CRPS 要使用 finite-ensemble bias correction/fair CRPS；最终报告仍给 empirical CRPS，并写清定义。该阶段的目标是改善可靠性，不替代表示创新的证据。

### 3.9 像素还是 latent

主协议 `128×128` 首版建议继续像素空间：

- 当前计算可承受；
- 避免确定性 VAE 压缩丢失小尺度不确定性；
- 更容易做 DiffCast 的公平 compute-matched 对照；
- U-DIP 的新意在随机坐标，不在 latent compression。

到 `384×384` 扩展时再使用 latent。若使用 latent，decoder 必须审计 uncertainty attenuation；最好使用 uncertainty-preserving 或统一时空 decoder，而不是把所有 stochasticity 压进一个确定性解码器。

---

## 4. 与 DiffCast 的公平比较：怎样“超过”才有故事

“超过 DiffCast”至少要回答三种不同问题，不能只选最有利的一张表。

### 4.1 三层基线

| 层级 | 对照 | 回答的问题 |
|---|---|---|
| B-official | official DiffCast 官方代码/checkpoint 或严格复现 | 对外部已发表方法是否更强 |
| B-backbone | 同一个 SDIR trend + DiffCast-like pixel residual diffusion | 收益是否来自概率表示而不是更强 deterministic backbone |
| B-compute | 参数量、训练 step、NFE、ensemble K 匹配的 pixel residual 模型 | 收益是否只是参数/算力更多 |

必须额外保留：

- naked SDIR / naked PhyDNet；
- SDIR + 当前 intensity-weighted residual；
- SDIR + universal pixel residual；
- SDIR + U-DIP；
- PhyDNet + 相同 U-DIP checkpoint。

### 4.2 主结果不能只报 ensemble mean CSI

正式表建议分三块：

**确定性/技巧**

- ensemble mean 的 CSI-M；
- `CSI-181` 或预注册的 `CSI-219`；
- CSI-pool4、CSI-pool16 / FSS；
- HSS、MAE、SSIM；
- centroid/track error、object displacement error。

**概率质量**

- CRPS / fair CRPS；
- threshold Brier score；
- reliability diagram 与 ECE；
- rank histogram；
- spread–skill ratio；
- coverage-width curve。

**效率**

- 参数量、训练 GPU-hours；
- sampling NFE；
- K=1/4/8/16 的 latency、峰值显存；
- CRPS/CSI 随 K 与 NFE 的 scaling curve。

### 4.3 推荐主终点

预注册两个共同主终点：

```text
Skill endpoint:  CSI-M + high-threshold CSI（层级检验）
Probability endpoint: CRPS（相对 DiffCast-like 要改善或至少不劣）
```

并设 guardrail：

- FAR 不能超过预注册容忍界；
- 新生/消散事件不能显著负迁移；
- SSIM/FSS 不能靠额外平滑换取；
- spread 不能靠无技巧噪声做大；
- clamp fraction 必须报告。

### 4.4 统计要求

- 核心模型、backbone-matched DiffCast-like、pixel residual 至少 3 seeds；
- 以 event 为单位 bootstrap 95% CI；
- paired bootstrap 比较同一事件上的差值；
- lead-time 和强度阈值多重比较需预注册主终点或做 FDR/层级检验；
- `report_test` 在方案和 checkpoint 规则冻结后一次性打开。

---

## 5. 分阶段实验路线与止损门

### Phase 0：冻结当前事实（1–2 天）

1. 等远端 intensity run 到预定 checkpoint，不因 loss 波动无限续训；
2. 用固定 K=8、20/50 NFE 在 `val_model` 生成评估；
3. 登记 R1 的 config hash、code commit、checkpoint hash 和结果；
4. 将远端临时 patch 整理为可审查 commit；
5. 删除/移出仓库范围的 core dump 之前先确认其调试价值，不能让它进入版本管理；
6. 同步本地与远端分支，之后所有正式实验从同一 commit 启动。

**Gate 0**：如果 intensity weighting 只降 denoising loss、不提升生成 CSI/CRPS，则停止继续扫权重。

### Phase 1：随机坐标审计（2–4 天）

不训练扩散，只生成 `d*,a*,M_reg`：

- 合成平移恢复误差；
- `Warp(μ,d*)` 相对 `μ` 的结构对齐改善；
- `a*` 在高置信平移区是否显著小于 `Y-μ`；
- 位移饱和率、边界越界率；
- 新生/消散区是否被错误解释为大形变；
- per-lead `d/a` 方差和频谱。

**Gate A**：若配准不能稳定减小高置信区误差，或 `d` 大面积触顶，停止 U-DIP，全力转向“显式时序 pixel residual + history guidance”的保底路线。

### Phase 2：小样本 overfit（1–2 天）

核心组：

| ID | 模型 | 目的 |
|---|---|---|
| R0 | 当前 v-pred pixel residual | 母基线 |
| R1 | R0 + intensity weight | 当前远端诊断 |
| T0 | R0 + explicit temporal path + lead embedding | 排除纯时序欠建模 |
| J0 | 新代码路径中固定 `d=0`，只生成 `a` | 验证代码路径公平 |
| J2 | `(d,a)` 双分支但无 coupling | 换坐标本身 |
| J3 | `(d,a)` + coupling + temporal bottleneck | U-DIP core |

使用 32–64 个样本、2k steps，检查完整采样而非只看 teacher-forced reconstruction。

**Gate B**：J3 必须能 overfit clean reconstruction，完整采样 finite，且位移/强度分支都不是全零或全饱和。

### Phase 3：10k-step 单 backbone pilot（2–4 天）

先只用 SDIR trend，比较 R0、R0-wide、T0、J0、J2、J3。相同：

- global batch；
- optimizer steps；
- NFE 与 K；
- checkpoint 选择规则；
- 参数量或给出 compute-matched 对照。

**Gate C**：J3 相对 J0/T0 至少同时满足：

- 高阈值 CSI 或 CSI-M 有一致正信号；
- centroid/track error 改善；
- CRPS 不超过预注册非劣界；
- 新生/消散子集无明显负迁移；
- 收益不只来自参数量。

### Phase 4：视频扩散式 guidance（2–3 天）

在通过 Gate C 的 J3 上加入：

- per-frame noise masking；
- history guidance；
- deterministic-anchor guidance。

只做一个小型 `w_h × w_μ` 网格，画 Pareto：

```text
x-axis: diversity / spread / CRPS
y-axis: CSI-M / trajectory skill
```

**Gate D**：guidance 必须改善时序一致性或 skill–spread Pareto；若只让 ensemble collapse，则不进入 hero model。

### Phase 5：多 backbone 与热插拔（3–7 天）

按以下顺序：

1. SDIR + PhyDNet pooled training；
2. 分别与 single-backbone adapter 比较；
3. 加第三个 backbone；
4. leave-one-backbone-out；
5. 失败时只开放 norm/LoRA few-shot adaptation。

**Gate E**：pooled adapter 在 seen backbones 上不能显著弱于各自专用 adapter；held-out 若失败，降级论文措辞，不能硬称 one-checkpoint universal。

### Phase 6：正式训练与外部验证

SEVIR 只能证明标准协议有效。CVPR 故事最好至少再有一个真实雷达域：

- 优先复用 DiffCast 已使用且许可/数据可获得的 HKO-7、Shanghai 或其他公开 radar benchmark；
- 若数据准备来不及，至少完成 `13→12@384` 独立扩展，但这不能完全替代跨数据集验证；
- 两种协议不得混表；
- 外部域至少有 deterministic baseline、DiffCast-like、U-DIP、关键消融。

---

## 6. 工程迁移方案

### 6.1 新增目录

```text
src/phyrd/models/probabilistic/udip/
  __init__.py
  model.py                 # ProbabilisticModel 接口
  decomposition.py         # Warp、d/a 重建、坐标规范
  registration.py          # d* 与 M_reg 生成
  denoiser.py              # 双分支时空网络
  diffusion.py             # per-frame noise/v-pred/DDIM
  conditioning.py          # history/anchor control
  guidance.py              # history + anchor guidance

scripts/
  precompute_trend_bank.py
  precompute_udip_targets.py
  audit_udip_decomposition.py

configs/diagnostics/
  udip_overfit_5to20.yaml
  udip_pilot_5to20.yaml

configs/active/5to20/
  train_ddp8_udip_sdir_seed42.yaml
  train_ddp8_udip_pool_seed42.yaml
```

不要新建独立 trainer；通过 registry 接入现有 `scripts/train.py` 与统一 evaluate CLI。

### 6.2 接口约束

```python
training_loss(history, target, trend) -> dict[str, Tensor]
sample(history, trend, ensemble_size, sampling_steps, ...) -> [B,K,T,1,H,W]
```

U-DIP checkpoint 必须保存：

- protocol hash；
- trend bank/backbone checkpoint hashes；
- `d/a` normalization；
- registration config hash；
- denoiser/diffusion/guidance config；
- git commit；
- train backbone list 与 sampling weights。

### 6.3 远端实验规范

每个实验目录固定为：

```text
artifacts/experiments/{model}/{timestamp}_{seed}/
  config_snapshot.yaml
  manifest.json
  checkpoints/
  metrics/
  predictions/
  diagnostics/
  console.log
```

禁止：

- 用 `patch_intensity.py` 一类临时脚本修改后直接启动正式实验；
- 不提交代码就训练；
- 复用/覆盖 B0、B1、R1 artifact；
- 在 `report_test` 上选 checkpoint 或 guidance；
- 将 core dump、大模型和预测 NPZ 纳入 Git。

### 6.4 842 的资源建议

表示审计和 overfit 用 1 卡；10k pilot 用 4 卡；通过 Gate 后再上 8 卡。起始配置：

```yaml
precision: bf16
prediction_type: v
diffusion_steps: 100
sampling_steps_pilot: 50
ensemble_size_val: 4
ensemble_size_report: 8  # 最终补 K=16 scaling
physics.enabled: false
```

batch size 以显存探针决定，但所有核心对照保持相同 global batch 或使用梯度累积对齐。记录 allocated/reserved peak、samples/s、NFE latency。

---

## 7. 风险、失败模式与备选路线

### 风险 1：`d/a` 不可辨识

总可以取 `d=0,a=Y-μ`，也可能用大形变减少 `a`。

对策：低分辨率、有界、平滑、时序连续的操作性规范；置信图；`J0/J2/J3`；打乱 `d*`；不做物理语义宣称。

### 风险 2：配准伪标签限制上限

对策：Gate A、合成平移、按 `M_reg` 分层、报告收益与伪标签质量相关性。若失败，保底采用 T0：显式时序 pixel residual + DFoT-style history/anchor guidance。

### 风险 3：SDIR 已经很 sharp，概率修正容易伤 CSI

对策：zero-init anchor control、预测 correction gate：

```text
Y_k = μ + g(X,μ,lead) ⊙ [Warp(μ,d_k)-μ+a_k]
```

`g` 初始接近 0，并加轻量 magnitude regularization。它是安全阀，不应被包装成主要创新。

### 风险 4：ensemble CRPS 改善只是噪声变大

对策：spread–skill、rank histogram、Brier/reliability、trajectory spread、coverage-width；同时保留 ensemble mean skill guardrail。

### 风险 5：pooled training 拉低强 backbone

对策：balanced sampling、style norm、backbone-ID dropout；若仍失败，用 shared trunk + 低秩 backbone adapter，论文定位改为“快速可适配”而不是“完全零样本”。

### 风险 6：视频模型太重，实验周期失控

对策：先 U-Net/DiT hybrid，注意力只放低分辨率；Sparse VideoGen/latent/rectified flow 都放到后续，不同时改变表示、架构、采样器和压缩空间。

### 立即停止条件

出现任一项，停止全量训练并回到上一 Gate：

1. `d` 大面积为零或触及最大位移；
2. J3 不胜 J0/T0/R0-wide；
3. 收益只在纯平移事件存在，新生/消散显著退化；
4. ensemble mean 指标改善但 CRPS/reliability 持续变差；
5. 只有打开旧 physics guidance 才能取得收益；
6. held-out backbone 崩溃却仍准备宣称 universal one-checkpoint；
7. 结果依赖 test-set 调 guidance、threshold 或 checkpoint。

---

## 8. 最小可发表版本与增强版本

### 8.1 最小可发表版本（应优先完成）

- SEVIR `5→20@128`；
- SDIR、PhyDNet、第三个 backbone；
- U-DIP 随机坐标；
- explicit temporal denoiser；
- history + anchor guidance；
- official / backbone-matched / compute-matched DiffCast 对照；
- 3 seeds + event bootstrap；
- CSI/CRPS/可靠性/轨迹四类证据；
- 一个 held-out-backbone 或 few-shot adaptation 实验。

### 8.2 增强项优先级

按收益/风险排序：

1. 第二雷达数据集；
2. `384×384` 扩展；
3. sampled CRPS/Brier fine-tune；
4. sparse attention / NFE distillation；
5. NWP/卫星等多模态 condition adapter；
6. latent diffusion / rectified flow 替换。

不要在主路线未通过前融入所有模块。CVPR 更看重一个清楚、可证伪、归因完整的视觉时空生成问题，而不是模块堆叠。

---

## 9. 建议的论文标题与摘要骨架

工作标题：

> **Beyond Pixel Residuals: A Hot-Swappable Deformation–Intensity Diffusion Adapter for Probabilistic Precipitation Nowcasting**

摘要骨架：

1. 确定性 nowcaster 技巧强但不能表达多模态未来；
2. 像素 residual diffusion 把位移误差编码为高幅正负强度残差；
3. 提出 U-DIP，在确定性趋势上联合采样 deformation correction 与 intensity innovation；
4. 引入 history/anchor-guided per-frame diffusion，实现时序一致和可控 diversity；
5. 一个 adapter 跨多个 deterministic backbones 使用；
6. 在公平 DiffCast 协议上同时改善 high-intensity skill、trajectory distribution 与 CRPS/reliability；
7. 外部域/高分辨率验证泛化和效率。

最终是否能写第 6、7 条，完全由实验决定。

---

## 10. 文献迁移边界

以下是本方案直接依赖的已核验方向：

- [DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting](https://arxiv.org/abs/2312.06734)：确定性全局运动 + 随机局地变化的 residual diffusion 母基线，也是必须公平超越的对象。
- [History-Guided Video Diffusion](https://proceedings.mlr.press/v267/song25b.html)：DFoT、逐帧 noise-as-mask 与 history guidance；迁移重点是灵活历史条件分数，而不是笼统的 temporal attention。
- [Sparse Video-Gen](https://proceedings.mlr.press/v267/xi25c.html)：空间/时间 attention head 稀疏性与推理加速；放在 hero 模型通过后的效率扩展。
- [Probabilistic Precipitation Nowcasting with Rectified Flow Transformers](https://arxiv.org/abs/2605.31204)：强调 uncertainty-preserving compression、frame-wise encoder 与 unified video decoder，说明确定性压缩会限制概率表达；用于 latent 扩展与强新基线边界。
- [AIFS-CRPS](https://arxiv.org/abs/2412.15832)：直接用 CRPS 类目标训练 ensemble 的依据；用于后期概率分数微调，不作为 U-DIP 主创新。

昨天检索清单中的 Astra、DiTS、LatentTSF、FLAM 等可以继续跟踪，但对当前最小方案不是必要依赖。尤其是动作世界模型、超长自回归和大规模多模态模块，不应因为“新”就直接搬进 `5→20` SEVIR。

---

## 11. 最终建议

**Go：** 保留现有 universal composer/registry/evaluation 工程，停止把 physics guidance 当主研究轴；用当前 intensity run 收尾诊断后，立刻并行进入形变–强度坐标 Gate A。

**主路线：** `U-DIP = coupled deformation–intensity coordinate + explicit space–time denoiser + history/anchor guidance + multi-backbone training`。

**保底路线：** 如果形变分解被 Gate A 证伪，则采用 `T0 = explicit temporal pixel residual + per-frame noise/history guidance + multi-backbone adapter`，但论文新意会明显下降，需要用更强的概率训练或跨 backbone 泛化补足。

**最重要的成功信号：** 不是单个样本更锐，而是相对 backbone-matched DiffCast-like，在相同 SDIR 趋势下同时看到：

- 高阈值 CSI/CSI-M 上升；
- centroid/track error 下降；
- CRPS 或 Brier 改善/非劣；
- spread 与 error 匹配而非无技巧放大；
- 同一 adapter 在至少两个 backbone 上成立。

达到这组证据，才真正形成“视频扩散思想迁移到 nowcasting、作为热插拔确定–概率框架的概率部分，并超越 DiffCast”的完整 CVPR 故事。
