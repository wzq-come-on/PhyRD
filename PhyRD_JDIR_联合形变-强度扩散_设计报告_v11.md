# PhyRD-JDIR 设计报告 v11

> **JDIR: Joint Deformation–Intensity Diffusion for Residual Nowcasting**  
> 核心变化：**不再直接扩散像素残差 `Y-μ`，而是联合扩散“形变修正 + 强度修正”，再通过可微 warping 重建未来雷达序列。**

- 文档状态：新路线研究设计，不修改 v10.3 已冻结方案
- 日期：2026-07-21
- 主协议：SEVIR VIL `5→20@128`
- 确定性骨干：冻结 SDIR，`μ=Pθ(X)`
- 新路线定位：概率生成结构改造，**不使用 physics loss，不使用推理期 physics guidance**
- 文件隔离：本文件是新建报告，不覆盖 `idea_report_master_v10.md`、`implementation_v10.md` 或 `user_requirements_v10.md`

---

## 0. 结论先行

我建议新扩散路线选 **JDIR：联合形变—强度扩散**。

它不是换 `epsilon-prediction`/`v-prediction`，也不是再加一个物理 loss。它改的是扩散学习的**随机变量表示**：

```text
现有方法：  Y = μ + r
               扩散直接学 r = Y - μ

JDIR：        Y ≈ Warp(μ, d) + a
               联合扩散 d（位置/形变修正）和 a（强度/生消修正）
```

选它的直接原因是：强回波只要位置错几个像素，`Y-μ` 就会变成一对高幅值的正负尖峰。这种“几何错位被编码成强度残差”对扩散很难学，也容易使最终样本在 CSI、轨迹和强回波上不升反降。JDIR 把位置错误放回形变坐标，把新生、消散和强度变化留在幅值坐标。

这条路线的优点：

1. 直接针对当前项目要改善的轨迹、空间错位、强回波和长 lead；
2. 与 v10 的“不可靠物理算子下的 reliability-gated guidance”是不同研究问题；
3. 可以继续使用已训好并冻结的 SDIR，不需要重做 benchmark；
4. 能用清晰的对照实验证明收益是来自表示改变，而不是参数变多。

但它不是低风险换壳。最大风险是形变 `d` 与强度修正 `a` 并不唯一，不良的配准目标会让 `a` 吞掉全部误差，或让 `d` 产生假位移。因此必须先通过形变表示 Gate，再上八卡正式训练。

---

## 1. 从当前代码看，问题不只是“训崩了”

### 1.1 当前生成变量是完整像素残差

当前 `PhyRDModel.diffusion_loss()` 计算：

```python
trend = SDIR(history)
residual = target - trend
loss = diffusion.training_loss(residual, history, trend)
```

这等价于让一个生成器同时学习：

- SDIR 的系统偏差；
- 降水团的位置误差；
- 轮廓扩张、收缩、分裂和合并；
- 强度增强、衰减、新生和消散；
- 真实条件分布中的多种可能未来。

这些误差在像素坐标下的统计形态差别很大，却被同一个 isotropic Gaussian noising 过程处理。

### 1.2 几何错位会被放大成正负残差

以一个尖锐雨团 `g(x)` 为例。若真值只是比 SDIR 预测平移 `δ`：

```text
Y(x) = g(x-δ),   μ(x)=g(x)
```

对小位移，像素残差近似为：

```text
Y(x)-μ(x) ≈ -δ·∇g(x)
```

它不是一个小而平滑的“随机细节”，而是沿强回波边界成对出现的高幅值正负双极残差。它对位置非常敏感，但普通 MSE 噪声预测并不直接知道“这是同一个雨团的错位”。

这能解释一种现象：扩散 loss 可以下降，样本也不再数值崩溃，但最终 CSI 和轨迹仍然不升。

### 1.3 当前时间建模还有一个工程缺口

当前 `ResidualDenoiser` 把 `20` 个未来帧压成 `20` 个 2D 通道，再用普通 `UNet2D` 一次输出。它没有显式的帧间共享、lead embedding 或时序路径混合。

需要注意：**增加时序注意力本身不是新意**。DiffCast 原文的 GTUNet 已经使用 ConvGRU/时序注意和 residual segment consistency。在 JDIR 里增加显式时序混合，只是为了让新表示工作，不能作为论文 hero claim。

---

## 2. 方法：Joint Deformation–Intensity Diffusion

### 2.1 输入与冻结趋势

沿用当前数据契约：

```text
X ∈ [0,1]^[B,5,1,128,128]
Y ∈ [0,1]^[B,20,1,128,128]
μ = stopgrad(SDIR(X))
```

SDIR checkpoint 全程冻结，JDIR 所有对照共享同一个 `μ`。

### 2.2 换坐标：从加性像素残差到形变—强度对

定义：

```text
d ∈ R^[B,20,2,h,w]       低分辨率稠密形变修正，建议 h=w=32
a ∈ R^[B,20,1,128,128]  有符号的强度/外观修正
```

重建操作：

```text
Ŷ = clip(Warp(μ, Up(d)) + a, 0, 1)
```

`Warp` 使用可微 bilinear grid sampling；`d` 在 `32×32` 上生成，再平滑上采样到 `128×128`，避免学出逐像素抖动。为避免尺度歧义，`d` 始终以 **128×128 网格的 pixel unit** 存储：从 `32×32` 上采样时只插值场值，不再乘 4；送入 `grid_sample` 前才按 `2d_x/(W-1), 2d_y/(H-1)` 转成 normalized coordinates。输出使用 `d=d_max*tanh(d_raw)` 做有界参数化，`d_max` 只在 `val_model` 上选择，起始诊断值为 16 个 128-grid pixels。

这里的 `d` 只称为 **deformation correction**，不宣称它是真实大气光流；`a` 是形变后的剩余强度修正，不宣称它等于真实微物理源汇。

### 2.3 训练目标如何得到

真实数据没有 `d` 标签。首先在 **train split 内**对 `μ` 与 `Y` 做冻结的无监督/自监督配准：

```text
d* = argmin_d  ρ_struct(Y, Warp(μ,d))
              + λ_smooth ||∇d||_1
              + λ_time   ||d_τ-d_{τ-1}||_1
              + λ_mag    ||d||_1
```

然后定义精确余项：

```text
a* = Y - Warp(μ,d*)
```

实现上建议：

1. `d*` 在 `32×32` 网格上优化；
2. `ρ_struct` 使用稳健强度项 + 局部梯度/结构项，不仅用 raw photometric MSE，避免把真实强度变化错当位移；
3. 对雨团内部纹理太少、边界越界或配准前后不一致区域，生成 `M_reg` 置信图；`M_reg` 不能只由强度 residual 定义；
4. `M_reg` 只用于降低不可辨识区域的 `d` 监督权重，不把低置信区强行设为无运动；
5. `a*` 始终用上式计算，保证分解可重建；
6. 配准程序不读 `val_calib` 和 `report_test` 来选参。

`Y` 出现在训练标签构造中不是泄漏；但在推理时，模型输入严格只能是 `X`、`μ` 和从它们提取的特征。

### 2.4 联合扩散变量

对 `d*` 和 `a*` 分别使用 train-only、per-lead/per-component 的中心与尺度：

```text
z_d = (d* - c_d) / s_d
z_a = (a* - c_a) / s_a
z_0 = {z_d, z_a}
```

首版两个分支使用**同一个 diffusion timestep `t` 和同一条 cosine schedule**，保持因果对照简单。不在首版同时加入多噪声日程、flow matching 或物理引导。

训练继续使用已经修复的 `v-prediction`：

```text
z_t = α_t z_0 + σ_t ε
v_target = α_t ε - σ_t z_0
```

形变与强度噪声必须在同一样本内成对生成和反演，不能把一个 member 的 `d` 和另一个 member 的 `a` 随机拼接。

### 2.5 双分支时空 denoiser

建议架构：

```text
                         +----------------------+
z_d(t), 32x32 ---------->| deformation branch   |--- v_d
                         |                      |
X, μ --> condition ---->| cross-scale coupling | 
                         |                      |
z_a(t),128x128 --------->| intensity branch     |--- v_a
                         +----------------------+
                                  |
                       temporal mixing at bottleneck
```

具体约束：

- `d` 分支：低分辨率 U-Net，输出每个 lead 的 2D deformation correction；
- `a` 分支：像素级 U-Net，保留强回波边界和新生/消散能力；
- 在 `32×32` 及更低分辨率交换特征，让“运动到哪里”与“在那里增强或衰减”联合建模；
- 时间维保持显式 `[B,T,C,H,W]`，并在低分辨率 bottleneck 使用共享 temporal attention 或 temporal convolution；
- 所有 lead 使用共享空间权重 + lead embedding，不再把 lead 只当作无序通道；
- condition encoder 只读 `X`、`μ` 及其 train-only 统计归一化结果。

这不是要强制运动与强度独立；恰恰相反，它们通过共享 bottleneck 和 cross-branch block 显式耦合。“分解输出坐标，但联合建模分布”是 JDIR 与硬拆独立过程的区别。

### 2.6 训练损失

核心损失：

```text
m_reg = m_floor + (1-m_floor) * M_reg,  0 < m_floor < 1
L_vd = sum(m_reg * (v_d-v_d*)^2) / sum(m_reg)
L_va = mean((v_a-v_a*)^2)
L_v = w_d * L_vd + w_a * L_va
```

由于两个分支已标准化，首版使用 `w_d=w_a=1`，不人为把运动分支权重调得更大。`m_reg` 是形变分支内的空间可靠性权重，与分支间标量权重 `w_d` 不是同一个量。`m_floor` 保证低置信区仍有有限监督，其值只在 `val_model` 上选择。`M_reg` 仅在训练/诊断时使用，不是推理输入。

只在中低噪声区域 `t <= t_rec`对预测的 clean pair 重建成雷达图：

```text
Ŷ_0 = Warp(μ, d̂_0) + â_0
L_rec = Charbonnier(Ŷ_0, Y)
      + λ_grad * Charbonnier(∇Ŷ_0, ∇Y)
```

总损失：

```text
L = L_v + 1[t <= t_rec] * λ_rec * L_rec
```

实施原则：

- 先训 `L_v` 纯生成基线，确认完整采样稳定；
- `λ_rec` 从 `0.01` 级别起做梯度比例诊断，不允许辅助项早期主导；
- `L_rec` 必须作用于模型生成的 `d̂_0,â_0`，不能在 target pair 上计算与生成器无关的假损失；
- 首版完全关闭 `physics.enabled`；
- 不在首版加不受控的高强度像素加权，先检验表示本身。

### 2.7 采样与 ensemble

从联合高斯噪声启动，用 v-prediction DDIM 同步生成 `(d,a)`：

```text
for k in 1...K:
    (d_k, a_k) ~ p_JDIR(d,a | X,μ)
    Y_k = clip(Warp(μ,d_k) + a_k, 0, 1)
```

概率评估至少使用 `K=8`，正式表建议 `K=16`，并显式报告：

- ensemble mean 的 CSI/HSS/MAE/SSIM/FSS；
- empirical ensemble CRPS；
- 固定 VIL 阈值的 Brier score/reliability；
- spread-skill ratio 与 rank histogram；
- 每个 lead 的 deformation spread、intensity spread 和最终像素 spread；
- final clamp fraction，防止用大量截断隐藏采样失真。

### 2.8 可选增强：概率分数微调

只有 JDIR 纯 `L_v` 通过采样 Gate 后，才考虑用 `K=4` 的短步可微 DDIM 做小学习率微调：

```text
L_prob = CRPS_K(Y_1:K, Y)
       + λ_brier * Σ_q Brier(P(Y>q), 1[Y>q])
```

可在 pixel、pool4 和 pool16 统计量上定义，但必须保持阈值、领域和样本加权的预注册。该步是可选的概率质量增强，不是 JDIR 的首个可行性 Gate。

---

## 3. 这与原 v10 物理路线是什么关系

| 维度 | v10.3 物理路线 | JDIR 新路线 |
|---|---|---|
| 主问题 | 物理算子不可靠时，何时应当减弱/关闭引导 | 像素残差把几何错位和强度变化混在一起 |
| 核心手段 | `C_flow/M_nadv` + weak transport + gated guidance | 形变—强度联合生成坐标 |
| 是否在采样中求物理梯度 | 是（B2/B4 等） | 否 |
| 是否依赖未来 flow 可靠性 | 是 | 否 |
| 形变场的语义 | 弱输运算子 | 为重建服务的生成坐标，不声称真实 flow |
| 论文归因 | 可靠性门控能否避免负迁移 | 换坐标能否改善轨迹、强回波和概率质量 |

结论：两者理论上可以后续组合，但首轮绝对不组合。否则无法判断改善来自新表示还是物理引导。

---

## 4. 为什么不选其他看起来更简单的方案

### 4.1 不选“均值修正 head + 随机残差扩散”作为主创新

这个方案在工程上合理，但与已有工作过于接近：

- DiffCast 已经宣称 global deterministic motion + local stochastic residual；
- CoST 明确提出 conditional mean–residual decomposition 和 scale-aware diffusion；
- CorrDiff 等大气 downscaling 方法也是 mean predictor + stochastic residual correction。

在 PhyRD 再加一个 bias head 可以作为对照或工程增强，但不适合单独支撑 ICLR 级主张。

### 4.2 不选“只加 temporal attention”

DiffCast GTUNet、RainDiff 和 FREUD 都已经强调时空依赖、token attention 或 unified temporal decoder。对当前 clean port 补时序结构很有必要，但它是母基线实现完整性，不是足够独立的新意。

### 4.3 不选“小波/频率扩散”作为当前主路线

当前 SDIR 已是频率解耦的确定性骨干；2025–2026 也已出现 wavelet diffusion、frequency-controlled diffusion 和针对极端降水的 WADEPre。再做频段噪声日程容易与确定性分支撞车，归因也不干净。

### 4.4 不选“把 DDPM 换成 flow matching”作为主创新

v10 文档已把 residual conditional flow matching 写成生成引擎迁移实验。只换采样方程无法解决“像素残差表示是否合适”的问题，也不足以解释为什么会改善轨迹和强回波。

---

## 5. 842 上的实验路线

### 5.1 不直接一步全训

842 是 8 卡机，但 JDIR 的新变量和双分支结构尚未通过 Gate。最有信息量的使用方式是：

```text
表示审计 -> 2k-step overfit/smoke -> 10k-step pilot -> 完整训练
```

这不是为了少用卡，而是为了防止八卡连续数天在一个已经可被 2k step 证伪的表示上浪费时间。

### 5.2 实验矩阵

| ID | 模型 | 作用 |
|---|---|---|
| `R0` | 当前 v-pred pixel residual，physics off | 母基线；必须先确认不再数值崩溃 |
| `R0-wide` | 参数/显存匹配的加宽 pixel residual | 排除“只是 JDIR 参数多” |
| `J0` | `d=0`，只扩散 `a=Y-μ` | 在新代码路径中复现像素残差 |
| `J1` | 只生成 `d`，`a=0` | 评估纯几何修正的上限与缺口 |
| `J2` | 联合 `(d,a)`，两分支不交互 | 验证换坐标本身 |
| `J3` | 联合 `(d,a)` + cross-branch coupling + temporal bottleneck | **JDIR hero** |
| `J4` | `J3` + 中低噪声 `L_rec` | 验证重建监督是否有净收益 |
| `J5` | `J3/J4` + sampled CRPS/Brier fine-tune | 概率质量增强，只在前面 Gate 通过后开 |

不允许用当前 physics B1 checkpoint 直接续训 JDIR；它的生成变量和 state dict 语义不同。可以共享冻结 SDIR checkpoint，但 JDIR 生成器必须新建 artifact。

### 5.3 建议启动顺序

1. **P0 形变目标生成审计**  
   在 train/val_model 小样本上运行配准，保存 `d*`、`a*`、`M_reg` 和诊断图；不训扩散。

2. **P1 32–64 样本 overfit**  
   `J0` 和 `J3` 各做 2k step，检查 v-loss、clean reconstruction、形变幅度、`a` 尺度和完整采样。

3. **P2 10k-step 开发 pilot**  
   只跑 `R0`、`J0`、`J3`，在 `val_model` 上用固定 K/NFE 比较，先回答换坐标有没有信号。

4. **P3 完整训练**  
   只对通过 P2 的组开全量 100 epoch，保留 best/last，用 `val_model` 选 checkpoint。

5. **P4 概率微调与正式评估**  
   如需要 `J5`，先用 `val_model` 选权重；`val_calib` 只用于风险/后校准；`report_test` 在方案冻结后一次性评估。

### 5.4 842 资源起始配置

新模型比当前 20-channel U-Net 更重，不应盲目沿用 `batch_size=32/rank`。建议起点：

```yaml
protocol: 5to20_128
world_size: 8
precision: bf16
batch_size_per_rank: 8
gradient_accumulation: 1  # 显存允许再升
diffusion_steps: 100
prediction_type: v
sampling_steps_pilot: 50
physics.enabled: false
```

显存探针后优先把 per-rank batch 递增到 `12/16`，但要保持对照组的 global batch 一致。必须记录 peak allocated/reserved memory，不根据 `nvidia-smi` 的低利用率盲目放大 batch。

### 5.5 新文件和 artifact 命名

建议新建：

```text
src/phyrd/models/jdir/
  decomposition.py
  registration.py
  denoiser.py
  diffusion.py

scripts/precompute_jdir_targets.py
scripts/train_jdir.py
scripts/evaluation/evaluate_jdir.py

configs/active/5to20/train_ddp8_jdir_5to20_vpred_seed42.yaml
configs/diagnostics/jdir_5to20_overfit.yaml

artifacts/experiments/phyrd_jdir_5to20_j3_seed42/
```

每个实验 ID 使用独立 artifact 目录，不覆盖：

- SDIR checkpoint；
- 已有 B0/B1 residual checkpoint；
- 旧 physics artifact；
- 其他 seed 的 JDIR checkpoint。

---

## 6. Gate 和停止条件

### Gate A：分解不是伪变量

在 `val_model` 小样本上：

- `Warp(μ,d*)` 在高 `M_reg` 区域的对齐误差必须显著低于 `μ`；
- `d*` 不能大面积顶到最大位移边界；
- 对已知平移的合成雨团，`d*` 能恢复位移方向与量级；
- 在高置信平流区，`a*` 的绝对值应小于原始 `Y-μ`；
- 新生/消散区不能被强行解释成超大形变。

若不满足，停在配准层，不开 JDIR 全训。

### Gate B：数值稳定

- `z_d/z_a` 每个 lead 的 std 在同量级；
- `v` target、model output、predicted clean pair 均为 finite；
- 100-step 与 50-step 采样不出现变形爆炸、黑图或大面积 clamp；
- 无 physics 情况下生成预测与 target 的相关系数显著大于零；
- 完整采样指标而不是只看 denoising loss 来选 checkpoint。

### Gate C：新表示有净收益

`J3` 必须在同 checkpoint 规则、同 ensemble size、同 NFE 下比 `J0` 至少满足：

- 高阈值 CSI 或 CSI-M 提升；
- centroid/object track error 降低；
- FSS/CSI-pool 不依赖额外平滑取得；
- CRPS 不超过预注册非劣界；
- 对新生、消散、增强、衰减事件分层报告，不能只报纯平移案例。

### Gate D：不是参数和伪标签效应

- `J3` 需要胜过参数匹配的 `R0-wide`；
- 打乱 `d*` 与样本对应关系应破坏收益；
- 只用 `a` 的 `J0` 不应与 `J3` 完全等价；
- 去掉 cross-branch coupling 的 `J2` 用于验证联合建模是否必要；
- 伪形变目标的质量与最终收益要做条件分析。

### 立即停止条件

任一情况出现时，不应继续堆超参：

1. `d` 大面积为零，几乎所有误差都由 `a` 生成；
2. `d` 大面积达到位移上限，通过极端形变欺骗重建损失；
3. `J3` 只提升 ensemble mean 的平滑指标，但 CRPS/可靠性持续变差；
4. `J3` 不胜 `J0` 或 `R0-wide`；
5. 收益只在高 `M_reg` 纯平移事件存在，对新生/消散显著负迁移；
6. 为得到收益必须同时加 physics guidance，导致新表示无法独立归因。

---

## 7. 创新边界与 ICLR 可能性

### 7.1 可以尝试的主张

如果实验支持，可将论文主张收紧为：

> **Additive pixel residuals are a poorly conditioned stochastic coordinate for sharp precipitation systems because displacement errors become high-amplitude signed residuals. JDIR instead learns a coupled conditional distribution over low-resolution deformation corrections and high-resolution intensity innovations, improving trajectory and extreme-echo skill without relying on inference-time physical gradients.**

中文：

> 对尖锐降水系统，加性像素残差会把位移误差编码为高幅值正负残差，因而不是理想的随机生成坐标。JDIR 联合建模低分辨率形变修正与高分辨率强度创新，在不使用推理期物理梯度的情况下改善轨迹和强回波概率预报。

### 7.2 不能宣称的内容

- 不能说首次将 diffusion 用于降水短临；
- 不能说 deterministic + stochastic decomposition 本身是新意；
- 不能说 temporal attention 或 two-branch U-Net 本身是新意；
- 不能把 `d` 当成真实风场，也不能把 `a` 当成真实微物理源汇；
- 不能因为 `d/a` ensemble 有 spread 就声称已完整分解 aleatoric/epistemic uncertainty；
- 不能在没有 `R0-wide`、`J0`、`J2` 的情况下把收益归因于 JDIR。

### 7.3 对 ICLR 的真实判断

这个设计比“再加一个物理 loss”或“再加一个 bias head”更有模型结构和可证伪的研究问题，也更容易解释为什么应该改善轨迹与强回波。

但仅凭方法构想不能判定“够 ICLR”。至少需要：

1. 在同一 `5→20@128` 协议上稳定胜过强的 DiffCast-like/RainDiff-like 残差对照；
2. 同时拿到强回波、轨迹/FSS 和 CRPS/可靠性证据；
3. 用参数匹配、伪形变打乱和新生/消散分层做出机制归因；
4. 至少 3 seeds + event bootstrap；
5. 完成正式、系统的 related-work 检索，确认没有同一表示的近期并行工作；
6. 最好在一个现有外部数据集上复核，但不需要新建 benchmark。

---

## 8. 自检：这个设计哪里可能不合理

### 8.1 `d` 与 `a` 不可辨识

对任意 `Y`，可以选 `d=0, a=Y-μ`，也可能用一个很大的 `d` 减小 `a`。所以不存在无条件的唯一分解。

对策：

- 用低分辨率、平滑、时序连续、小幅度的 `d*` 定义一个**操作性规范**；
- 用 `M_reg` 标出可靠区，对不可辨识区不做过度物理解释；
- 论文只宣称它是有用的生成坐标，不声称因果解耦。

### 8.2 配准偏差可能成为上限

如果 `d*` 本身错了，扩散可能只会学会复制伪形变。

对策：先做 Gate A；在评估中报告性能随 `M_reg` 的条件曲线；对已知位移的合成雨团做小型单元/诊断实验。这不是建新 benchmark。

### 8.3 双分支可能增加成本却没有收益

对策：使用 `R0-wide`、`J0`、`J2`、`J3` 四组核心对照；在同参数、同 NFE、同 global batch 下比较。

### 8.4 形变可能损伤新生/消散

如果训练强迫所有变化都由 `d` 解释，就会重复旧物理路线的负迁移问题。

对策：`a` 是必需分支；低 `M_reg` 区降低 `d` 监督权重；必须单独报新生、消散、增强和衰减子集。

### 8.5 像素 CRPS 好不代表路径分布好

对策：概率评估之外，报告 centroid track、component matching、lead-wise displacement spread、FSS 和 pooled-threshold reliability。

### 8.6 “形变 + 强度”可能已有近期并行工作

本报告在 2026-07-21 对 DiffCast、PreDiff、RainDiff、FREUD、STLDM、StormDiT、CoST、WADEPre 及若干运动轨迹扩散工作做了初步边界检索，尚未发现与本报告完全同构的降水短临方法。但这不是系统文献综述，因此当前只能写“novelty hypothesis”，不能写“first”。

---

## 9. 初步文献边界（不代替正式 related work）

- [DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting](https://arxiv.org/abs/2312.06734)  
  已有 deterministic motion + stochastic pixel residual 和 GTUNet；因此 JDIR 不把“确定性 + 残差”或“时序 attention”当新意。

- [PreDiff: Precipitation Nowcasting with Latent Diffusion Models](https://arxiv.org/abs/2307.10422)  
  已有 latent diffusion 和去噪期知识对齐；JDIR 不使用知识/物理梯度作为主创新。

- [Collaborative Deterministic–Probabilistic Forecasting / CoST](https://openreview.net/forum?id=dg1npGNK0d)  
  已有 conditional mean–residual decomposition 和 scale-aware diffusion；因此本报告放弃“多加一个均值修正 head”的原候选。

- [RainDiff: End-to-end Precipitation Nowcasting via Token-wise Attention Diffusion](https://arxiv.org/abs/2510.14962)  
  已有 pixel-space residual diffusion 与 token-wise attention；JDIR 必须通过随机变量表示和对照证明区分。

- [Probabilistic Precipitation Nowcasting with Rectified Flow Transformers / FREUD](https://arxiv.org/abs/2605.31204)  
  已有 uncertainty-preserving compression 和 unified video decoder；只换成 rectified flow 或统一解码器不足以构成本项目新主张。

- [STLDM: Spatio-Temporal Latent Diffusion Model for Precipitation Nowcasting](https://arxiv.org/abs/2512.21118)  
  已有确定性条件网络 + latent enhancement 两阶段结构；JDIR 的差异必须落在 deformation–intensity joint coordinate。

- [StormDiT: A generative AI model bridges the 2–6 hour gray zone](https://arxiv.org/abs/2601.20342)  
  它对“确定性平流 + 随机扩散”的硬切割提出了质疑。JDIR 因此不独立生成两个互不交互的过程，而是在共享潜空间中联合建模 `p(d,a|X,μ)`。

- [What Happens Next? Anticipating Future Motion by Generating Point Trajectories](https://openreview.net/forum?id=t1vMYl1yhe)  
  说明直接生成运动轨迹可比生成像素更高效地表达运动不确定性，但该工作不是降水短临，也不包含 JDIR 的强度分支。

- [AIFS-CRPS](https://arxiv.org/abs/2412.15832)  
  表明直接用严格/近似公平概率分数训练可以改善天气 ensemble；因此 JDIR 的 CRPS/Brier 微调应被定位为概率质量增强，而不是单独的新意。

---

## 10. 最终建议

### Go

建议继续到代码与 842 pilot，但按以下顺序：

```text
P0 形变目标审计
-> J0/J3 小样本 overfit
-> R0/J0/J3 10k-step pilot
-> 通过 Gate 后才开全量正式训练
```

我对“这条路线比当前继续调 physics loss 更值得开新实验”的判断是 **Go**；对“它现在已经足够发 ICLR”的判断是 **尚不成立，需要机制对照和正式结果证明**。

### 最小成功信号

在 10k-step pilot 阶段，如果 `J3` 相对 `J0` 同时出现：

- centroid/track error 下降；
- 高阈值 CSI 上升；
- predicted spread 非零且 CRPS 不变差；
- 新生/消散子集无明显负迁移；

就值得让 842 进入完整训练。如果只是 train loss 更低，或只是视觉上更锐，不构成继续投入的信号。
