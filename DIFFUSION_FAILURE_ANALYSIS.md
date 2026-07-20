# PhyRD 残差扩散坍塌分析与修复方案

> 历史实验说明：本文分析的是 v10.1 compact 2D U-Net 确定性 checkpoint 与旧 residual diffusion，不验证 v10.2 SDIR。旧确定性结果只保留作迁移前基线，禁止继续训练或接入新 residual 阶段。

> 状态：诊断报告，不包含代码修改  
> 实验：`phyrd_v10_1_ddp8_seed42_weather30537`  
> 数据协议：SEVIR VIL，`13→12 @ 128×128`  
> 结论日期：2026-07-17

## 1. 结论摘要

这次实验不是“训练 loss 数值爆炸”，而是**训练目标看似收敛、生成采样却结构性坍塌**。

最高置信的直接故障链路是：

1. 当前模型使用 `epsilon-prediction`；
2. cosine 日程在 `t=99` 的 `sqrt(alpha_bar)` 仅为 `0.0004928`；
3. 采样用 `x0=(x_t-sigma*epsilon_hat)/alpha` 反推干净残差，噪声预测误差会被放大约 `2029×`；
4. 20 步 DDIM 从 `99` 直接跳到 `94`，首跳仍会把 epsilon 误差放大约 `158×` 后送入下一状态；
5. 中间没有 residual normalization、`x0` clipping 或 dynamic thresholding；
6. 训练只监控平均 epsilon MSE，没有按 timestep 检查 `x0` 恢复误差，也没有按生成质量选 checkpoint；
7. 最终只在 `trend + residual` 后做 `[0,1]` clamp，大量越界负值会被压成零，形成最终的“黑图式”坍塌。

因此，**主要修复方向不是简单增加 epoch、显存或采样 seed，而是改掉低 SNR 端病态的参数化和训练/验证闭环**。

建议的首选方案是：

- residual 按训练集、按 lead 做标准化；
- 从 epsilon-prediction 改为 v-prediction；
- 在标准化 residual 域加入动态阈值或稳健裁剪；
- 先关闭 physics loss，证明无物理 residual diffusion 能稳定生成；
- 用 100 步采样建立正确性基线，再验证 50/20 步加速；
- 用生成指标而不是仅用 denoising loss 选择 checkpoint；
- 基础扩散通过后，再逐步加入 physics training/guidance。

## 2. 已观察到的故障

### 2.1 确定性分支正常，残差扩散严重退化

全量 `report_test` 共 1466 个事件：

| 指标 | 确定性分支 | 残差扩散输出 | 变化 |
|---|---:|---:|---:|
| CSI ↑ | 0.287410 | 0.000986 | 基本失效 |
| CSI-pool4 ↑ | 0.296836 | 0.010748 | 严重下降 |
| CSI-pool16 ↑ | 0.308485 | 0.061430 | 严重下降 |
| HSS ↑ | 0.361680 | -0.000229 | 接近随机 |
| LPIPS ↓ | 0.328520 | 0.514377 | 明显变差 |
| SSIM ↑ | 0.632822 | 0.379393 | 明显变差 |
| MAE/VIL ↓ | 7.372674 | 18.436913 | 约 2.5 倍 |

当前 `ensemble_size=1`，所以报告中的 CRPS 等于 MAE，只能视为单成员 sanity check，不能代表概率预报质量。

### 2.2 输出分布已经坍塌

预测产物诊断显示：

- 最终残差扩散预测中 `99.7368%` 的像素等于零；
- 残差扩散预测与真值的相关系数为 `-0.00341`；
- 确定性预测与真值的相关系数为 `0.89589`；
- 目标残差标准差约为 `0.05878`；
- 目标修正的平均绝对值为 `7.37 VIL`；
- 最终输出相对确定性预测的平均绝对修正达到 `19.43 VIL`。

注意：现有 NPZ 保存的是 clamp 后的最终预测，没有保存 DDIM 每一步的原始 residual。因此可以确认最终输出被压到零，但尚不能仅凭该 NPZ 还原每一步 raw residual 的符号和幅度。

## 3. 可以排除或基本排除的原因

### 3.1 不是普通的 loss/NaN 数值发散

残差阶段完整训练 4700 steps：

- 所有记录均为有限值；
- `loss_gen` 从约 `1.13` 降至末段均值约 `0.0171`；
- 末段 physics loss 均值约 `0.1312`；
- 没有 NaN/Inf；
- checkpoint 正常保存和加载。

所以“训崩”指的是**生成分布坍塌**，不是优化器直接数值爆炸。

### 3.2 不是确定性趋势模型导致

相同测试集上，确定性分支的 CSI、SSIM、MAE 和相关性都正常。将扩散修正加入以后性能才崩溃，因此问题位于 residual diffusion 训练/采样链路。

### 3.3 不是推理期 physics guidance 直接导致

失败的 `predict_report_test.py` 调用：

```text
model.sample(history, ensemble_size=1, sampling_steps=20)
```

没有传入 `guidance_factory`。因此这次失败不是 proximal guidance 在推理时把样本推坏。

physics loss 在训练时开启，仍可能是放大因素，见第 5 节。

### 3.4 不是“不同 rank seed”本身导致

DDP 的模型初始化保持一致；残差扩散训练中各 rank 使用不同随机噪声流是合理且必要的，否则不同卡会重复同一噪声样本、降低有效 batch 多样性。

实验级 seed 相同不等于所有 rank 必须产生逐元素相同的 diffusion noise。

## 4. 最高置信的直接故障机制

### 4.1 `epsilon→x0` 在最高噪声端病态

当前反演公式为：

```text
x0_hat = (x_t - sigma_t * epsilon_hat) / alpha_t
```

令噪声预测误差为：

```text
delta = epsilon_hat - epsilon
```

则 `x0` 误差为：

```text
x0_error = -(sigma_t / alpha_t) * delta
```

当前 100 步 cosine 日程的关键数值：

| timestep | alpha_bar | sqrt(alpha_bar) | x0 误差放大倍数 |
|---:|---:|---:|---:|
| 80 | 0.0851462 | 0.291798 | 3.43× |
| 90 | 0.0195444 | 0.139801 | 7.15× |
| 94 | 0.00605964 | 0.0778437 | 12.85× |
| 98 | 0.000242857 | 0.0155839 | 64.17× |
| 99 | 0.000000242857 | 0.000492805 | 2029.20× |

目标 residual 的标准差只有 `0.05878`。在 `t=99`，干净 residual 留在 `x_t` 中的标准差信号约为：

```text
0.0004928 × 0.05878 ≈ 2.90e-5
```

也就是说，网络需要在接近纯高斯噪声的输入上完成极高精度的抵消，才能恢复小尺度 residual。平均 epsilon MSE 看起来不大，并不代表这种抵消足够准确。

### 4.2 20 步 DDIM 的首跳会继续放大误差

实际 20 步 schedule 为：

```text
99, 94, 89, 83, 78, 73, 68, 63, 57, 52,
47, 42, 36, 31, 26, 21, 16, 10, 5, 0
```

从 `t=99` 到 `t=94` 时，仅 `alpha_94 * x0_error` 这一项就会把 epsilon 误差放大约：

```text
sqrt(alpha_bar_94) × sigma_99 / sqrt(alpha_bar_99)
≈ 157.96×
```

举例：

- epsilon 误差 `0.001` 会在首跳产生约 `0.158` 的状态误差；
- 目标 residual 的整体标准差只有 `0.0588`；
- 要让首跳误差不超过一个 residual 标准差，`t=99` 的 epsilon 误差需低于约 `3.72e-4`。

现有训练日志只记录所有 timestep 混合后的平均 MSE，没有证据表明 `t=99` 达到了这个精度。

### 4.3 没有中间态范围控制

当前采样器在每步反推出 `clean` 后，未执行：

- residual 标准化域裁剪；
- static `x0` clipping；
- dynamic thresholding；
- 每步 finite/range 检查。

只有最终输出执行：

```text
(trend + residual).clamp(0, 1)
```

这无法修复错误采样轨迹，只会把越界结果压到边界。最终 `99.74%` 的零像素正是这种机制的典型表现。

## 5. 高风险放大因素，但尚未完成因果隔离

### 5.1 physics loss 作用在所有随机 timestep 的 `x0_hat`

训练时先从随机 timestep 反推：

```text
prediction_x0 = trend + clean_prediction
```

随后直接对该预测施加 weak transport loss。高噪声 timestep 的 `clean_prediction` 正是最病态的部分。

训练早期观察到：

- step 20：`loss_gen≈1.283`；
- `loss_phys≈207.79`；
- 乘以 `lambda_train=0.01` 后，physics 项约为 `2.08`，大于生成项。

这说明训练早期 physics 梯度可能主导更新，并且它作用的恰好是高噪声端不稳定的 `x0_hat`。

但尚未训练“完全关闭 physics、其他条件相同”的 residual baseline，所以目前应表述为：

> physics loss 是高风险放大因素，不能在完成 B0 无物理扩散对照前认定为唯一根因。

### 5.2 训练目标与最终生成质量脱节

当前训练只优化并记录平均 epsilon MSE，未记录：

- 分 timestep epsilon MSE；
- 分 timestep `x0` MAE/MSE；
- 每个 lead 的 residual 误差；
- 从纯噪声完整采样后的验证 CSI/MAE/SSIM；
- 输出零/一饱和比例；
- residual 均值、标准差和分位数；
- ensemble spread。

因此即使训练 loss 持续下降，也无法及时发现生成分布已经坍塌。

### 5.3 20 步加速在正确性基线之前使用

训练日程为 100 步，报告直接使用 20 步 DDIM。20 步未必是根因，但它放大了首跳跨度，也让问题更难定位。

必须先证明 100 步采样正确，再将 50/20 步作为速度—质量折中实验。

### 5.4 residual 尺度没有显式标准化

目标 residual：

- mean 约 `-0.00431`；
- std 约 `0.05878`；
- 大量值集中在零附近；
- 不同 lead 的误差尺度不同。

模型却直接用单位高斯作为扩散终点。理论上扩散可以处理这种尺度差异，但在仅 100 步、epsilon-prediction 和 aggressive DDIM 跳步组合下，尺度不匹配会显著增加低 SNR 端恢复小残差的难度。

### 5.5 缺少验证集 checkpoint 选择

当前 checkpoint 按固定 epoch 保存，最终直接使用 `checkpoint_final.pt`。没有证据证明 final checkpoint 的生成质量最好，也没有排除早期 checkpoint 尚未坍塌的可能。

## 6. 修复方案与优先级

## P0：先建立可定位的诊断闭环，不立即重训

在修改训练目标之前，应对现有 checkpoint 做以下只读诊断：

1. **分 timestep 去噪评估**  
   在 `t={0,5,10,20,...,90,94,98,99}` 上分别统计：
   - epsilon MSE；
   - `x0` MAE/MSE；
   - `x0` 分位数与越界比例；
   - 每个 lead 的误差。

2. **采样轨迹审计**  
   对 8–32 个固定样本保存每个 DDIM step 的：
   - `x_t` mean/std/min/max；
   - `epsilon_hat` mean/std；
   - `x0_hat` mean/std/min/max；
   - finite 比例；
   - `trend+x0_hat` 的零/一饱和比例。

3. **采样步数对照**  
   固定 checkpoint、样本和 seed，对比：
   - 100 steps；
   - 50 steps；
   - 20 steps。

4. **checkpoint 轨迹对照**  
   对 step `470/1410/2350/3290/4230/4700` 做小样本生成，判断是从一开始就失败，还是后期训练退化。

这些诊断会决定后续是以参数化问题为主，还是还存在明显的 checkpoint/训练阶段问题。

## P1：首选稳定参数化

### 方案 A：v-prediction（推荐）

将训练目标改为：

```text
v = alpha_t * epsilon - sigma_t * x0
```

反演使用：

```text
x0 = alpha_t * x_t - sigma_t * v_hat
epsilon = sigma_t * x_t + alpha_t * v_hat
```

核心优势：反推 `x0` 不再除以趋近零的 `alpha_t`，直接移除 `2029×` 的病态除法。

这是当前最推荐的结构性修复。

### 方案 B：保留 epsilon-prediction 的最小修复

如果暂时不切换参数化，至少需要同时执行：

- 避免从数值上极端的 `t=99` 直接开始；
- 调整 terminal SNR/beta cap；
- 每步进行 residual-domain `x0` thresholding；
- 对低 SNR timestep 使用适当 loss weighting；
- 先使用完整 100 步采样验证。

单独把 `sampling_steps=20` 改成 100 不足以保证修复，因为 100 步仍会经过病态的 `t=99`。

## P2：标准化 residual

使用训练集统计量，建议按 forecast lead 计算稳健中心与尺度：

```text
z_t = (residual_t - center_t) / scale_t
```

可选统计：

- center：mean 或 median；
- scale：std、MAD 或稳健分位距；
- 为 scale 设置下限，避免弱回波 lead 的尺度过小。

扩散模型训练和采样都在 `z` 域进行，最终再反标准化回 residual 域。

要求：统计量必须只由 train split 得到，并写入 checkpoint/protocol，不能从验证或测试集估计。

## P3：加入每步 `x0` 范围控制

推荐在标准化 residual 域使用 dynamic thresholding，例如按每个样本/lead 的绝对值高分位数缩放，并设置合理上界。

不建议直接在原始 residual 域盲目裁成 `[-1,1]`，因为不同 lead、强回波事件和确定性误差的尺度不一致。

范围控制的目标是防止单步异常污染整条采样轨迹，不是用 clamp 掩盖模型质量问题。因此仍需同时记录 clamp/threshold 触发比例。

## P4：分离基础扩散与物理模块

建议使用严格的阶段门槛：

1. **B0：无 physics residual diffusion**  
   `lambda_train=0`，不使用 inference guidance。必须先证明：
   - 输出不饱和；
   - residual 分布与目标同量级；
   - 至少不把确定性基线灾难性破坏；
   - 多成员 ensemble 不完全塌缩。

2. **B1：physics training fine-tune**  
   仅在 B0 通过后加入。建议：
   - physics 权重按 SNR/timestep 门控；
   - 不对极低 SNR 的不稳定 `x0_hat` 施加强物理梯度；
   - 监控生成质量与 physics violation 的 Pareto 关系。

3. **B2：inference guidance**  
   在无 guidance 和 fixed guidance 都有稳定结果后，再加入 reliability-gated guidance。

不能把“基础扩散自身失败”与“物理引导是否有效”混在同一次实验中解释。

## P5：重建验证与 checkpoint 选择规则

每个 checkpoint 至少在固定 validation 子集上生成并记录：

- CSI、CSI-pool4、CSI-pool16；
- HSS；
- MAE、SSIM；
- residual mean/std/quantiles；
- zero/one saturation ratio；
- prediction-target correlation；
- ensemble spread（K≥8 时）；
- 采样耗时和 NFE。

checkpoint 不能再仅按最低 epsilon training loss 选择。

建议主选择规则：

1. 先满足无坍塌硬门槛；
2. 再按预注册的 deterministic/generative 主指标选择；
3. 最终只在 report_test 评一次。

## 7. 推荐的下一轮实验矩阵

| 实验 | 参数化 | residual标准化 | physics train | sampling | 目的 |
|---|---|---|---|---|---|
| D0 | 当前 epsilon | 否 | 是 | 20 | 已失败参考 |
| D1 | 当前 epsilon | 否 | 否 | 100 | 隔离 physics 与跳步影响 |
| D2 | 当前 epsilon | 是 | 否 | 100 | 检验尺度修复 |
| D3 | v-pred | 是 | 否 | 100 | 推荐基础模型 |
| D4 | v-pred | 是 | 否 | 50/20 | 验证采样加速 |
| D5 | v-pred | 是 | SNR门控 | 最佳步数 | 检验 physics training |
| D6 | v-pred | 是 | 最佳设置 | reliability guidance | 完整 PhyRD |

执行顺序必须是 `D1→D2→D3→D4→D5→D6`。如果 D3 未通过，不应继续调 physics guidance。

## 8. 下一轮训练的硬门槛

### Gate A：单步去噪稳定

- 所有 timestep 输出 finite；
- `t=98/99` 的 `x0` 分位数不出现数量级爆炸；
- 分 timestep `x0` 误差曲线可解释；
- 各 lead 无单通道异常。

### Gate B：无物理完整采样稳定

- zero saturation 不应接近 100%；
- prediction-target correlation 必须显著大于零；
- predicted residual std 与 target residual std 同量级；
- MAE 不允许从确定性 `7.37 VIL` 恶化到当前 `18.44 VIL` 量级。

### Gate C：概率预报有效

- ensemble size 至少 K=8；
- 成员间 spread 非零且随 lead 合理变化；
- 报告真正的 ensemble CRPS；
- K=1 只保留为 MAE sanity check。

### Gate D：物理模块有净收益

- 相同随机 seed、NFE 和计算预算；
- physics 版本不能靠输出变平滑来换取 violation 下降；
- CSI/MAE/SSIM 与 physics violation 同时报出；
- 与无 physics、fixed guidance、conditioning-only 对照。

## 9. 不建议采用的“快速修复”

以下操作可能让结果表面不再全黑，但不能解决根因：

- 只增加训练 epoch；
- 只增大模型或显存占用；
- 只更换 seed；
- 只把最终 clamp 改得更宽；
- 只把 20 步改成 100 步；
- 在基础扩散未稳定前继续增加 physics 权重；
- 用 K=1 的 CRPS 宣称概率建模有效；
- 只看训练 epsilon loss 选择 checkpoint。

## 10. 最终判断

当前 residual checkpoint 不适合继续做正式指标或 physics 归因实验，但确定性 checkpoint 可以保留。

最合理的修复路线不是从 physics guidance 开始调参，而是：

```text
保留确定性模型
→ 诊断现有 checkpoint 的 timestep/trajectory
→ residual 标准化
→ v-prediction
→ 无 physics 的 100-step 基础扩散通过门槛
→ 逐步验证 50/20-step
→ 加入 SNR 门控 physics training
→ 最后验证 reliability-guided sampling
```

在完成上述 D1–D3 之前，应将当前结论表述为：

> PhyRD 的确定性分支有效；当前 residual diffusion 实现因低 SNR 端反演病态和缺少生成式验证而发生采样坍塌。physics loss 可能放大该问题，但其独立因果作用仍需无物理对照确认。

## 11. 证据文件

- `src/phyrd/models/diffusion.py`
- `src/phyrd/models/phyrd.py`
- `scripts/train.py`
- `configs/archive/train_ddp8_residual_sevir.yaml`
- `artifacts/server30537/residual_train_log.json`
- `artifacts/server30537/metrics_report_test.json`
- `artifacts/server30537/metrics_deterministic_baseline.json`
- `artifacts/server30537/prediction_diagnostics.json`
- `artifacts/server30537/predict_report_test.py`
