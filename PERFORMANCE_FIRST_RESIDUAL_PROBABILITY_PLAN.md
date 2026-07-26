# PhyRD 性能优先的残差概率模块迁移方案

更新时间：2026-07-24

## 1. 唯一优先级

本项目不再把轻量、热插拔、结构简洁或单纯的新颖性当作约束。模型是否进入下一阶段，只看两个硬条件：

1. 在同一 `5to20@128 / report_test / 5600 samples` 协议下，`CSI`、`CSI_pool4`、`CSI_pool16`、`HSS`、`SSIM` 必须同时高于匹配的确定性 backbone。
2. 固定案例上的预测必须肉眼可见地更接近 GT，不能出现 TrajRes-DF 的颗粒、破碎边界、随机散点和后期闪烁。

如果只改善 CRPS、只改善 pooled CSI，或仅让结果更锐但 SSIM/结构变差，均判定为失败。

backbone 采用两级策略：PhyDNet 是快速结构验证平台，SDIR 是最终主结果平台。所有候选概率模块先在训练更快的 PhyDNet 上完成训练、全量测试和可视化；只有同时通过五项指标与视觉门槛的结构，才迁移到训练成本更高的 SDIR。论文不能用较弱的 PhyDNet 基线替代 SDIR 主结果，但开发阶段也不应把大量算力浪费在尚未验证的 SDIR 实验上。

## 2. 当前正式基线与失败结论

| 模型 | CSI | CSI_pool4 | CSI_pool16 | HSS | SSIM | 结论 |
|---|---:|---:|---:|---:|---:|---|
| PhyDNet deterministic | 0.269791 | 0.277409 | 0.297413 | 0.347121 | 0.700970 | 快速验证的匹配基线 |
| PhyDNet + TrajRes-DF | 0.254858 | 0.269329 | 0.300721 | 0.326172 | 0.524332 | 仅 pool16 微升，其余四项下降；可视化颗粒严重，淘汰 |
| SDIR deterministic | 0.304496 | 0.304929 | 0.319770 | 0.396627 | 0.748057 | 最终主结果基线 |
| SDIR + residual v-pred | 0.294096 | 0.308822 | 0.335645 | 0.380774 | 0.702224 | pooled CSI 升，但像素 CSI/HSS/SSIM 下降，淘汰 |
| SDIR + U-DIP, K=4 | 0.291952 | 0.315490 | 0.349005 | 0.380321 | 0.654896 | pooled CSI 升，但像素 CSI/HSS/SSIM 下降，淘汰 |

因此，目前没有一个已有概率模块通过“五项同时提升”的门槛。

### 2.1 TrajRes-DF 为什么会产生颗粒

这不是抽象猜测，代码中存在三个直接问题：

1. 训练使用 100 步线性 beta（`1e-4 -> 0.02`）。末步
   `alpha_bar_T = 0.36356`，即 `sqrt(alpha_bar_T) = 0.60296`。
   训练末端输入仍含约 60.3% 的干净残差信号，但推理却从纯高斯噪声开始，训练/推理起点不匹配。
2. denoiser 只有单一 1/4 分辨率特征，主体是 64 通道的 depthwise 3D 卷积；输出通过双线性插值恢复到 128，没有高分辨率 skip，也没有逐级重建。它很难恢复连续边界和小尺度强回波。
3. 20 帧被拆成 4 个 5 帧片段递推，前一片段的随机误差成为后一片段的条件，后期误差和纹理噪声会累积。

结论：停止 TrajRes-DF 后续正式训练；保留结果作为失败分析，不在此结构上继续打补丁。

## 3. 文献证据后的结构选择

### 3.1 第一来源：CasFormer（CasCast，ICML 2024）

CasCast 已经在与本项目相同的 SEVIR 阈值和 pool1/4/16 指标上证明：

- EarthFormer 到 CasCast：CSI 0.4310 -> 0.4401；
- CSI_pool4 0.4319 -> 0.4640；
- CSI_pool16 0.4351 -> 0.5225；
- HSS 0.5411 -> 0.5602；
- SSIM 0.7756 -> 0.7797。

它是目前最符合需求的第一迁移来源，因为论文已经证明同一个概率模块能让五项指标同时提高，而不是只改善 CRPS。

CasFormer 的关键不是“换成普通 Transformer”，而是：

1. 每个 future lead 的 noisy target 与同一 lead 的确定性预测一一对应；
2. 先做 frame-wise DiT 编码，降低不同 lead 混在一起造成的优化冲突；
3. 再把所有 future lead 聚合后做 sequence-wise DiT，联合建模整个未来序列；
4. 使用 adaLN-Zero 注入扩散时间。

论文消融显示 1-frame split 优于 6-frame split 和普通 sequence-wise DiT。这正好反驳当前 4×5 片段递推方案。

### 3.2 第二来源：FREUD / LSM（CVPR 2026）

FREUD 的关键价值不是取代残差故事，而是解决可视化质量：

- frame-wise encoder 保持每帧条件的因果和对应关系；
- united video decoder 对所有帧联合解码；
- hierarchical 3D token merge/split、空间/时间 factorized attention 和 U-shaped skip 共同保留局部结构；
- 论文中 united decoder 相比 frame-wise decoder 将 temporal dMAE 降低约 33%，并显著减少 flicker；
- 其大模型在 SEVIR 上达到 CRPS 0.0190、SSIM 0.7841；CFG 版本 SSIM 0.7937。

FREUD 单独使用时 localization 指标并不总是超过 CasCast，因此不把它直接作为第一模型。它作为 CasFormer 的联合时空解码头，目标明确：修复颗粒、闪烁和结构不连续。

### 3.3 暂不优先的来源

- DFoT（ICML 2025）的 per-frame noise 和 3D full attention适合第三阶段解决长时一致性，但没有 CasCast 那样直接的 SEVIR 五指标证据。
- STCDiT（CVPR 2026）的 anchor feature modulation 适合在确定性结构被概率分支破坏时加入，但优先级低于直接面向降水的 FREUD united decoder。
- FREUD 的完整 latent flow 作为 Plan B；它的 CRPS/SSIM 很强，但无确定性先验时 CSI/HSS 弱于 CasCast，不应成为第一枪。

## 4. 最终主模型：Residual CasFormer with United Temporal Decoder

工作名：`ResCasFormer-UTD`。

### 4.1 总体路径

```text
history x
   |
   +--> frozen PhyDNet / SDIR --------------------> deterministic mean μ
                                                       |
target y ----------------------------------------------+--> residual r0 = y - μ
                                                                      |
Gaussian noise ε + diffusion timestep t ------------------------------+
                                                                      v
              [noisy residual r_t,j ; matching deterministic μ_j]
                                  |
                         frame-wise patch embed
                                  |
                   CasFormer frame-wise DiT blocks
                    （20 帧共享权重、逐帧对应）
                                  |
                sequence aggregation over all 20 leads
                                  |
                    sequence-wise DiT bottleneck
                                  |
              FREUD united spatiotemporal decoder
        （3D token merge/split + spatial/temporal attention + U skips）
                                  |
                     predicted residual/noise
                                  |
                          μ + predicted r0
                                  |
                            final forecast
```

### 4.2 保留 DiffCast 故事的部分

- 确定性分支负责可预测的大尺度运动和回波主体；
- 概率分支只学习 `r = y - μ`，修复确定性预测遗漏的小尺度、强回波和边界；
- 最终输出仍然是 `y_hat = μ + r_hat`；
- 概率分支先在 PhyDNet 上快速验证，再原样迁移到 SDIR；主论文使用 SDIR 做最终主结果，PhyDNet 结果用于结构筛选和跨 backbone 证据。

### 4.3 从 CasFormer 原样迁移的部分

- 20 个 future leads 不再当普通 2D channels；
- 对每个 lead 独立执行 patch embedding 和 frame-wise DiT；
- noisy residual frame 与同 lead 的 `μ_j` 直接拼接；
- 使用 adaLN-Zero；
- frame-wise 特征在相同空间位置上聚合，然后进入 sequence-wise DiT；
- 一次性联合预测 20 帧，不再 4×5 自回归。

第一版建议按论文容量起步，而不是缩成小网络：

- patch size 4；
- frame width 256，frame depth 12；
- sequence width 1152，sequence depth 12；
- 16 heads，MLP ratio 4；
- bf16、gradient checkpointing；
- 8×H800 全量训练。

### 4.4 从 FREUD 原样迁移的部分

CasFormer 原始 head 只是线性 projection + unpatchify。第二阶段把它替换为 FREUD 风格 united decoder：

- 对 `[T,H,W]` 联合解码；
- factorized spatial attention + temporal attention；
- 多尺度 3D token merge/split；
- U-shaped skip；
- `μ` 的逐帧特征作为 conditional tokens 在中间尺度注入；
- 全部 20 帧共享一次联合解码，禁止 frame-wise 独立输出。

这部分的 motivation 很直接：CasFormer 解决“每个未来时刻应该听哪一帧确定性条件”，FREUD 解决“所有未来帧如何联合恢复成连续、无闪烁的高分辨率序列”。

### 4.5 为什么第一版使用 pixel-space residual

CasCast 自己的消融指出低维 autoencoder 会损伤 HSS/CSI，极端阈值尤其敏感。当前输入只有 128×128，且有 8×H800，没有必要为了显存引入 VAE 重建误差。

因此第一版：

- 对标准化后的 signed residual 直接扩散；
- 保留 128×128 输出；
- 用 patch token 降低计算量，但最终由多尺度 united decoder 恢复像素；
- 不使用 VAE，不让 compression error 污染 CSI-181/219 和 SSIM。

## 5. 训练与推理必须一致

### 5.1 第一阶段 P1：PhyDNet + 纯 CasFormer 替换

目的：只验证 CasFormer 的 frame-wise guidance + sequence aggregation 是否能让五项指标同时提升。

- backbone：冻结的 PhyDNet；
- target：标准化 residual；
- denoiser：CasFormer；
- head：CasFormer 原始 joint projection/unpatchify；
- 1000-step cosine schedule，确保末端接近纯噪声；
- v-prediction，复用当前已经验证过数值稳定性的 residual diffusion parameterization；
- DDIM 20-step sampling；
- 一次性 20 帧；
- K=10 用于概率评测和 ensemble mean；
- 同时保存 K=1 典型样本，避免 ensemble mean 掩盖颗粒。

P1 是结构迁移实验，不加入自创 loss、gate、trajectory adapter 或 segment recursion。PhyDNet 训练和推理快，因此允许完整训练和全量 `report_test`，而不是只用少量 validation 猜测结构是否有效。

### 5.2 第二阶段 P2：在 PhyDNet 上加入 FREUD united decoder

只有 P1 的数值或可视化至少有一项明确接近通过时才启动。

- 从 P1 checkpoint 初始化 frame-wise encoder 和 sequence bottleneck；
- 用 FREUD hierarchical united decoder 替换线性 head；
- `μ` feature 作为 decoder conditional tokens；
- 仍然使用相同 residual target、schedule 和 sampler；
- 先冻结 P1 主体训练 decoder，再全模型联合微调。

P2 预期主要改善 SSIM、边界连续性、late-lead temporal consistency，并压制颗粒。

### 5.3 第三阶段 P3：PhyDNet 联合微调

概率模块通过后，才进行短程联合微调：

- PhyDNet learning rate 为概率分支的 1/20 到 1/50；
- checkpoint 不是按总 loss 选，而是按五项指标硬约束选；
- 若任一核心指标在 `val_model` 低于冻结版本，立即退回冻结 backbone。

联合训练不是第一步，因为需要先证明概率模块本身有效并保持可归因性。

### 5.4 第四阶段 S1/S2：迁移到 SDIR

只有 PhyDNet 上已经通过五项指标和视觉门槛的最佳结构才进入 SDIR：

- S1：在配置中把 deterministic backbone 从 PhyDNet 切换为 frozen SDIR；
- 整个概率模块保持为同一个实现：不改 denoiser 结构、接口、扩散 schedule、loss、sampler 和默认超参数，不新增 SDIR 专用分支；
- 代码层面应当只需要修改配置中的 backbone 名称、deterministic checkpoint 和 residual statistics 路径；
- 因为 residual target 定义为 `r = y - μ`，PhyDNet 和 SDIR 产生的 `μ` 不同，所以需要重新统计 SDIR residual 的 center/scale，并用 SDIR residual 重新训练同一个概率模块；这属于换训练数据分布，不属于修改模型结构；
- 可以额外测试一次直接加载 PhyDNet 概率权重的 zero-shot backbone swap，作为通用性诊断，但不把它当作主实验；
- S1 通过后再做 S2：以概率分支 1/20 到 1/50 的学习率联合微调 SDIR；
- PhyDNet 上有效不代表在更强的 SDIR 上必然有效，因此 SDIR 仍需独立通过五项硬门槛。

## 6. 硬性通过/淘汰协议

### 6.1 数值门槛

快速验证阶段的 PhyDNet 基线：

- CSI > 0.269791
- CSI_pool4 > 0.277409
- CSI_pool16 > 0.297413
- HSS > 0.347121
- SSIM > 0.700970

迁移后的 SDIR 主结果基线：

- CSI > 0.304496
- CSI_pool4 > 0.304929
- CSI_pool16 > 0.319770
- HSS > 0.396627
- SSIM > 0.748057

正式论文结果不能只以小数点后噪声为“提升”。最终采用 paired event bootstrap：

- 对同一批 5600 个 test samples 重采样；
- 报告五项指标的 paired delta 和 95% CI；
- 五项 delta 的 95% CI 下界都大于 0 才算严格通过。

进入 PhyDNet 全量 `report_test` 前，`val_model` 的建议安全目标为：

- CSI 至少 +0.005；
- CSI_pool4 至少 +0.010；
- CSI_pool16 至少 +0.015；
- HSS 至少 +0.005；
- SSIM 至少 +0.005。

达不到则不消耗一次正式 report_test。

### 6.2 可视化门槛

固定一组来自 `val_calib` 的 16 个案例，覆盖：

- 普通降水；
- 强对流；
- 快速平移；
- 生消过程；
- 高阈值 181/219；
- 80–100 min late leads。

每个案例必须在同一张图展示：

1. Input；
2. Ground Truth；
3. 匹配的 deterministic backbone（PhyDNet 筛选阶段或 SDIR 主结果阶段）；
4. 单个概率样本 K=1；
5. K=10 ensemble mean；
6. `r_hat` residual；
7. absolute error。

定量辅助检查：

- non-rain 区域的 false speckle rate；
- 连通域数量/湿区面积，防止碎片化；
- Laplacian high-frequency energy 与 GT 的差；
- 相邻帧 temporal dMAE；
- 80–100 min 的单独 SSIM/CSI。

若指标提高但固定图仍出现明显颗粒或闪烁，同样判失败。

## 7. 实验顺序

| 顺序 | 实验 | 回答的问题 | 是否全量 report_test |
|---|---|---|---|
| E0 | 评测/registry 补齐 pool4、pool16，并验证 schedule 端点 | 尺子和扩散过程是否正确 | 否 |
| P1 | PhyDNet + pixel residual CasFormer | 顶会已验证的 frame-wise guided DiT 能否替换 GTUNet | 过 val 门槛后，在 PhyDNet 上全量测试 |
| P2 | P1 + FREUD united decoder | 联合时空解码能否进一步提高 SSIM、抑制颗粒 | 过 val 门槛后，在 PhyDNet 上全量测试 |
| P3 | P2 + 低学习率联合微调 PhyDNet | 联合优化是否让五项继续提高 | 仅优于冻结版后 |
| S1 | 最佳 PhyDNet 结构原样迁移到 frozen SDIR | 对更强 backbone 是否仍然有效 | 过 SDIR val 门槛后 |
| S2 | S1 + 低学习率联合微调 SDIR | 能否形成最终主结果 | 仅优于 S1 后 |

不再同时铺开五六个未经验证的小模块。每个正式训练只回答一个问题。

## 8. Plan B

如果 P1 在 PhyDNet 上明确失败，不在 CasFormer 上继续堆小模块，直接在 PhyDNet 上转为 CVPR 2026 FREUD/LSM 路线：

- 使用其 clean integration implementation；
- 先保留 PhyDNet 作为 deterministic prior；通过后再替换为 SDIR；
- 将 full-future target 改为 residual field；
- 采用其 masking-based rectified-flow training 和 factorized spatiotemporal SiT；
- 使用 united probabilistic decoder；
- 在 validation 上选择 deterministic-prior strength。

FREUD 论文中 deterministic prior 已把 CSI 从 0.3864 提到 0.4455、HSS 从 0.5011 提到 0.5735，证明概率预测与确定性先验的结合方向有效；但其 pooled CSI 和该设置的 SSIM 未完整报告，所以仍必须走本项目的五指标硬门槛。

## 9. 论文故事

一句话版本：

> Deterministic nowcasters reliably capture predictable mesoscale motion but blur stochastic small-scale precipitation. We model only the residual uncertainty with frame-aligned diffusion encoding and reconstruct the entire future through a united spatiotemporal decoder, sharpening local extremes without sacrificing deterministic localization.

对应三层 motivation：

1. **为什么预测 residual**：确定性 backbone 已经解释大尺度可预测成分，概率模型不应重复生成整幅图。
2. **为什么 frame-wise guided encoder**：每个 noisy residual lead 必须与同一 lead 的 deterministic forecast 对齐，避免 20 帧条件混淆。
3. **为什么 united decoder**：局部修正必须作为一个连续未来序列联合重建，不能逐帧或逐片段独立生成，否则会产生颗粒和闪烁。

## 10. 主要参考

- CasCast, ICML 2024: https://proceedings.mlr.press/v235/gong24a.html
- FREUD / Probabilistic Precipitation Nowcasting with Rectified Flow Transformers, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Schusterbauer_Probabilistic_Precipitation_Nowcasting_with_Rectified_Flow_Transformers_CVPR_2026_paper.html
- History-Guided Video Diffusion / Diffusion Forcing Transformer, ICML 2025: https://proceedings.mlr.press/v267/song25b.html
- STCDiT, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html

## 11. P1 代码实现拆分

今天只实现和运行 P1：`PhyDNet + Residual CasFormer`。P2 的 FREUD united decoder 不与 P1 同时实现，避免第一轮实验无法判断增益来源。

### Step 1：新增独立概率模块目录

新增：

```text
src/phyrd/models/probabilistic/rescasformer/
  __init__.py
  blocks.py
  denoiser.py
  model.py
```

模块名固定为 `rescasformer`，实现现有 `ProbabilisticModel` 接口：

```python
training_loss(history, target, trend)
sample(history, trend, ensemble_size, sampling_steps)
```

`ForecastComposer`、deterministic backbone 接口和训练器不应为 PhyDNet/SDIR 分别修改。若实现正确，后续换 backbone 只改 YAML。

### Step 2：实现 CasFormer 基础块

在 `blocks.py` 中实现：

- `TimestepEmbedder`
- `PatchEmbed2D`
- `AdaLNZeroDiTBlock`
- 2D sin-cos positional embedding
- `FinalLayer`
- `patchify/unpatchify`

要求：

- adaLN-Zero 和输出层 zero initialization；
- 支持 bf16；
- 不依赖 CasCast 仓库运行；
- 根据论文结构 clean-room 重写并在文件头注明架构来源；
- 不引入 `timm` 等服务器上未确认存在的新依赖。

### Step 3：实现 Residual CasFormer denoiser

在 `denoiser.py` 中实现 `ResidualCasFormerDenoiser`：

输入：

```text
noisy_residual: [B, 20, 1, 128, 128]
trend:          [B, 20, 1, 128, 128]
timestep:       [B]
history:        [B, 5, 1, 128, 128]  # 保留公共接口，P1 不额外编码
```

数据路径：

1. 将 trend 从 `[0,1]` 映射到 `[-1,1]`；
2. 每个 lead 拼接 `[noisy_residual_j, trend_j]`；
3. 20 帧共享同一个 patch embedding；
4. 每帧独立经过共享的 frame-wise DiT blocks；
5. 对同一空间 patch 的 20 帧特征执行 CasFormer sequence aggregation；
6. 经过 sequence-wise DiT blocks；
7. joint projection 一次输出全部 20 帧；
8. unpatchify 为 `[B,20,1,H,W]`。

P1 使用发布版 CasFormer 的核心拓扑，不加入 segment recursion、trajectory adapter、额外 gate 或自创 decoder。

正式配置：

```text
patch_size = 4
frame_hidden = 256
frame_depth = 12
frame_heads = 4
sequence_hidden = 1152
sequence_depth = 12
sequence_heads = 16
mlp_ratio = 4
output_frames = 20
```

测试配置允许把宽度、深度和分辨率缩小，但正式训练不缩模型。

### Step 4：接入稳定 residual diffusion

在 `model.py` 中实现 `ResidualCasFormerModel`：

- `clean_residual = target - trend`；
- 读取现有 PhyDNet train-split residual statistics；
- 每个 lead 独立 center/scale normalization；
- 复用当前 `GaussianResidualDiffusion` 的 cosine schedule、v-prediction、x0 stabilization 和 DDIM；
- 正式训练 `diffusion_steps=1000`；
- 正式推理 `sampling_steps=20`；
- 输出始终为 `(trend + residual).clamp(0,1)`；
- 支持 K=1 和 K=10，不修改 evaluator。

这一步只替换 denoiser，不同时更换经过验证的扩散数学实现，以便把结果变化归因到 CasFormer。

这里的 `diffusion_steps=1000` 是前向加噪过程的离散时间刻度，不是 optimizer 更新次数。正式训练量单独设为 `optimization.max_steps=200000`；每次 optimizer step 随机抽取一个 diffusion timestep，因此训练过程中会反复覆盖 1000 个噪声等级。

### Step 5：注册与配置化

修改：

```text
src/phyrd/models/probabilistic/__init__.py
```

注册：

```text
rescasformer -> phyrd.models.probabilistic.rescasformer:ResidualCasFormerModel
```

新增正式配置：

```text
configs/active/5to20/train_ddp8_phydnet_rescasformer_5to20_v14_seed42.yaml
```

关键配置：

```text
stage: residual
freeze_deterministic: true
deterministic.name: phydnet_external
probabilistic.name: rescasformer
diffusion_steps: 1000
sampling_steps: 20
ensemble_size: 10
max_steps: 200000
precision: bf16
post_training_test.split: report_test
post_training_test.register_result: true
```

正式目标 effective global batch size 为 32，与 CasCast 的 200k-step 训练量对齐。首个显存探测从每卡 batch size 2 开始；若每卡 batch size 4 能运行，则 8 卡直接得到 global batch 32；若只能每卡 batch size 2，则在训练器中加入 gradient accumulation 2，仍保持 effective global batch 32。模型结构不因显存探测而缩小。

### Step 6：单元测试

新增：

```text
tests/test_rescasformer.py
```

必须覆盖：

1. frame-wise 与 sequence-wise tensor shape；
2. patchify/unpatchify round trip；
3. `training_loss` 有限且可 backward；
4. K=2 sampling 输出形状正确；
5. residual normalization/denormalization round trip；
6. 1000-step schedule 末端接近纯噪声；
7. checkpoint save/load round trip；
8. 同一概率模块分别接 tiny PhyDNet-like 与 SDIR-like backbone，无需修改概率模块代码；
9. gradient accumulation 与直接 global batch 的 loss scaling 一致。

### Step 7：服务器 smoke test

代码同步到选定服务器后依次执行：

1. 单卡 import + unit tests；
2. 单卡 16/32 samples 前向、反向和采样；
3. 单卡小样本 overfit，确认 loss 明显下降且生成结果不是纯噪声；
4. 8 卡 DDP smoke，确认每卡数据、梯度同步、checkpoint、resume 和日志正常；
5. 生成一张 Input/GT/PhyDNet/P1 的小样本可视化。

smoke test 不登记到 `RESULTS_REGISTRY.csv`。

### Step 8：八卡正式训练与全量测试

smoke 全部通过后：

- 创建/复用 tmux `wzq`；
- 8 卡启动 P1，训练 200,000 optimizer steps，effective global batch 32；
- validation 每 5 epochs；
- 保留 best checkpoint 和完整 train log；
- 训练完成后自动执行 K=10、5600 samples 的 `report_test`；
- 把 CSI、CSI_pool4、CSI_pool16、HSS、SSIM、CRPS 写入结果表；
- 生成固定案例的 K=1 与 K=10 mean 对比图。

只有 P1 同时通过五项 PhyDNet 基线和可视化门槛，下一次实验才实现 P2 的 FREUD united decoder。
