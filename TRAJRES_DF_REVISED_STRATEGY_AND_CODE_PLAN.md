# TrajRes-DF 失败复盘、调整后实验计划与代码修改计划

更新时间：2026-07-24  
状态：方案评审稿；尚未实施、尚未启动新一轮正式训练

## 1. 当前结论

不继续执行原计划里的“J-Final = 在 J-Core 上只加一个高频门控”。

R-015 已经证明两件事：

1. 残差概率建模有价值：CRPS 从纯 PhyDNet 的 7.824252 降到了
   6.707285。
2. 当前 TrajRes-DF 的生成质量不合格：CSI、HSS、MAE、MSE、SSIM
   全部退化，而且可视化中出现明显颗粒和散斑。

| 指标 | 纯 PhyDNet R-006 | TrajRes-DF R-015 | 变化 |
|---|---:|---:|---:|
| CRPS | 7.824252 | 6.707285 | 改善 |
| CSI | 0.269791 | 0.254858 | 退化 |
| HSS | 0.347121 | 0.326172 | 退化 |
| MAE | 0.030683 | 0.034648 | 退化 |
| MSE | 0.004413 | 0.004812 | 退化 |
| SSIM | 0.700970 | 0.524332 | 严重退化 |

因此下一步不是给旧模型继续堆模块，而是先把失败拆开，再只跑一组经过
小验证的正式修复实验。

## 2. 当前失败的根因排序

### 2.1 第一优先级：扩散训练终点与推理初始分布不匹配

当前 TrajRes 使用 100 步线性 beta：

```text
beta = linspace(1e-4, 0.02, 100)
alpha_bar[T] = 0.363563
sqrt(alpha_bar[T]) = 0.602962
```

也就是说，训练中的最后一个噪声状态仍包含约 60.3% 的干净残差信号；
推理却直接从纯高斯噪声开始。这个终点并没有充分扩散到标准高斯，采样初态
与训练状态存在明显分布差异。

调整：改为仓库中已经验证过的 cosine schedule，使终点接近纯噪声，并复用
稳定的 v-prediction、`x0` clipping 与 DDIM 实现。

### 2.2 第一优先级：去噪器不是完整 U-Net

当前 `TemporalResidualDenoiser`：

- 连续两次 stride=2，下采样到 1/4；
- 只在 1/4 分辨率做两层 depthwise 3D convolution；
- 直接双线性插值回 128×128；
- 没有 encoder-decoder 多尺度结构；
- 没有任何 spatial skip connection。

它可以学习低频位移和大体形状，却没有可靠路径重建边缘与局部纹理。双线性
放大后的粗特征再叠加随机残差，很容易表现为现在看到的颗粒和散斑。

调整：换成真正的 factorized spatiotemporal U-Net，保留 128、64、32
三个尺度的空间 skip，只在低分辨率特征上做时间混合与注意力。

### 2.3 第一优先级：4×5 分段递推会累计误差

当前推理流程每 5 帧重新从高斯噪声采样，并把上一段生成残差作为下一段
prefix。任何散斑和幅度偏差都会被后续段作为条件继续传播。训练时的
clean/noisy/zero prefix 仍然不能完全复现模型自己的错误分布。

调整：正式修复版改为“一次生成完整 20 帧，但显式保留时间维”
`[B,C,T,H,W]`。这不是旧版把 20 帧压成 20 个无序 2D channel，而是在
3D/factorized temporal blocks 中显式建模 20 帧演化，同时消除 segment
递推误差。

### 2.4 第一优先级：联合训练破坏了热插拔前提和残差标定

残差 center/scale 是用原始冻结 PhyDNet 预先统计的；J-Core 在前 5 epoch
后又用 195 epoch 更新 PhyDNet。此时：

- 残差目标 `y - mu` 持续变化；
- 固定 center/scale 不再对应当前 `mu`；
- 概率模块一边追逐移动目标，一边还要适配固定归一化；
- 最终模型也不再是真正可挂到原始 PhyDNet 上的热插拔模块。

调整：下一组主实验完全冻结 PhyDNet。这样残差统计稳定，结果能够严格回答
“同一个确定性预测器加上概率插件是否变好”。联合微调只在该版本成功后作为
后续可选消融，不进入当前正式队列。

### 2.5 第二优先级：训练目标与最终图像质量脱节

当前仅优化随机 segment 上的 v-MSE，best checkpoint 也按最多 32 个
validation batch 的 denoising loss 选择。它不会直接惩罚：

- 最终 forecast 的颗粒；
- 时间不连续；
- CSI/SSIM 相对确定性基线的下降；
- 采样后的残差幅度过大。

调整：

- 保留 v-MSE 为主目标；
- 加小权重的 `x0` residual Huber、多尺度重建和时间差分损失；
- 用固定 val 子集的真实采样指标选 checkpoint，而不是只看 v-MSE。

## 3. 先做的小验证：不占用一天一组的正式实验名额

所有调参都只使用 `val_model`，不碰 `report_test`。`report_test` 仍然只在
正式训练结束后运行一次。

### V0：旧 checkpoint 的三段式归因

不重新训练，读取 R-015 checkpoint，在固定的 512 个 `val_model` 样本上
分别评估：

1. 原始纯 PhyDNet：`mu_base`；
2. R-015 checkpoint 内联合更新后的 PhyDNet：`mu_joint`；
3. `mu_joint + residual`。

同时按 lead 和四个 5-frame 区间输出 CSI、SSIM、MAE、残差标准差、
Laplacian 高频能量和时间差分误差。

回答的问题：

- 如果 `mu_joint` 已经明显差于 `mu_base`，说明联合训练本身伤害了 backbone；
- 如果第 2、3、4 段逐段恶化，说明 segment/prefix 累计是主因；
- 如果 K=10 mean 仍有高频能量异常，说明不是单个样本随机，而是生成分布
  存在系统性散斑。

预计耗时：单机推理约 1–2 小时。

### V1：旧 checkpoint 的残差标定扫描

对同一批样本复用生成 residual，不重跑网络训练：

```text
y_hat(alpha) = mu_joint + alpha * residual
alpha in {0.00, 0.125, 0.25, 0.50, 0.75, 1.00}
K in {1, 4, 10}
```

额外测试 residual-only Gaussian blur：

```text
sigma in {0.0, 0.5, 1.0}
```

这不是最终方法，只用于判断：

- 缩小 residual 后恢复：主要是幅度标定问题；
- 轻微平滑后恢复：主要是高频采样问题；
- 二者都不能恢复：去噪器和分段生成是结构性失败。

输出一个 CSV、一个 JSON 和固定 8 个 case 的对比图。最优 alpha 只用于
初始化新模型的 gate bias，不作为 test 调参结果。

预计耗时：如果缓存 residual，额外约 20–40 分钟。

### V2：新模型过拟合与采样一致性门

代码完成后先做两级技术验证：

1. 20-step smoke：检查 DDP、显存、loss、EMA、checkpoint 和 sample shape；
2. 小数据过拟合：固定 64 个 train 样本、16 个 validation 样本，验证
   clean reconstruction 能下降，DDIM sample 不产生系统性散斑。

通过条件：

- cosine schedule 终点 `sqrt(alpha_bar[T]) < 0.02`；
- `alpha=0` 严格复现 deterministic trend；
- gate 始终位于 `[0,1]`；
- 训练后的小样本 `x0` 重建和时间差分误差显著下降；
- K=1 和 K=4 mean 的高频能量不出现数量级异常；
- 断点恢复与旧 checkpoint 加载测试通过。

预计耗时：30 分钟 smoke，加 1–3 小时小数据过拟合。

只要 V2 不通过，就不占用 7/8 卡跑 200 epoch。

## 4. 唯一保留的下一组正式实验

实验名：`J-Repair: Frozen PhyDNet + Gated ST-Residual Diffusion`

### 4.1 模型

```text
history x ───────────────┐
                        ├─ multi-scale spatial condition
frozen PhyDNet -> mu ────┤
                        v
noisy residual [T=20] -> factorized ST-U-Net -> raw residual
                                                   |
                                    bounded calibration gate
                                                   |
                                     y_hat = mu + gated residual
```

具体约束：

- PhyDNet 全程冻结；
- 一次生成完整 20 帧；
- 20 帧是显式时间维，不是普通 2D channels；
- 空间尺度为 128 → 64 → 32 → 64 → 128；
- full-resolution 和 half-resolution skip 保留边缘；
- 时间卷积/注意力主要放在 32×32 bottleneck，控制显存；
- history 与完整 deterministic trajectory 以多尺度空间 feature 注入，
  不再只用 global-average token；
- cosine diffusion + v-prediction + dynamic `x0` clipping；
- DDIM 20-step 正式推理；
- EMA 权重用于 validation 和 test；
- learned gate 对 residual 做有界校准，不能任意放大 correction。

### 4.2 损失

第一版不加入“高频增强损失”，避免继续放大颗粒。采用：

```text
L = L_v
  + lambda_x0   * Huber(r_x0, r_gt)
  + lambda_pyr  * multi_scale_L1(mu + r_x0, y)
  + lambda_temp * L1(delta_t(mu + r_x0), delta_t(y))
  + lambda_gate * gate_regularization
```

初始建议：

```text
lambda_x0   = 0.10
lambda_pyr  = 0.05
lambda_temp = 0.05
lambda_gate = 0.01
```

小验证只能在窄范围内检查稳定性，不能用 `report_test` 搜权重。

### 4.3 训练和 checkpoint 选择

- max epochs：200；
- backbone：冻结；
- 概率模块初始 LR：`2e-4`；
- precision：bf16；
- 每卡 batch size 先以 4 做显存 smoke；峰值显存低于 60 GiB 时再升到 6/8；
- v-loss validation：每 5 epoch；
- 固定 256 个 `val_model` 样本，K=4、DDIM 10-step 采样验证：每 10 epoch；
- best checkpoint 先满足“CSI/SSIM 不低于 val deterministic 容差”，再按
  val CRPS 选择；
- early stopping patience：40 epoch，但最多仍允许跑满 200；
- 训练完成后自动在完整 5600 个 `report_test` 样本上以 K=10、DDIM 20
  评估，并写入 `RESULTS_REGISTRY.csv`。

### 4.4 正式成功标准

最低标准是相对同协议纯 PhyDNet R-006：

| 指标 | 必须满足 |
|---|---:|
| CRPS | < 7.824252 |
| CSI | >= 0.269791 |
| HSS | >= 0.347121 |
| MAE | <= 0.030683 |
| MSE | <= 0.004413 |
| SSIM | >= 0.700970 |

另外设置强目标，但不作为最低通过线：

- CRPS 接近或低于 SDIR residual diffusion R-004 的 5.984013；
- CSI 接近或高于 0.294；
- SSIM 高于 0.702；
- 固定可视化 case 中 K=1 与 ensemble mean 均无系统性颗粒；
- 后 10 帧不能出现明显比前 10 帧更严重的误差爆炸。

如果 J-Repair 仍不能同时守住 CSI/SSIM，就停止 PhyDNet 概率插件这条结构
继续堆叠，不再跑第三个门控变体；转回效果已经更稳定的
`SDIR + residual_diffusion_vpred` 主线。

## 5. 代码修改计划

原则：新增模块，不覆盖 `trajres_diffusion`。R-015 和之前所有 checkpoint
继续由旧类加载，新代码使用新的模型名和配置。

### 5.1 新概率模块

新增目录：

```text
src/phyrd/models/probabilistic/trajres_unet_gated/
    __init__.py
    blocks.py
    conditioning.py
    denoiser.py
    diffusion.py
    model.py
```

职责：

- `blocks.py`
  - 2D spatial ResBlock；
  - factorized `(1,3,3)` spatial conv + `(3,1,1)` temporal conv；
  - spatial-only down/up sample；
  - bottleneck temporal attention；
  - time embedding/FiLM。
- `conditioning.py`
  - 编码 5 帧 history 和 20 帧 trend；
  - 生成 128、64、32 三尺度空间条件；
  - 按 lead 注入，不做全局平均后丢掉空间位置。
- `denoiser.py`
  - 输入 `[B,C,T,H,W]`；
  - 完整 encoder/decoder 与 skip；
  - 输出 20 帧 v-prediction；
  - 同时输出 calibration gate logits。
- `diffusion.py`
  - 复用现有 cosine schedule、v-prediction、DDIM 与 dynamic clipping；
  - 增加 min-SNR weighting；
  - 计算 `x0`、pyramid、temporal 和 gate 辅助损失；
  - 保存 raw/gated residual 诊断量。
- `model.py`
  - 实现统一 `ProbabilisticModel` 接口；
  - 训练时返回总 loss 和分项；
  - 推理时返回 `trend + gated_residual`；
  - `alpha=0` 的诊断路径严格返回 trend；
  - 保持 `checkpoint_as_diffusion=True` 的兼容约定。

注册修改：

```text
src/phyrd/models/probabilistic/__init__.py
```

新增名字 `trajres_unet_gated`，不改变旧名字 `trajres_diffusion`。

### 5.2 诊断脚本

新增：

```text
scripts/evaluation/diagnose_trajres_failure.py
```

功能：

- 同时加载 pure PhyDNet 与 R-015 joint checkpoint；
- 输出 `mu_base`、`mu_joint`、raw members、ensemble mean；
- 扫 alpha、K 和 residual blur；
- 输出按 lead/segment 指标；
- 输出高频能量、时间差分误差；
- 缓存 residual，避免每个 alpha 重复扩散采样；
- 只允许 `val_model`，默认拒绝 `report_test`，防止测试集调参。

输出目录：

```text
artifacts/diagnostics/trajres_r015/
    decomposition.json
    alpha_k_sweep.csv
    per_lead_metrics.csv
    cached_residuals/
    figures/
```

### 5.3 训练器

修改：

```text
scripts/train.py
```

计划改动：

- `stage: residual` 下支持记录所有 `loss_*` 分项；
- 增加 EMA 创建、每 step 更新、保存、恢复和验证加载；
- 增加固定样本 sampled validation；
- best checkpoint 支持“结构指标约束 + CRPS”规则；
- 日志增加：
  - `loss_v`；
  - `loss_x0`；
  - `loss_pyramid`；
  - `loss_temporal`；
  - `gate_mean/std`；
  - `raw_residual_std`；
  - `gated_residual_std`；
  - `sampled_val_CRPS/CSI/SSIM`；
- smoke 的 `--max-steps` 继续禁止 full test 和结果登记。

不新增训练 stage。正式配置直接使用现有 `stage: residual` 和
`freeze_deterministic: true`，避免修改已有 `joint_residual` 行为。

### 5.4 checkpoint 与评测

修改：

```text
scripts/evaluation/post_training.py
scripts/evaluation/evaluate_residual_diffcast.py
```

计划改动：

- checkpoint 有 EMA 时默认评估 EMA；没有时保持旧行为；
- 指标 JSON 记录 `weights_used: ema|online`；
- 正式表格指标不改字段，确保 `RESULTS_REGISTRY.csv` 向后兼容；
- 额外诊断指标写入独立的 `diagnostics` 字段；
- 自动生成 pure baseline、K=1、K=10 mean、GT 的固定 case 面板。

### 5.5 新配置

新增：

```text
configs/diagnostics/trajres_r015_failure_val.yaml
configs/diagnostics/trajres_unet_gated_smoke.yaml
configs/diagnostics/trajres_unet_gated_overfit.yaml
configs/active/5to20/train_ddp_phydnet_trajres_unet_gated_5to20_v14_seed42.yaml
```

正式配置必须明确：

```yaml
stage: residual
model:
  freeze_deterministic: true
  probabilistic:
    name: trajres_unet_gated
```

### 5.6 测试

新增：

```text
tests/test_trajres_unet_gated.py
tests/test_trajres_diagnostics.py
```

覆盖：

- 20 帧显式时间维 shape；
- encoder/decoder skip 真正参与梯度；
- 空间条件变化会改变对应区域输出；
- gate 范围和 `alpha=0` 恒等性；
- cosine terminal noise gate；
- `x0` clipping；
- DDIM sample 有限值；
- EMA save/resume；
- pure/joint checkpoint 分解；
- 新旧 TrajRes checkpoint 互不冲突。

## 6. 执行顺序

```text
V0/V1：旧 checkpoint 推理归因
        |
        v
实现新模块 + 单元测试
        |
        v
V2：20-step smoke + 64样本过拟合
        |
        +-- 不通过：修代码，不启动正式训练
        |
        v
J-Repair：唯一一组 200 epoch 正式训练
        |
        v
完整 report_test(K=10) + 结果登记 + 固定 case 可视化
```

这版调整后的核心故事仍然是“预测残差比直接生成未来图像更容易”，但工程上
改成更严格的表述：

> 冻结任意确定性 backbone，把其误差作为稳定目标；使用显式时空 U-Net
> 建模整段 residual evolution，再通过有界校准只在确定性预测需要修正的
> 区域注入概率残差。

它同时保住了热插拔性，也正面针对 R-015 暴露的散斑、递推累计、归一化漂移
和采样终点失配。
