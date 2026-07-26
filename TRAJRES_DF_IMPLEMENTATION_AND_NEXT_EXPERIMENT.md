# TrajRes-DF：PhyDNet 上的时序残差扩散设计与下一步实验

更新时间：2026-07-23  
状态：J-Core 代码与正式配置已完成；已在 weather-30240 排除异常 GPU 4，
使用 GPU 0,1,2,3,5,6,7 启动校准 → 200 epoch 正式训练流水线。

## 1. 一句话目标

把 PhyDNet 当作确定性趋势预测器 \(\mu=f_\theta(x)\)，扩散模型不直接生成未来雷达图，而是生成

\[
r = y-\mu,\qquad \hat y=\mu+\hat r .
\]

论文故事是：**未来图像的多模态生成很难，而条件在一个较强确定性预报上的剩余误差更小、更集中；概率模块只需学习“确定性预报哪里应被锐化、削弱或位移”。**

该概率模块保持热插拔接口，后续既能挂到 PhyDNet，也能挂到 SDIR；J-Core 首先在 PhyDNet 上验证，因为它可端到端训练，且已有正式 `report_test` 基线。

## 2. 为什么不是原来的 20-channel U-Net

旧残差扩散把 20 个 future leads 当成 20 个普通 2D channels 一次去噪。这样虽然实现简单，但网络没有明确的时间轴，也没有强制第 6 帧依赖第 5 帧，容易得到：

- CRPS 变好，但 CSI、SSIM 和 MAE 变差；
- 每帧看起来有随机性，跨帧演化却不连贯；
- 训练时能看到完整真值残差，推理时只能依靠自己生成的结果，存在 exposure gap。

TrajRes-DF 改为 **4 段 × 5 帧** 的顺序残差生成。每个 segment 内保留 `[B,C,T,H,W]` 时间轴，segment 间以前一段生成残差作为 prefix。

## 3. J-Core 网络结构

### 3.1 输入与输出

- 历史观测：`x: [B,5,1,128,128]`
- PhyDNet 轨迹：`mu: [B,20,1,128,128]`
- 真实残差：`r=y-mu`
- 单次去噪目标：5 帧残差 segment
- 推理顺序：`r[0:5] -> r[5:10] -> r[10:15] -> r[15:20]`

### 3.2 Temporal Residual Denoiser

每帧先通过共享 2D spatial encoder 降到 1/4 分辨率；随后在低分辨率特征上使用 depthwise 3D convolution，显式沿 5 帧时间轴交换信息，最后上采样回原分辨率预测 diffusion velocity。

这样保留了 2D 卷积的空间效率，同时不再把时间混同为无序 channels。

### 3.3 Trajectory Context Adapter

历史 5 帧与 PhyDNet 的完整 20 帧轨迹分别编码为 trajectory tokens。当前 segment 的每个 lead 作为 query，通过 multi-head cross-attention 读取完整轨迹。

作用不是重新预测 PhyDNet，而是回答：

- 当前 lead 的确定性轨迹处于怎样的运动阶段；
- 这一段残差应该在哪些区域增强、削弱或移动；
- 后续 segment 如何与完整 100 分钟轨迹保持一致。

### 3.4 Diffusion parameterization

- 训练目标：`v-prediction`
- diffusion steps：100
- 正式推理：DDIM 20 steps
- residual normalization：由冻结的 PhyDNet 在完整 train split 上逐 lead 估计 center / scale
- ensemble：正式结果 `K=10`

### 3.5 Noisy-prefix training

第一段 prefix 为零；后续段从上一段真实残差构造 prefix：

- 25% clean prefix；
- 65% noisy prefix；
- 10% zero prefix。

这对应 DiffCast 中“训练与推理输入不同”的核心思想，但额外用 noisy prefix 主动缩小差距。推理时 prefix 永远来自上一段模型生成结果。

## 4. 联合训练

训练 stage 为 `joint_residual`：

\[
\mathcal L = 0.5\,\mathcal L_{\rm PhyDNet}
            +0.5\,\mathcal L_{\rm diffusion}.
\]

- epoch 1–5：PhyDNet warm-up 冻结，只训练 TrajRes-DF；
- epoch 6–200：端到端联合优化；
- PhyDNet LR：`1e-5`；
- TrajRes-DF LR：`4e-4`；
- bf16，8 卡，每卡 batch size 8；
- 每 5 epoch 在 `val_model` 选择最佳 checkpoint。

训练日志同时保存：

- `metrics/train_log.json`：兼容原有汇总格式；
- `metrics/train_log.jsonl`：每次 log 立即 append + flush；
- `loss_det`、`loss_diff`、总 loss；
- 两部分 grad norm 与 learning rate；
- 当前 segment、prefix mode、残差幅度。

## 5. 训练结束自动 full test

正式配置启用 `post_training_test`。正常完成 200 epoch 后，8 个训练进程会：

1. 自动读取 `checkpoints/checkpoint_best.pt`；
2. 在完整 `report_test`（5600 样本，不允许 `max_samples`）上测试；
3. 使用 ensemble size 10、DDIM 20 steps；
4. 保存 `metrics/report_test_best_full.json`；
5. 自动追加一行到仓库根目录 `RESULTS_REGISTRY.csv`；
6. 以 metrics 文件路径做幂等判断，避免同一结果重复登记。

因此不再需要训练完以后手工寻找 checkpoint、重新领测试任务。

## 6. checkpoint 兼容性

旧 checkpoint 没有迁移，也没有修改其模块结构。原来的：

- `deterministic`；
- `residual`；
- SDIR / U-DIP / universal residual diffusion

仍按原逻辑加载。

新 `joint_residual` checkpoint 继续沿用顶层键：

- `deterministic`：联合更新后的 PhyDNet；
- `diffusion`：完整 TrajRes-DF（包括 denoiser、diffusion schedule、residual statistics）；
- `optimizer`、`epoch`、`step`、`val_loss`。

现有 `evaluate_residual_diffcast.py` 无需专门识别新网络即可加载和测试。

## 7. 已实现文件

- `src/phyrd/models/probabilistic/trajres_diffusion/model.py`
- `src/phyrd/models/probabilistic/trajres_diffusion/__init__.py`
- `src/phyrd/models/probabilistic/__init__.py`
- `src/phyrd/models/composer.py`
- `src/phyrd/models/deterministic/phydnet_external/model.py`
- `scripts/train.py`
- `scripts/evaluation/post_training.py`
- `src/phyrd/evaluation/results_registry.py`
- `scripts/estimate_residual_stats.py`
- `configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml`
- `tests/test_trajres_diffusion.py`

## 8. 下一次要跑的实验（唯一主实验）

实验名：**J-Core PhyDNet + TrajRes-DF joint**。

### 8.0 当前远端执行状态

- 节点：`weather-30240`
- 排除卡：物理 GPU 4。空闲时核心温度约 51–52°C，其他卡约 25–31°C。
- 使用卡：`CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7`
- tmux 会话：`wzq`
- 校准窗口：`wzq:trajres_calib`
- 接力窗口：`wzq:trajres_pipeline`
- 校准日志：`artifacts/calibration/phydnet_residual_stats_train_5to20.log`
- 正式训练日志：`artifacts/logs/trajres_joint_7gpu_200e.log`
- 接力规则：校准成功并通过 20-lead 统计检查后，直接运行 200 epoch
  正式训练；按用户指示跳过 smoke。

启动脚本：

- `scripts/launch_trajres_calibration_7gpu.sh`
- `scripts/launch_trajres_after_calibration_7gpu.sh`

### 8.1 先估计 PhyDNet train residual statistics

这一步只跑冻结模型推理，不是新训练实验；输出是正式训练的固定归一化参数。

```bash
cd /test1/wzq/PhyRD
conda activate /test1/wzq/envs/PhyRD
torchrun --standalone --nproc_per_node=8 scripts/estimate_residual_stats.py \
  --config configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml \
  --output artifacts/calibration/phydnet_residual_stats_train_5to20.json \
  --split train
```

必须检查输出满足：

- `samples` 等于完整 train split 数量；
- center / scale 都正好 20 个；
- scale 全部大于 `1e-4`；
- 文件路径与正式 YAML 的 `residual_stats_path` 一致。

### 8.2 先做一次代码 smoke test

只验证 DDP、显存、loss 与 checkpoint，不登记为实验结果：

```bash
torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml \
  --max-steps 20
```

smoke artifact 使用独立目录，确认无误后清除，不能与正式实验共用目录。

### 8.3 启动 200 epoch 正式训练

```bash
tmux new -s wzq
cd /test1/wzq/PhyRD
conda activate /test1/wzq/envs/PhyRD
torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml
```

不要额外启动手工 test；正常训练结束会自动完成 full `report_test`。

## 9. J-Core 判定标准

必须与同一协议下的 PhyDNet 确定性正式基线 `R-006` 比较：

| 指标 | PhyDNet baseline | J-Core 最低目标 |
|---|---:|---:|
| CRPS | 7.824252 | 明显下降 |
| CSI | 0.269791 | 不低于 baseline |
| HSS | 0.347121 | 不低于 baseline |
| MAE | 0.030683 | 不高于 baseline |
| SSIM | 0.700970 | 不低于 baseline |

核心成功条件是：**CRPS 提升，同时 CSI/SSIM 不再像 universal probability 和 U-DIP 那样明显下降。**

## 10. J-Core 之后只保留一个条件分支

- 如果 CRPS、CSI、SSIM 同时达到最低目标：先做可视化与分 lead / 分阈值分析，不立即堆模块。
- 如果 CRPS 明显提升，但高阈值 CSI 或 SSIM 略低：下一组才加入 high-frequency gated residual，命名 J-Final。
- 如果整体都没有提升：先检查 residual scale、prefix 分布和 segment error accumulation，不进入高频门控。

一天只跑一组训练实验，因此当前队列中没有 P1–P5 五组全量消融；先用一个完整 J-Core 回答“更好的 temporal residual denoiser 是否成立”。
