# PhyRD 工程化模型组合与实验目录重构方案（v11）

> 状态：方案记录，尚未开始实施
>
> 本文记录当前讨论形成的工程化方向。现阶段先完成昨晚扩散实验的独立 test，再由实验结果决定是否收尾；只有确认旧实验可以结束后，才进行目录和训练框架重构。

## 1. 当前阶段与执行顺序

PhyRD 的最终目标是一个“确定性模型 + 概率性模型”的组合预测系统。确定性部分负责给出趋势或条件基线，概率性部分在此基础上建模未来的不确定性。

当前不立即移动目录或改训练器，执行顺序固定为：

1. 使用与其他 5→20 模型一致的 `report_test` 协议，评估昨晚训练的扩散实验。
2. 查看确定性基线与加入扩散后的指标、样例和可视化结果。
3. 由用户判断该扩散结果是否已经达到可以收尾的程度。
4. 如果确认收尾，停止/关闭对应旧实验，但保留其 checkpoint、日志、配置和结果。
5. 在旧实验可回滚、可复现的前提下，再开始目录、模型接口和训练流程的工程化重构。

本轮只记录方案，不修改现有训练代码，不移动现有文件，不覆盖或删除已有实验产物。

## 2. 目标模型组织方式

### 2.1 顶层概念

模型由两个可独立替换的部分组成：

```text
deterministic model  ->  trend / conditional baseline
                                      \
                                       composite forecast
probabilistic model  ->  samples / distribution
```

训练器不应知道某个概率模型的内部实现，也不应把某个物理假设写死在 `train.py` 中。模型选择、超参数和可选扩展均由配置文件控制。

### 2.2 确定性模型目录

每个确定性模型使用一个独立目录，而不是把一个模型压缩成单个巨型 Python 文件：

```text
src/phyrd/models/deterministic/
├── base.py
├── registry.py
├── sdir_official/
│   ├── __init__.py
│   ├── model.py
│   ├── encoder.py
│   ├── decoder.py
│   └── losses.py
├── simvp/
│   ├── __init__.py
│   ├── model.py
│   └── ...
└── ...
```

目录中的组件只服务于对应的确定性模型。真正跨模型复用的底层算子才放到公共模块中。

### 2.3 概率性模型目录

每个概率模型同样使用一个独立目录；目录内部可以包含该模型所需的多个组件：

```text
src/phyrd/models/probabilistic/
├── base.py
├── registry.py
├── residual_diffusion/
│   ├── __init__.py
│   ├── model.py
│   ├── denoiser.py
│   ├── scheduler.py
│   ├── sampler.py
│   └── extensions/
│       └── physics_guidance.py
├── jdir/
│   ├── __init__.py
│   ├── model.py
│   ├── registration.py
│   ├── denoiser.py
│   ├── reconstruction.py
│   ├── diffusion.py
│   └── extensions/
│       ├── physics_guidance.py
│       └── calibration.py
├── flow_matching/
│   ├── __init__.py
│   ├── model.py
│   ├── vector_field.py
│   └── sampler.py
└── ...
```

后续新增概率模型时，只需新增一个模型目录并注册，不需要继续向训练器添加越来越多的分支。

## 3. 物理、校准和运动模块的归属

物理设计不一定是最终路线，因此物理不是 PhyRD 的核心硬编码假设，而是某个概率模型的可选扩展。

模型相关的扩展放在该模型目录内部，例如：

```text
probabilistic/jdir/extensions/physics_guidance.py
probabilistic/jdir/extensions/calibration.py
probabilistic/residual_diffusion/extensions/physics_guidance.py
```

不再在模型目录之外维护一个与所有模型强绑定的顶层 `extensions/`。如果某些底层物理算子确实被多个模型共同使用，可以保留为公共低层工具；但“如何把物理项接入某个概率模型”应由该概率模型自己的扩展负责。

这样做的好处是：

- 继续走物理约束路线时，只启用对应的 `physics_guidance` 扩展；
- 后续放弃物理路线时，只需把扩展列表设为空，或归档该扩展，不会破坏确定性模型和概率模型主体；
- 校准、运动先验等也可以按模型独立替换，不会污染公共训练器。

## 4. 统一接口与配置驱动选择

### 4.1 配置示例

```yaml
model:
  deterministic:
    name: sdir_official
    params:
      checkpoint: /path/to/sdir_checkpoint.pt
  probabilistic:
    name: jdir
    params:
      prediction_type: v
    extensions: []

train:
  mode: probabilistic
```

启用物理扩展时，仅修改配置：

```yaml
model:
  probabilistic:
    name: jdir
    extensions:
      - name: physics_guidance
        enabled: true
        params:
          lambda_train: 0.01
```

### 4.2 工厂和组合器

建议新增两个公共入口：

- `factory.py`：根据配置中的 `name` 从 registry 构造确定性模型、概率模型及其扩展；
- `composer.py`：把确定性模型和概率模型组合为完整的预测器。

训练脚本只负责读取配置、数据迭代、优化器、日志和 checkpoint 调度，不负责判断当前是不是 SDIR、残差扩散、JDIR 或物理扩散。

### 4.3 最小统一接口

确定性模型提供：

```python
trend = deterministic.predict(history)
```

概率模型提供：

```python
losses = probabilistic.training_loss(
    history=history,
    target=target,
    trend=trend,
)

ensemble = probabilistic.sample(
    history=history,
    trend=trend,
    ensemble_size=n,
)
```

扩展通过概率模型的 hook 参与训练或采样；训练器不需要理解“物理”这个词，也不需要写 `if stage == ...` 的模型专用逻辑。

## 5. 实验与 checkpoint 目录

实验目录使用“确定性模型名_概率性模型名”作为组合名，再使用完整时间戳区分同一组合的多次运行：

```text
artifacts/experiments/
└── sdir_official_jdir/
    └── 20260721_143052/
        ├── checkpoints/
        │   ├── checkpoint_best.pt
        │   └── checkpoint_last.pt
        ├── metrics/
        │   ├── train.jsonl
        │   ├── val_model.json
        │   └── report_test.json
        ├── predictions/
        │   └── report_test/
        │       ├── arrays/
        │       └── visualizations/
        ├── logs/
        └── config_snapshot.yaml
```

约定如下：

- 时间戳格式固定为 `YYYYMMDD_HHMMSS`，不再额外使用短随机 ID；
- 同一确定性/概率性组合可以运行多次，每次都有独立时间戳目录，绝不覆盖；
- `checkpoint_best.pt` 和 `checkpoint_last.pt` 必须位于 `checkpoints/`，不再直接放在 artifact 根目录；
- 每个运行目录必须保存完整解析后的 `config_snapshot.yaml`，不能只依赖外部 YAML；
- 不强制增加独立的 `run_manifest.yaml`，Git、服务器、GPU 等信息暂不作为必须字段；
- `report_test` 结果允许用户手动删除和重跑，但系统不得在没有明确意图的情况下静默覆盖；如需覆盖，应使用显式 `--overwrite` 或先手动清理目标目录。

确定性单独实验可采用：

```text
artifacts/experiments/sdir_official_deterministic_only/20260721_143052/
```

## 6. 共享代码与评估位置

通用数据、评估和实验工具继续集中管理：

```text
src/phyrd/data/          # 数据集、协议切分、加载
src/phyrd/evaluation/    # CRPS、CSI、HSS、MAE、校准等
src/phyrd/models/        # 确定性/概率性模型及组合器
src/phyrd/train/         # 通用训练循环、checkpoint、日志
scripts/                 # 薄 CLI，仅负责参数解析和调用库代码
configs/                 # active、diagnostics、archive 分层
artifacts/               # 本地实验产物，不进入 Git
```

评估逻辑只保留一套核心实现和一套 CLI。CLI 不再通过修改 `sys.path` 去引用另一个目录中的评估代码；脚本应直接导入 `phyrd.evaluation` 的公共 API。

继续使用现有数据协议：

- `train`：训练；
- `val_model`：模型选择；
- `val_calib`：概率风险校准；
- `report_test`：最终横向报告。

同一 checkpoint 的 `report_test` 仍可与其他 5→20 模型保持一致的测试协议；`val_model` 和 `val_calib` 只服务于模型选择和校准，不改变最终横向测试定义。

## 7. 迁移原则

当前代码仍包含旧结构，例如旧的 `PhyRDModel`、残差扩散实现、训练脚本中的模型分支，以及 artifact 根目录下的旧 checkpoint 布局。本轮不直接删除或重命名这些内容。

后续迁移按兼容优先的顺序执行：

1. 先新增 registry、factory、base interface 和 composer；
2. 给现有 SDIR 和 residual diffusion 增加新目录中的兼容包装；
3. 让旧 import 路径继续可用，避免已有实验脚本立即失效；
4. 增加新的 checkpoint 布局和配置快照；
5. 用新配置启动一个小规模 smoke test；
6. 通过测试后，才把正式训练切换到新布局；
7. 旧实验目录、旧日志和旧 checkpoint 永久保留在 archive/或原位置，除非用户明确要求清理。

现有 `src/phyrd/physics/` 可以暂时保留。只有在新模型接口稳定后，才把“模型特定的物理接入逻辑”迁移到对应概率模型的 `extensions/`，避免一次性进行破坏性移动。

## 8. 必须补充的工程测试

重构完成后至少增加以下测试：

- registry 能按配置找到并构造确定性模型和概率模型；
- 不同概率模型可以在不修改训练器的情况下切换；
- composer 能正确传递 deterministic 的 `trend`；
- `training_loss()` 和 `sample()` 的输出形状、设备和 dtype 正确；
- 每次运行生成完整 `config_snapshot.yaml`；
- checkpoint 始终写入 `checkpoints/checkpoint_best.pt` 和 `checkpoints/checkpoint_last.pt`；
- 同组合多次运行不会覆盖旧目录；
- `report_test` 默认不静默覆盖，显式 `--overwrite` 才允许覆盖；
- 旧 checkpoint 和旧 import 路径仍能被兼容层读取。

## 9. 当前结论

这个目录设计是合理的：确定性模型和概率性模型各自以“模型包”为单位组织，模型内部的小组件跟随模型放置，物理/校准/运动等作为可选扩展放在对应模型内部。这样既支持当前扩散实验，也为后续 JDIR、flow matching 或完全不带物理约束的概率模型留下空间。

原计划是先完成昨晚扩散实验的 `report_test`，再决定是否收尾并迁移。该判断现已完成，B1 已停止并保留产物；后续按第 10 节的迁移记录继续推进。

## 10. v11 第一阶段实施记录（2026-07-21）

本阶段已完成以下低风险迁移：

- 停止正式 B1 训练，保留远端 checkpoint、日志和所有测试 JSON；
- 将确定性 SDIR 实现整理为 `models/deterministic/sdir_official/` 包，并保留旧 import 兼容层；
- 将 residual diffusion 的 denoiser、UNet、扩散调度和采样代码整理到 `models/probabilistic/residual_diffusion/`，并保留旧模块路径兼容层；
- 新增 probabilistic registry、`ForecastComposer` 和 config factory；
- 训练器和 residual evaluator 已改为通过 factory 构建组合模型，旧 checkpoint payload 仍可读取；
- 新 checkpoint 默认写入 `checkpoints/checkpoint_best.pt` 和 `checkpoints/checkpoint_last.pt`，并保存 `config_snapshot.yaml`；
- 实验登记中 E-010 已更新为 stopped，原因是 report_test 未超过 SDIR 基线。

旧 artifact 目录没有被搬动或覆盖。后续仍需在独立分支上完成统一评估 CLI、predictions/metrics 子目录和新概率模型（如 JDIR）的接入。
## 11. 第二阶段收口记录（2026-07-21）

- 评估统一入口为 `python -m scripts.evaluate`；通过 `--mode` 选择 `artifact`、`protocol`、`residual_diffcast` 或 `deterministic_diffcast`。
- `scripts/evaluation/common.py` 继续作为共享指标核心，旧的协议脚本保留为兼容别名；新启动脚本不再直接调用分散的评估文件。
- 评估入口已通过本地编译检查，并在两台服务器完成源码编译与 `phyrd` 注册表导入检查。
- 旧实验产物属于服务器本地数据，不纳入 Git；本轮不复制、不删除、不覆盖 checkpoint、日志和已有测试 JSON。新实验按 v11 目录规范写入独立时间戳目录。
- 新运行目录现在预创建 `checkpoints/`、`metrics/` 和 `predictions/`；训练日志与运行摘要写入 `metrics/`，旧目录仍由兼容读取逻辑支持。

## 12. v11.1：通用概率适配层（2026-07-21）

v11 的 registry/composer 解决的是“可以从配置替换 backbone”；v11.1 的目标进一步变为“同一个概率模型对多个冻结 backbone 有效”。因此不能继续使用某个 SDIR 训练集计算出的 `residual_stats_path`。该统计量会把概率模型绑定为 `P(SDIR residual | history, SDIR trend)`，直接替换成 PhyDNet trend 时会发生残差分布错配。

新的条件概率目标固定为：

```text
P(y | history, trend),  trend = frozen_backbone(history)
residual = y - trend ∈ [-1, 1]
```

所有 backbone 都必须输出 `[B,T,1,H,W]`、值域 `[0,1]` 的 trend。`universal_residual_diffusion` 仅使用 `history`、`trend` 与固定残差坐标；不读取任何 backbone 专属的 residual center/scale。运动、物理或校准仍是可选概率扩展，而不是前提。

训练配置使用 `model.deterministic_pool`。pool 中每个 backbone 均加载独立 checkpoint 并冻结；每个 batch 按 `uniform`、`weighted_random` 或 `round_robin` 规则同步选择一个 backbone。当前第一组 pool 是 SDIR 与外部 PhyDNet adapter：

```text
SDIR checkpoint ───┐
                   ├──> frozen backbone pool ──> universal residual diffusion
PhyDNet checkpoint ┘
```

这不是把两个 backbone 融合成一个确定性网络；每个 batch 只有一个 trend。概率模型必须在两个条件分布上都学习 residual。验证阶段使用配置指定的固定 backbone；正式结果需要分别在每个 backbone 上独立 report_test，并报告“纯 backbone”与“backbone + 同一概率 checkpoint”的差值。

第一个可运行配置为：

```text
configs/active/5to20/train_ddp8_universal_probabilistic_pool_5to20_v11_seed42.yaml
```

它使用唯一时间戳实验目录，并保存配置快照、`checkpoints/`、`metrics/` 和 `predictions/`。旧 SDIR 专属 B1 与其 checkpoint、评估结果保持不变，只作为历史对照。
