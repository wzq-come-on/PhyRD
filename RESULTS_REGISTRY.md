# PhyRD 结果登记

唯一的全量结果表是 `RESULTS_REGISTRY.csv`。

规则：

- 只有完整 `report_test`/全量 `test` 才登记为正式结果。
- smoke、probe、overfit、少量样本 sanity check 不登记到正式结果表。
- 每条结果保留确定性 backbone、概率模块、协议、split、样本数、ensemble size、world size、checkpoint epoch 和主要指标。
- 原始 JSON/log 不在本地时，必须在 `evidence` 和 `notes` 中注明“文档记录”或远端路径，不能伪装成本地原始结果。
- 后续每次全量 test 完成后，先追加 CSV，再向用户汇报结果。
