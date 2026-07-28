# Unified SEVIR evaluation

Use one public CLI for every evaluation mode:

```bash
python -m scripts.evaluate --mode protocol --protocol 5to20 ...
python -m scripts.evaluate --mode residual_diffcast ...
python -m scripts.evaluate --mode deterministic_diffcast ...
python -m scripts.evaluate --mode artifact --predictions predictions.npz
python -m scripts.evaluate --mode report --row SDIR 确定性 1 sdir.json --row PhyDNet 确定性 1 phydnet.json
```

`common.py` owns the shared metric and visualization implementation. The
residual and deterministic modules are model adapters invoked by the dispatcher;
their direct file paths are retained only as compatibility aliases for old
launchers.

For a `deterministic_pool` configuration, residual evaluation must name the
backbone explicitly, for example `--mode residual_diffcast --backbone sdir_official`.

The two entry points share one metric implementation:

- `evaluate_5to20.py`: five input frames, twenty output frames.
- `evaluate_13to12.py`: thirteen input frames, twelve output frames.

CSI is computed globally over the complete split at all six SEVIR VIL
thresholds: `16, 74, 133, 160, 181, 219`. The JSON contains the mean `CSI`,
each `CSI_<threshold>`, `CSI_pool4`, `CSI_pool16`, all pooled threshold values,
per-threshold HSS, MAE, MSE, CRPS, SSIM, LPIPS, lead-time MAE, and visualization
paths. `CSI_pool4` and `CSI_pool16` are spatial pooling scales, not intensity
thresholds.

Example (8-card DDP on the shared server):

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.evaluate \
  --mode protocol --protocol 5to20 \
  --model phydnet --checkpoint /path/to/checkpoint.pth \
  --data-path /path/to/sevir_vil_only_25frames_384_diffcast.h5 \
  --output /path/to/results/phydnet_5to20.json \
  --visualization-dir /path/to/results/phydnet_5to20_visuals
```

For `sdir`/`phyrd`, add `--config /path/to/experiment.yaml`; for `13to12`,
use `--protocol 13to12` with the corresponding checkpoint and dataset root.

## Paper-style comparison table

After running the protocol evaluator for each method, combine the JSON files
into the table used in the paper:

```bash
python -m scripts.evaluate --mode report \
  --row "SDIR" "确定性" 1 /path/to/sdir.json \
  --row "PhyDNet" "确定性" 1 /path/to/phydnet.json \
  --row "SDIR + U-DIP" "残差扩散" 4 /path/to/udip.json \
  --output /path/to/comparison_5to20.md
```

The first row is the reference for the automatically generated conclusion
column. The report uses `CSI_global` (falling back to `CSI`), `SSIM`, `HSS`,
`CRPS`, and `LPIPS`, and writes `—` when a metric was not evaluated.
