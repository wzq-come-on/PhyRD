# Unified SEVIR evaluation

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
torchrun --standalone --nproc_per_node=8 evaluate_5to20.py \
  --model phydnet --checkpoint /path/to/checkpoint.pth \
  --data-path /path/to/sevir_vil_only_25frames_384_diffcast.h5 \
  --output /path/to/results/phydnet_5to20.json \
  --visualization-dir /path/to/results/phydnet_5to20_visuals
```

For `sdir`/`phyrd`, add `--config /path/to/experiment.yaml`; for `13to12`,
use the corresponding 13-to-12 checkpoint and dataset root.
