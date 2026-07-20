# Official DiffCast Audit

> v10.3 update (2026-07-20): this audit records the earlier `13→12@384` comparison rationale. The active PhyRD protocol is now the matched DiffCast HDF5 `5→20@128` task; use `PROTOCOL.yaml` for current split and calibration rules.

Audit date: 2026-07-15

- Repository: `https://github.com/DeminYu98/DiffCast.git`
- Commit: `e9340933556d7e351a087a79429a663c3498d738`
- License: GPL-3.0
- Published default protocol in code: `frames_in=5`, `frames_out=20`, `img_size=128`, `seq_len=25`.
- Environment lock: Python 3.8.5, PyTorch 1.12.1, CUDA 11.3, Diffusers 0.27.2.
- SEVIR split in official loader: train before 2019-01-01, validation until 2019-10-01, test afterwards. It does not provide the dedicated `val_model/val_calib/report_test` separation required by PhyRD.
- Official loader uses generic 25-frame windows with stride 5. The PhyRD main protocol instead fixes indices `[12:37]` from each 49-frame event.
- Spatial preprocessing has two distinct code paths: the effective `SEVIRTorchDataset` path normalizes VIL by `/255` and applies `transforms.Resize((img_size,img_size))` with default `img_size=128`; under its locked torchvision 0.13.1 environment this is the legacy non-antialiased tensor bilinear path. A generic `(t,h,w)` downsample helper uses `avg_pool2d`, but the wrapper passes `downsample_dict=None`, so that helper is not active in official SEVIR runs.
- `datasets/preprocess.py` is a Shanghai-radar PNG-to-HDF5 script with 5→20 windows; it is not the SEVIR preprocessing path and must not be copied for the 13→12 protocol.
- The diffusion U-Net treats forecast time as 2D channels and is tightly coupled to its deterministic backbone and legacy training runner.

Conclusion: keep the official repository runnable in an independent environment for baseline reproduction, but do not modify it into the main `13→12@384` research codebase.
