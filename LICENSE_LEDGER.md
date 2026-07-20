# License Ledger

| Component | Source | License | Usage |
|---|---|---|---|
| PhyRD main package | This workspace | Project-owned / not yet published | Independent implementation |
| DiffCast official | `DeminYu98/DiffCast@e934093` | GPL-3.0 | Isolated baseline only; no source copied or imported |
| SEVIR loader design reference | MIT SEVIR public format | MIT-compatible format knowledge | Independent minimal HDF5 reader |
| SDIR paper/official repository | `RuntimeWarning/SDIR` / arXiv:2606.02661 | README claims MIT; referenced LICENSE file absent in inspected checkout | Official source under `third_party/sdir_official`; PhyRD adapter in `src/phyrd/models/deterministic/sdir_official.py` |
| PyTorch | PyTorch project | BSD-style | Runtime dependency |
| LPIPS | richzhang/PerceptualSimilarity | BSD-2-Clause | Evaluation dependency |
| OpenCV | OpenCV project | Apache-2.0 | Optional Farneback motion estimator |
