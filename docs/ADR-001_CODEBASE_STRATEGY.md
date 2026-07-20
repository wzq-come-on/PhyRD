# ADR-001: Codebase Strategy

> v10.3 update (2026-07-20): this ADR preserves the original `13→12@384` route-analysis evidence. The current execution protocol is `5→20@128` under `PROTOCOL.yaml`; this historical ADR does not define the active task.

Status: Accepted, 2026-07-15

## Decision

Use Route B: an independent standard-protocol implementation under `src/phyrd`. Keep official DiffCast under `external_baselines/DiffCast` as an isolated GPL-3.0 baseline.

## Evidence

The target server has four NVIDIA H800 80GB GPUs and can support a 384² pilot, but hardware capacity does not remove the protocol and maintenance mismatches. The official repository targets 128², Python 3.8/PyTorch 1.12, a 5→20 split, and a three-way date split. PhyRD needs a fixed 13→12 event window, four isolated splits, reliability/motion fields, differentiable transport loss, clean-residual proximal guidance, and eight explicit evaluation metrics.

## Consequences

- Fairness is maintained through artifact-level baseline exchange and shared manifests/metrics.
- The main package does not import or copy official DiffCast source.
- Official reproduction remains a separate evidence task; the current delivery validates the new interfaces and real SEVIR I/O, not model quality.
- A future full experiment must benchmark official DiffCast and the backbone-matched clean port under registered compute budgets.
