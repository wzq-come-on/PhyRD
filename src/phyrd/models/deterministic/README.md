# Deterministic backbones

This directory is the only supported home for PhyRD deterministic forecast
adapters. Upstream repositories stay under `third_party/`; each adapter here
normalizes the model to PhyRD's `[B,T,1,H,W]` interface.

## Add a backbone

1. Add one adapter package in this directory when the model has multiple
   components; a single module is acceptable for a genuinely small adapter.
2. Add one lazy `register_backbone("name", "module:Class")` entry in
   `deterministic/__init__.py`. Lazy registration prevents one backbone's
   optional dependencies from being required when another backbone is used.
4. Implement `forward(history)` for inference. `training_loss(history, target)`
   must return `DeterministicLossOutput`; backbone-specific scalar diagnostics go
   in its `metrics` mapping, so the shared trainer remains model-agnostic.
5. Select it in YAML:

```yaml
model:
  deterministic:
    name: name
    params:
      model_resolution: 128
```

Checkpoint protocol metadata stores this complete `name`/`params` mapping, so
incompatible deterministic models cannot be loaded silently.
