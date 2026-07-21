#!/usr/bin/env bash
set -euo pipefail
cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
exec > /test1/wzq/PhyRD/artifacts/eval_residual_vpred_bs32_best_test.log 2>&1
exec /root/miniconda3/envs/sdir/bin/torchrun --standalone --nproc_per_node=8 \
  python -m scripts.evaluate --mode residual_diffcast \
  --config configs/diagnostics/train_ddp8_residual_diffcast_5to20_vpred_bs32.yaml \
  --checkpoint artifacts/experiments/phyrd_residual_vpred_diffcast_5to20_ddp8_bs32_seed42/checkpoint_best.pt \
  --output artifacts/eval_residual_vpred_bs32_best_test.json \
  --split test --batch-size 8 --num-workers 4 --sampling-steps 20 --device cuda:0
