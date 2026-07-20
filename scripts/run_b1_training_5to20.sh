#!/usr/bin/env bash
set -euo pipefail
cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
exec > /test1/wzq/PhyRD/artifacts/train_residual_b1_v2.log 2>&1
exec /root/miniconda3/envs/sdir/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/train.py \
  --config configs/archive/train_ddp8_residual_diffcast_5to20_vpred_bs32_b1_v2.yaml
