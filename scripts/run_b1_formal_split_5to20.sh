#!/usr/bin/env bash
set -euo pipefail
cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
exec > /test1/wzq/PhyRD/artifacts/train_residual_b1_formal_split_attempt2.log 2>&1
exec /root/miniconda3/envs/sdir/bin/torchrun --standalone --master-port 29618 --nproc_per_node=8 \
  scripts/train.py \
  --config configs/active/5to20/train_ddp8_residual_diffcast_5to20_vpred_b1_formal_split_seed42.yaml
