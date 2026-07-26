#!/usr/bin/env bash
set -euo pipefail

cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
export PYTHONUNBUFFERED=1

/test1/wzq/envs/PhyRD/bin/python -m torch.distributed.run \
  --standalone \
  --master_port 29690 \
  --nproc_per_node=7 \
  scripts/train.py \
  --config configs/diagnostics/temporal_residual_dit_joint_ddp7_smoke_5to20.yaml \
  2>&1 | tee artifacts/temporal_residual_dit_joint_ddp7_b16_smoke.log
