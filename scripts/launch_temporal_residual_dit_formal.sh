#!/usr/bin/env bash
set -euo pipefail

cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
export PYTHONUNBUFFERED=1

/test1/wzq/envs/PhyRD/bin/python -m torch.distributed.run \
  --standalone \
  --master_port 29691 \
  --nproc_per_node=7 \
  scripts/train.py \
  --config configs/active/5to20/train_ddp7_phydnet_temporal_residual_dit_5to20_v15_seed42.yaml \
  2>&1 | tee artifacts/temporal_residual_dit_joint_ddp7_b16_launch.log
