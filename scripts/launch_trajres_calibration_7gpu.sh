#!/usr/bin/env bash
set -uo pipefail

cd /test1/wzq/PhyRD
mkdir -p artifacts/calibration

log_path=artifacts/calibration/phydnet_residual_stats_train_5to20.log
exit_path=artifacts/calibration/phydnet_residual_stats_train_5to20.exit
rm -f "$exit_path"

env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
  PYTHONPATH=src:. \
  /test1/wzq/envs/PhyRD/bin/python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=7 \
    scripts/estimate_residual_stats.py \
    --config configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml \
    --output artifacts/calibration/phydnet_residual_stats_train_5to20.json \
    --split train \
    2>&1 | tee "$log_path"

status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$exit_path"
exit "$status"
