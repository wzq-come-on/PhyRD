#!/usr/bin/env bash
set -euo pipefail

while pgrep -f 'master_port 29704' >/dev/null 2>&1; do
  sleep 20
done

cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

/test1/wzq/envs/PhyRD/bin/python scripts/visualize_compare_k1_k10.py \
  --config configs/active/5to20/train_ddp7_phydnet_temporal_residual_dit_5to20_v15_seed42.yaml \
  --checkpoint artifacts/experiments/phydnet_external_temporal_residual_dit_joint_ddp7_b16/20260726_000430/checkpoints/checkpoint_last.pt \
  --deterministic-checkpoint /test1/wzq/Weather/PhyDNet/save/sevir_diffcast_setting/phydnet_sevir_5in20out_best.pth \
  --data /test1/wzq/Weather/PhyDNet/data/sevir/sevir_vil_only_25frames_384_diffcast.h5 \
  --output artifacts/experiments/phydnet_external_temporal_residual_dit_joint_ddp7_b16/20260726_000430/visualizations/compare_phydnet_temporal_k1_k10_last.png \
  --sampling-steps 20 \
  --device cuda:0
