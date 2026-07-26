#!/usr/bin/env bash
set -euo pipefail

cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
export CUDA_VISIBLE_DEVICES=4
export PYTHONUNBUFFERED=1

RUN_DIR=artifacts/experiments/phydnet_external_temporal_residual_dit_joint_ddp7_b16/20260726_000430
CHECKPOINT="$RUN_DIR/checkpoints/checkpoint_best.pt"
mkdir -p "$RUN_DIR/metrics" "$RUN_DIR/visualizations"

/test1/wzq/envs/PhyRD/bin/python scripts/evaluate_composite_checkpoint.py \
  --config configs/active/5to20/train_ddp7_phydnet_temporal_residual_dit_5to20_v15_seed42.yaml \
  --checkpoint "$CHECKPOINT" \
  --output "$RUN_DIR/metrics/report_test_best_k10.json" \
  --split report_test \
  --batch-size 1 \
  --num-workers 4 \
  --ensemble-size 10 \
  --sampling-steps 20 \
  2>&1 | tee "$RUN_DIR/metrics/report_test_best_k10.log"

/test1/wzq/envs/PhyRD/bin/python scripts/visualize_compare_trajres.py \
  --config configs/active/5to20/train_ddp7_phydnet_temporal_residual_dit_5to20_v15_seed42.yaml \
  --checkpoint "$CHECKPOINT" \
  --deterministic-checkpoint /test1/wzq/Weather/PhyDNet/save/sevir_diffcast_setting/phydnet_sevir_5in20out_best.pth \
  --data /test1/wzq/Weather/PhyDNet/data/sevir/sevir_vil_only_25frames_384_diffcast.h5 \
  --output "$RUN_DIR/visualizations/report_test_best_k10.png" \
  --ensemble-size 10 \
  --sampling-steps 20 \
  --method-label "Temporal Residual DiT" \
  --device cuda:0 \
  2>&1 | tee "$RUN_DIR/visualizations/report_test_best_k10.log"
