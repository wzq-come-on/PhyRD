#!/usr/bin/env bash
set -euo pipefail
cd /test1/wzq/PhyRD
export PYTHONPATH=/test1/wzq/PhyRD/src
exec > /test1/wzq/PhyRD/artifacts/eval_sdir_source_final_test_current.log 2>&1
exec /root/miniconda3/envs/sdir/bin/torchrun --standalone --nproc_per_node=8 \
  scripts/evaluation/evaluate_5to20.py --model sdir \
  --checkpoint artifacts/experiments/phyrd_sdir_source_diffcast_5to20_ddp8_seed42/checkpoint_final.pt \
  --config configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml \
  --data-path /test1/wzq/Weather/PhyDNet/data/sevir/sevir_vil_only_25frames_384_diffcast.h5 \
  --output artifacts/eval_sdir_source_final_test_current.json \
  --split test --batch-size 8 --num-workers 4 --skip-lpips --device cuda:0
