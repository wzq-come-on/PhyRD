#!/usr/bin/env bash
set -u

EXP=/test1/wzq/PhyRD/artifacts/experiments/phyrd_sdir_source_diffcast_5to20_ddp8_seed42
SUMMARY="$EXP/run_summary.json"
CHECKPOINT="$EXP/checkpoint_best.pt"
OUT=/test1/wzq/Weather/evaluate/results/sdir_source_final_5to20.json
VIS=/test1/wzq/Weather/evaluate/results/sdir_source_final_5to20_visuals
CONFIG=/test1/wzq/PhyRD/configs/active/5to20/train_ddp8_sdir_source_diffcast_5to20.yaml
DATA=/test1/wzq/Weather/PhyDNet/data/sevir/sevir_vil_only_25frames_384_diffcast.h5

echo "Watching $EXP for completed training..."
while true; do
  if [[ -f "$SUMMARY" && -f "$CHECKPOINT" ]] && grep -q '"status": "completed"' "$SUMMARY"; then
    if [[ ! -f "$OUT" ]]; then
      echo "Training completed; evaluating the best validation checkpoint."
      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        /root/miniconda3/envs/sdir/bin/torchrun --standalone --nproc_per_node=8 \
        /test1/wzq/PhyRD/scripts/evaluation/evaluate_5to20.py \
        --model sdir --config "$CONFIG" --checkpoint "$CHECKPOINT" \
        --data-path "$DATA" --split test --batch-size 8 --num-workers 4 \
        --img-size 128 --output "$OUT" --visualization-dir "$VIS" \
        --visualization-samples 0,1,2
    else
      echo "Evaluation already exists: $OUT"
    fi
    break
  fi
  sleep 60
done
