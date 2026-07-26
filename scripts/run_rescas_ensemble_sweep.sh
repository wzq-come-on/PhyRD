#!/usr/bin/env bash
set -u

ROOT=/test1/wzq/PhyRD
PY=/test1/wzq/envs/PhyRD/bin/python
CONFIG=$ROOT/artifacts/experiments/phydnet_external_rescasformer_ddp7/20260724_151243/config_snapshot.yaml
CHECKPOINT=$ROOT/artifacts/experiments/phydnet_external_rescasformer_ddp7/20260724_151243/checkpoints/checkpoint_best.pt
OUTDIR=$ROOT/artifacts/experiments/phydnet_external_rescasformer_ddp7/20260724_151243/metrics
K1=$OUTDIR/report_test_k1_full.json
STOP_AT='2026-07-25 21:10:00'

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
export PYTHONUNBUFFERED=1

echo "[sweep] waiting for K=1 result: $K1"
while [ ! -s "$K1" ]; do
  sleep 30
done

for K in 4 10 20; do
  now=$(date +%s)
  deadline=$(date -d "$STOP_AT" +%s)
  remaining=$((deadline - now))
  if [ "$remaining" -le 300 ]; then
    echo "[sweep] deadline reached; stopping before K=$K"
    break
  fi
  output="$OUTDIR/report_test_k${K}_full.json"
  log="$OUTDIR/report_test_k${K}_full.log"
  echo "[sweep] starting K=$K, timeout=${remaining}s"
  timeout --foreground "$remaining" "$PY" -m torch.distributed.run \
    --standalone --nproc_per_node=7 \
    scripts/evaluate_composite_checkpoint.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --output "$output" \
    --split report_test \
    --batch-size 1 \
    --num-workers 4 \
    --sampling-steps 20 \
    --ensemble-size "$K" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 124 ]; then
    echo "[sweep] K=$K reached the 21:10 deadline; releasing GPUs"
    break
  fi
  if [ "$rc" -ne 0 ]; then
    echo "[sweep] K=$K failed with exit code $rc; stopping"
    break
  fi
done
echo "[sweep] finished; GPUs are released"
