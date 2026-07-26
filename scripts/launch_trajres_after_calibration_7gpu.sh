#!/usr/bin/env bash
set -uo pipefail

cd /test1/wzq/PhyRD
mkdir -p artifacts/logs

calibration_path=artifacts/calibration/phydnet_residual_stats_train_5to20.json
calibration_exit=artifacts/calibration/phydnet_residual_stats_train_5to20.exit
formal_log=artifacts/logs/trajres_joint_7gpu_200e.log
formal_exit=artifacts/logs/trajres_joint_7gpu_200e.exit
config=configs/active/5to20/train_ddp8_phydnet_trajres_joint_5to20_v13_seed42.yaml
python=/test1/wzq/envs/PhyRD/bin/python

while [[ ! -f "$calibration_exit" ]]; do
  sleep 30
done

if [[ "$(tr -d '[:space:]' < "$calibration_exit")" != "0" ]]; then
  printf 'Calibration failed; formal pipeline stopped.\n'
  exit 1
fi

env STATS_PATH="$calibration_path" "$python" -c '
import json
import os
from pathlib import Path

stats = json.loads(Path(os.environ["STATS_PATH"]).read_text())
assert stats["samples"] > 0
assert len(stats["center"]) == 20
assert len(stats["scale"]) == 20
assert all(value >= 1e-4 for value in stats["scale"])
print("residual statistics validated:", stats["samples"], "samples")
'

rm -f "$formal_exit"
env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
  PYTHONPATH=src:. \
  "$python" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=7 \
    scripts/train.py \
    --config "$config" \
    2>&1 | tee "$formal_log"
formal_status=${PIPESTATUS[0]}
printf '%s\n' "$formal_status" > "$formal_exit"
exit "$formal_status"
