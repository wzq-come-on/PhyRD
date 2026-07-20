#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
RUN_ROOT="artifacts/experiments/phyrd_sdir_13to12_ddp8_seed42"

mkdir -p "${RUN_ROOT}"
cp configs/archive/train_ddp8_deterministic_sevir.yaml "${RUN_ROOT}/"
cp configs/archive/train_ddp8_residual_sevir.yaml "${RUN_ROOT}/"
nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,driver_version \
  --format=csv,noheader > "${RUN_ROOT}/gpu_at_launch.csv"
sha256sum scripts/train.py src/phyrd/models/phyrd.py \
  src/phyrd/models/deterministic/sdir_official.py PROTOCOL.yaml \
  > "${RUN_ROOT}/code_sha256.txt"

torchrun --standalone --nproc_per_node=8 scripts/train.py \
  --config configs/archive/train_ddp8_deterministic_sevir.yaml \
  2>&1 | tee -a "${RUN_ROOT}/deterministic_console.log"

if [[ "${RUN_RESIDUAL:-0}" == "1" ]]; then
  torchrun --standalone --nproc_per_node=8 scripts/train.py \
    --config configs/archive/train_ddp8_residual_sevir.yaml \
    2>&1 | tee -a "${RUN_ROOT}/residual_console.log"
else
  echo "Residual diffusion stage is disabled; set RUN_RESIDUAL=1 explicitly to enable it."
fi
