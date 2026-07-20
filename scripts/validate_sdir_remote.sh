#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TMUX:-}" ]] || [[ "$(tmux display-message -p '#S')" != "wzq" ]]; then
  echo "Refusing to run: launch this script inside tmux session wzq." >&2
  exit 2
fi

ROOT="/test1/wzq/PhyRD"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/artifacts/validation_sdir_${STAMP}"
mkdir -p "${OUT}"
cd "${ROOT}"

conda run --no-capture-output -n PhyRD python -m pytest \
  tests/test_model_shapes.py -q 2>&1 | tee "${OUT}/model_tests.log"

conda run --no-capture-output -n PhyRD python scripts/validate_sdir_cuda.py \
  --input-frames 5 --output-frames 20 --resolution 128 \
  2>&1 | tee "${OUT}/cuda_5to20.log"

conda run --no-capture-output -n PhyRD python scripts/train.py \
  --config configs/diagnostics/ddp_probe_deterministic.yaml --max-steps 1 \
  2>&1 | tee "${OUT}/real_sevir_single_gpu.log"

if [[ "${RUN_DDP8:-0}" == "1" ]]; then
  conda run --no-capture-output -n PhyRD torchrun \
    --standalone --nproc_per_node=8 scripts/train.py \
    --config configs/diagnostics/ddp_probe_deterministic_diffcast_5to20.yaml --max-steps 1 \
    2>&1 | tee "${OUT}/real_sevir_ddp8_5to20.log"
else
  echo "DDP8 skipped; set RUN_DDP8=1 only on an eight-GPU host." | tee "${OUT}/ddp8_skipped.log"
fi

sha256sum src/phyrd/models/deterministic/sdir_official.py \
  src/phyrd/models/deterministic/registry.py src/phyrd/models/phyrd.py scripts/train.py \
  > "${OUT}/validated_code_sha256.txt"
echo "SDIR validation artifacts: ${OUT}"
