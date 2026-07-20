#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="PhyRD"
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
CONDA_MAIN="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main"
CONDA_R="https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Reusing existing conda environment: ${ENV_NAME}"
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -n "${ENV_NAME}" python=3.11 pip -y \
    --override-channels \
    -c "${CONDA_MAIN}" \
    -c "${CONDA_R}"
fi

if conda run -n "${ENV_NAME}" python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA-enabled PyTorch is already available"
else
  conda run -n "${ENV_NAME}" python -m pip install \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url "${PIP_INDEX_URL}"
fi

conda run -n "${ENV_NAME}" python -m pip install -e '.[dev]' \
  --index-url "${PIP_INDEX_URL}"
conda run -n "${ENV_NAME}" python -c \
  "import torch, phyrd; print('PhyRD', phyrd.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.version.cuda)"
