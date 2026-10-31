#!/usr/bin/env bash
# AI fill on macOS (Apple silicon) or Linux (1 Nov 2026): the Python it runs in and the base model.
# The Windows counterpart is scripts\GENFILL.ps1 -Setup.
#
#   scripts/genfill_setup.sh            # .venv + packages + Wan2.1-VACE-1.3B (about 25 GB) into models/genfill
#
# Then, in the app: File > Model packages > AI fill v3 > Download (the LoRA and brightness.json), and
# Deliver > AI fill > Setup… points at this folder if the app is not inside it. On a Mac the fill runs
# on the GPU through Metal (PyTorch MPS) in float32; 16 GB of memory is the least, 32 GB is comfortable.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
BASE=${RUDRA_GENFILL_BASE:-models/genfill/Wan2.1-VACE-1.3B-diffusers}

[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch          # PyPI's wheels: Metal (MPS) on a Mac, CUDA on Linux
.venv/bin/python -m pip install -e ".[genfill]" "opencolorio==2.5.2"
mkdir -p "$(dirname "$BASE")"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Wan-AI/Wan2.1-VACE-1.3B-diffusers', local_dir='$BASE')"
.venv/bin/python - <<'PY'
import torch
from rudra.genfill.run import pick_device
print('torch', torch.__version__, '-> AI fill runs on', *pick_device('auto'))
PY
echo "AI fill is set up: $(pwd)/.venv, base model in $BASE"
