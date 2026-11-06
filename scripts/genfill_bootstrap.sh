#!/bin/bash
# AI fill bootstrap for macOS and Linux (7 Nov 2026): what "Set up automatically" runs when this
# computer has no Python for the fill. Everything goes under ROOT (the app's own data folder), and
# nothing on the system changes:
#   ROOT/uv      uv, the Python installer (downloaded once from its GitHub release)
#   ROOT/python  a private Python 3.12 (uv's managed CPython)
#   ROOT/venv    the fill's environment: PyTorch, diffusers, OpenColorIO, OpenEXR, ...
# then rudra.genfill.setup checks the GPU and the disk and downloads Wan2.1-VACE-1.3B into BASE.
#   genfill_bootstrap.sh ROOT BASE REPO
# REPO: the folder that holds rudra/ (the app's python/ folder, or the repository).
# stdout: one JSON object per line for the app ({"event": "step" | "bootstrap" | ...}); the
# installers' own progress goes to stderr. Exit code: rudra.genfill.setup's (0 ready, 1 failed, 2 no GPU).
# RUDRA_TORCH_INDEX overrides where PyTorch comes from (default: PyPI on macOS, CUDA 12.8 on Linux).
set -euo pipefail
ROOT="$1"; BASE="$2"; REPO="$3"
step() { printf '{"event": "step", "step": "bootstrap", "text": "%s"}\n' "$1"; }
fail() { printf '{"event": "done", "ok": false, "error": "%s"}\n' "$1"; exit 1; }
mkdir -p "$ROOT"
[ -f "$REPO/rudra/genfill/setup.py" ] || fail "no rudra/genfill in $REPO"

UV="$ROOT/uv/uv"
if [ ! -x "$UV" ]; then
  case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) T=aarch64-apple-darwin ;;
    Darwin-x86_64) T=x86_64-apple-darwin ;;
    Linux-x86_64) T=x86_64-unknown-linux-gnu ;;
    Linux-aarch64) T=aarch64-unknown-linux-gnu ;;
    *) fail "no uv build for $(uname -s) $(uname -m)" ;;
  esac
  step "Downloading uv, the Python installer (about 20 MB)"
  tmp="$ROOT/uv-download"; rm -rf "$tmp"; mkdir -p "$tmp" "$ROOT/uv"
  curl -fL --retry 3 -o "$tmp/uv.tar.gz" "https://github.com/astral-sh/uv/releases/latest/download/uv-$T.tar.gz" >&2 \
    || fail "could not download uv (network?)"
  tar -xzf "$tmp/uv.tar.gz" -C "$tmp"
  found="$(find "$tmp" -type f -name uv | head -1)"
  [ -n "$found" ] || fail "the uv download has no uv in it"
  mv "$found" "$UV"; chmod +x "$UV"; rm -rf "$tmp"
fi
export UV_PYTHON_INSTALL_DIR="$ROOT/python" UV_CACHE_DIR="$ROOT/cache" UV_NO_PROGRESS=1

PY="$ROOT/venv/bin/python"
if [ ! -x "$PY" ]; then
  step "Installing a private Python 3.12 for the AI fill"
  "$UV" venv --seed --python 3.12 "$ROOT/venv" >&2 || fail "could not make the Python environment"
fi
printf '{"event": "bootstrap", "python": "%s"}\n' "$PY"

if [ "$(uname -s)" = "Darwin" ]; then IDX="${RUDRA_TORCH_INDEX:-}"; else IDX="${RUDRA_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"; fi
if ! "$PY" -c "import torch" 2>/dev/null; then
  step "Installing PyTorch (about 2.5 GB on Linux with CUDA, 0.3 GB on a Mac)"
  if [ -n "$IDX" ]; then "$UV" pip install --python "$PY" torch --index-url "$IDX" >&2 || fail "PyTorch did not install (see the log)"
  else "$UV" pip install --python "$PY" torch >&2 || fail "PyTorch did not install (see the log)"; fi
fi
step "Installing the AI fill's Python packages"
"$UV" pip install --python "$PY" "numpy>=1.26,<3" "opencv-python-headless>=4.9" "OpenEXR>=3.2" "opencolorio>=2.5" \
  "safetensors>=0.4" "diffusers>=0.35" "transformers>=4.45" "peft>=0.13" "accelerate>=1.0" sentencepiece ftfy \
  "huggingface-hub>=0.25" "PyYAML>=6" "tqdm>=4.66" >&2 || fail "the AI fill's packages did not install (see the log)"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
set +e
"$PY" -m rudra.genfill.setup --base "$BASE" --json --skip-packages
exit $?
