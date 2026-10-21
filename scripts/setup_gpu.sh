#!/usr/bin/env bash
# One-time setup of the GPU host: an isolated venv with GVHMR + tracking deps, a static ffmpeg,
# GVHMR checkpoints and the SMPL / SMPL-X body models. Everything lands under $ROOT; nothing
# system-wide is touched (the host is shared with other services).
#
#   ROOT=~/rybakina3d bash scripts/setup_gpu.sh
set -euo pipefail

ROOT="${ROOT:-$HOME/rybakina3d}"
cd "$ROOT"
mkdir -p bin weights body_models/smplx body_models/smpl

log() { printf '\n[setup %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# --- static ffmpeg (the host has none) ---------------------------------------------------------
if [[ ! -x bin/ffmpeg ]]; then
  log "ffmpeg"
  tmp=$(mktemp -d)
  curl -fsSL -o "$tmp/ff.tar.xz" \
    https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
  tar -xf "$tmp/ff.tar.xz" -C "$tmp"
  cp "$tmp"/ffmpeg-*/bin/{ffmpeg,ffprobe} bin/
  rm -rf "$tmp"
fi
export PATH="$ROOT/bin:$PATH"

# --- python env ----------------------------------------------------------------------------------
if [[ ! -d .venv ]]; then
  log "venv"
  uv venv --python 3.11 .venv
fi
export VIRTUAL_ENV="$ROOT/.venv"
log "torch (cu126)"
uv pip install "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu126
# chumpy (a legacy smplx dependency) does not build under PEP 517 isolation.
uv pip install pip setuptools wheel "numpy>=1.26,<2.3"
uv pip install --no-build-isolation "chumpy==0.70"
log "gvhmr + project deps"
# gvhmr[preproc] minus cython-bbox: the host has no C compiler, and we feed GVHMR our own tracks.
uv pip install "gvhmr==1.6.1" "pycolmap>=0.6" "ultralytics>=8.3" "lapx" "yacs" \
  "scenedetect[opencv]>=0.6.4" "typer>=0.12" "pydantic>=2.7" "pyyaml>=6" "rich>=13" "scipy>=1.11" \
  "librosa>=0.10"
uv pip install -e ".[gpu]" --no-deps

# --- checkpoints + body models -------------------------------------------------------------------
export GVHMR_CHECKPOINTS="$ROOT/weights" GVHMR_BODY_MODELS="$ROOT/body_models"
log "gvhmr checkpoints"
.venv/bin/gvhmr download demo || .venv/bin/gvhmr --help

log "body models (HF mirror)"
hf_get() {  # repo file dest
  [[ -s "$3" ]] || curl -fL --retry 3 -o "$3" "https://huggingface.co/$1/resolve/main/$2"
}
hf_get camenduru/SMPLer-X SMPLX_NEUTRAL.npz body_models/smplx/SMPLX_NEUTRAL.npz
hf_get camenduru/SMPLer-X SMPL_NEUTRAL.pkl body_models/smpl/SMPL_NEUTRAL.pkl

log "done"
.venv/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
