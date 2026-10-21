#!/usr/bin/env bash
# Push the code to the GPU host (never .env, inputs or run outputs). Usage: scripts/sync.sh
# `r3d remote` sets R3D_HOST / R3D_REMOTE_ROOT from the config (remote:) or .env; run by hand, export R3D_HOST.
set -euo pipefail
HOST="${R3D_HOST:?set R3D_HOST=user@gpu-host}"
REMOTE="${R3D_REMOTE_ROOT:-rybakina3d}"
cd "$(dirname "$0")/.."
rsync -az --delete \
  --include='/src/***' --include='/configs/***' --include='/scripts/***' --include='/tests/***' --include='/web/***' \
  --include='/pyproject.toml' --include='/README.md' --include='/naomi_style.png' \
  --exclude='__pycache__' --exclude='*' \
  ./ "$HOST:$REMOTE/"
echo "synced -> $HOST:$REMOTE"
