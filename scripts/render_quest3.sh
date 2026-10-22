#!/usr/bin/env bash
# A character variant's Quest 3 set, shown as it grows: the clips render on the GPU host in batches, and after
# every batch the 2064x2208 reel is composed from what exists and the review site is republished (same URL).
# An interrupted run resumes where it stopped (finished clips are fresh and skipped).
#
#   scripts/render_quest3.sh                          # configs/naomi.yaml, quest3, 4 clips a batch
#   CONFIG=configs/naomi.yaml PROFILE=draft BATCH=13 scripts/render_quest3.sh
#   DETACH=1 scripts/render_quest3.sh                 # push, then run the loop on the GPU host and return
#
# From the workstation each batch is its own `r3d remote` run, then compose here and `r3d publish`. With DETACH=1
# the loop itself moves to the host (ON_HOST=1 there: render, compose and `r3d site` locally, logs/quest3.log),
# so the workstation may sleep or shut down; the reel and its report then stay on the host (a pull keeps the
# workstation's own copies, see remote.WORKSTATION_ARTWORK).
set -euo pipefail
cd "$(dirname "$0")/.."
R3D=.venv/bin/r3d
export R3D_CONFIG="${CONFIG:-configs/naomi.yaml}"
PROFILE="${PROFILE:-quest3}"
BATCH="${BATCH:-4}"

if [[ -n "${DETACH:-}" ]]; then
  .venv/bin/python -c "from rybakina3d import remote; from rybakina3d.config import Config; remote.push(Config.load())"
  HOST=$(.venv/bin/python -c "from rybakina3d import remote; from rybakina3d.config import Config; print(remote.host(Config.load()))")
  ROOT=$(.venv/bin/python -c "from rybakina3d.config import Config; print(Config.load().remote.root)")
  # the same config on the host, project-relative (an absolute workstation path does not exist there)
  REL=$(.venv/bin/python -c "import os; from pathlib import Path; from rybakina3d.config import PROJECT_ROOT
print(Path(os.environ['R3D_CONFIG']).resolve().relative_to(PROJECT_ROOT))")
  # only the loop goes to the background (inside { }), so ssh returns as soon as it has started
  ssh -o BatchMode=yes "$HOST" "cd ~/$ROOT && mkdir -p logs && { ON_HOST=1 CONFIG='$REL' PROFILE='$PROFILE' \
    BATCH='$BATCH' setsid nohup bash scripts/render_quest3.sh > logs/quest3.log 2>&1 < /dev/null & } \
    && echo started on $HOST: ~/$ROOT/logs/quest3.log"
  exit 0
fi
if [[ -n "${ON_HOST:-}" ]]; then
  # a render already running here (a workstation loop that went away mid-batch) finishes first: never two at once
  while pgrep -f "r3d artwork motion-rende[r]" > /dev/null; do sleep 30; done
  RENDER=($R3D artwork motion-render) SHOW=(env -u R3D_CONFIG $R3D site)   # the site is the run's, not the variant's
else
  RENDER=($R3D remote artwork motion-render) SHOW=($R3D publish)
fi

# the moveset in render order: motion_render.FIRST, then curation order (no mapfile: macOS ships bash 3.2)
CLIPS=()
while IFS= read -r c; do CLIPS+=("$c"); done < <(.venv/bin/python - <<'EOF'
from rybakina3d.artwork.motion_render import FIRST
from rybakina3d.config import Config
from rybakina3d.util import read_json

kept = read_json(Config.load().run.curation)["kept"]
print("\n".join([c for c in FIRST if c in kept] + [c for c in kept if c not in FIRST]))
EOF
)
(( ${#CLIPS[@]} )) || { echo "no clips to render (is curation.json there?)" >&2; exit 1; }

for ((i = 0; i < ${#CLIPS[@]}; i += BATCH)); do
  batch=("${CLIPS[@]:i:BATCH}")
  args=()
  for c in "${batch[@]}"; do args+=(--clip "$c"); done
  echo "=== $PROFILE clips $((i + 1))-$((i + ${#batch[@]})) of ${#CLIPS[@]}: ${batch[*]}"
  "${RENDER[@]}" --profile "$PROFILE" "${args[@]}"
  $R3D artwork motion-compose --profile "$PROFILE"
  "${SHOW[@]}"
done
echo "done: ${#CLIPS[@]} clips at $PROFILE"
