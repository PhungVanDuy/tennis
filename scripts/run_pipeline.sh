#!/usr/bin/env bash
# End-to-end run for one athlete. Workstation stages call hosted APIs (Gemini, fal) or need the 720p previews;
# GPU stages run on the configured host through `r3d remote`, which syncs code and the run folder both ways.
#
#   scripts/run_pipeline.sh               # everything
#   FROM=motion scripts/run_pipeline.sh   # resume from a stage
set -euo pipefail
cd "$(dirname "$0")/.."
R3D=.venv/bin/r3d

# curate runs twice: first to fix labels and contact frames from the motion (qa's timing checks need the
# refined contact), then again with the qa gate; qa runs again after retarget to score the character's motion.
# artwork renders on the host and repaints/composes here (the fal key stays on the workstation).
STAGES=(ingest shots index select character clips track camera motion curate qa curate2 retarget qa2 signature artwork site)
FROM="${FROM:-ingest}"

run() {
  case "$1" in
    ingest|shots|index|select|camera|signature|site) $R3D "$1" ;;
    character) $R3D character refs && $R3D character stylize && $R3D character mesh ;;
    clips|track|motion|curate|qa|retarget) $R3D remote "$1" ;;
    curate2|qa2) $R3D remote "${1%2}" ;;
    artwork) $R3D artwork poses && $R3D remote artwork render && $R3D artwork paint && $R3D artwork compose \
      && $R3D remote artwork motion-render && $R3D artwork motion-paint && $R3D artwork motion-compose ;;
  esac
}

started=0
for s in "${STAGES[@]}"; do
  [[ "$s" == "$FROM" ]] && started=1
  (( started )) || continue
  echo "=== $s"
  run "$s"
done
echo "done — review with: $R3D serve"
