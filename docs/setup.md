# Setup and operations

This document covers setting up the two machines the pipeline uses: a workstation that calls the hosted model
APIs, and a Linux GPU host that runs tracking, GVHMR and Blender. It lists the secrets and private inputs a clone
lacks, explains what `r3d remote` moves between the machines, and shows how to run, resume and force stages. It
then walks through `configs/rybakina.yaml`, gives the time and cost figures the code and reports record, and
lists the failure modes this project hit. [architecture.md](architecture.md) describes the design.
[../README.md](../README.md) has the overview and the results.

## Machines at a glance

| | Workstation | GPU host |
|---|---|---|
| Runs | `ingest shots index select camera signature site serve`, `character *`, `artwork poses paint compose motion-paint motion-compose` | `clips track motion curate qa retarget`, `artwork render motion-render` (always through `r3d remote` from the workstation); the shared review site after `r3d publish` |
| Talks to | Gemini API, fal, Wikimedia Commons, the GPU host (ssh + rsync) | no model API. It downloads packages, weights and body models during setup, the YOLO weights on first use, and `cloudflared` on the first `r3d publish` |
| Holds secrets | yes, in `.env` | no: `.env` never leaves the workstation |
| Python | ≥ 3.11 (`requires-python` in `pyproject.toml`), 3.12 used | 3.11 (`scripts/setup_gpu.sh`) |
| Extras | `api`, `dev` | `gpu` (installed by `setup_gpu.sh`) |

`curate` and `qa` are pure numpy and would run anywhere. They run on the host because that is where
`scripts/run_pipeline.sh` sends them.

## Workstation

Requirements:

- [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer.
- `ffmpeg` and `ffprobe` on `PATH`. `ingest`, `site`, `motion-paint`, `motion-compose` and some tests use
  them. `media._bin` looks first in `<project>/bin/`, then on `PATH`.
- `ssh` and `rsync` for `r3d remote` and `scripts/sync.sh`. ssh runs with `-o BatchMode=yes`, so the host must
  accept key authentication without a prompt.
- Blender (optional). `character mesh` converts Tripo's FBX to GLB with a local Blender for the A/B mesh. It
  uses `R3D_BLENDER_LOCAL`, else `/Applications/Blender.app/Contents/MacOS/Blender`, else `blender` on `PATH`. If
  this fails, the Tripo generator records an error and the Meshy rig is still used. Tests and all other
  workstation stages need no Blender.
- Fonts (optional). `artwork compose` uses macOS system fonts (Baskerville, Didot, Snell Roundhand from
  `/System/Library/Fonts/Supplemental`), then DejaVu (`/usr/share/fonts/truetype/dejavu`), then Pillow's default.
- YuNet weights (only for `r3d character face`, an optional stage for art-team variants that the Naomi character
  does not use; see [character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant)). The
  stage reads OpenCV's YuNet face detector from `weights/yunet/face_detection_yunet_2023mar.onnx` under the
  project root (`YUNET` in `character/face.py`) and stops with the download URL when the file is missing. It
  comes from opencv_zoo. `weights/` is gitignored, so a clone lacks it. The stage also runs the local Blender,
  found as for `character mesh`.

  ```bash
  mkdir -p weights/yunet
  curl -fL -o weights/yunet/face_detection_yunet_2023mar.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```

Install:

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[api,dev]"
.venv/bin/r3d --help
.venv/bin/pytest -q && .venv/bin/ruff check src tests     # 170 tests, about 25 s (see the note below)
```

| Extra | Contents | Where |
|---|---|---|
| (base) | numpy, scipy, pydantic, pyyaml, typer, rich, opencv-python-headless | both |
| `api` | google-genai, fal-client, requests, pymupdf, pillow, scenedetect, librosa, python-dotenv | workstation |
| `gpu` | gvhmr[preproc] 1.6.1, ultralytics, scenedetect, librosa | GPU host (`setup_gpu.sh` installs it with pinned torch and `--no-deps`) |
| `dev` | pytest, ruff | workstation |

The tests run on synthetic data. They need the `api` extra (Pillow, librosa), but no GPU, Blender or network.
Tests that encode video are skipped when `ffmpeg` is missing. They pass on a bare clone without any of the private
inputs: where the style reference is missing, `paint.paper_colour()` uses its measured value (`PAPER_FALLBACK`).

## Secrets: `.env`

`util.secret` reads the first non-empty value from the environment, then from `<project>/.env` (the project
root, not the current directory). Keys are never logged. `.env` is gitignored, and `scripts/sync.sh` never sends
it.

| Name | Used by | Notes |
|---|---|---|
| `GEMINI_API_KEY` | `index`, `select`, `character stylize`, `character mesh` (side-view facing check) | |
| `FAL_KEY` or `FAL_API` | `character mesh`, `artwork paint`, `artwork motion-paint` | `fal.client()` copies either into `FAL_KEY` |
| `R3D_HOST` | `r3d remote`, `scripts/sync.sh` | `user@gpu-host`. Used when `remote.host` in the config is empty (the committed default). `sync.sh` run by hand reads the environment only, so export it |

Optional environment variables:

| Name | Effect |
|---|---|
| `R3D_CONFIG` | config file instead of `configs/rybakina.yaml`. `r3d remote` forwards it to the host as a project-relative path |
| `R3D_BLENDER` | Blender binary for `retarget`, `artwork render` and `motion-render`. `r3d remote` sets it to `remote.blender` on the host |
| `R3D_BLENDER_LOCAL` | workstation Blender for `character mesh` |
| `R3D_REMOTE_ROOT` | target folder for `scripts/sync.sh` (default `rybakina3d`, relative to the remote `$HOME`); `r3d remote` sets it from `remote.root` |

## Private inputs

The repository holds none of the inputs below; `.gitignore` excludes them. Put them in the project root under
these exact names, which come from the code and config.

| File | Referenced by | Used by | What it is |
|---|---|---|---|
| `Elena-Rybakina-Shots.mp4` | `source.video` in `configs/rybakina.yaml` | `ingest`, `select` (workstation); `clips` (GPU host) | the broadcast shot compilation (30.8 min, 1920×1080, 25 fps, with audio). `source.youtube` records its origin; the pipeline does not download it. It must be on **both** machines, see [GPU host](#gpu-host) |
| `Intelligent Internet Mail - Task - 3D tennis.pdf` | `CLIENT_PDF` in `character/refs.py` | `character refs` | the client brief as a PDF. `refs` extracts five embedded images by page and image index (`PDF_REFS`): two style references and three identity photos. Another PDF needs new `PDF_REFS` entries |
| `Screenshot 2026-10-21 at 14.05.59.png` | `SHEET_SCREENSHOT` in `character/refs.py` | `character refs` | a screenshot with a partial view of a target-style character sheet, cropped by `SHEET_CROP` into a style reference for `stylize` |
| `naomi_style.png` | `STYLE_REFERENCE` in `artwork/layout.py` | `artwork paint`, `compose`, `motion-render`, `motion-paint`, `motion-compose`; 5 tests | the client's style reference character sheet. The sheet layout follows it, `paint` crops one figure of each kind from it (`STYLE_CROPS`), and `paint.paper_colour()` samples the paper colour from it. `scripts/sync.sh` copies it to the host because `motion-render` composites on that paper |
| `assets/client/` | nothing | none | images extracted from the deck for inspection. No stage reads them |
| `artexamplesfortennisnaomiosaka/` | `character.refs_dir` in `configs/naomi.yaml`; file names in `ART_REFS` (`character/art.py`) | `character refs` with `configs/naomi.yaml`, which copies them into `variants/naomi/character/refs/` for `stylize`, `mesh` and `face` | the art team's renders of the Naomi character: the A-pose front, the art-direction sheet and the head renders ([character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant)). Only the Naomi variant needs it |

`character refs` also downloads four freely licensed photos from Wikimedia Commons (`COMMONS` in
`character/refs.py`) and records each file's Commons page URL and licence in `character/refs/refs.json`.

## GPU host

### Requirements

- Linux x86-64, an NVIDIA GPU, and a driver that runs CUDA 12.6 wheels (torch 2.7.1 + cu126). This run used one
  A100 40 GB; Blender rendered with OptiX.
- `uv`, `curl`, `tar` with xz support, `rsync`, and an ssh login with key authentication.
- **Blender 5.2** (this run: 5.2.2 LTS) at `remote.blender` (`/usr/local/bin/blender` in the config).
  `setup_gpu.sh` does not install it. `retarget`, `artwork render` and `artwork motion-render` need it.
- No C compiler and no system ffmpeg are needed; see below.

### First-time setup

```bash
# on the workstation
export R3D_HOST=user@gpu-host
scripts/sync.sh                                           # code -> ~/rybakina3d/ on the host
rsync -P Elena-Rybakina-Shots.mp4 "$R3D_HOST:rybakina3d/"  # the source video, once (nothing else sends it)

# on the host (interactive shell, so uv is on PATH)
cd ~/rybakina3d && bash scripts/setup_gpu.sh              # ROOT=… overrides, but r3d remote expects ~/<remote.root>
```

`scripts/setup_gpu.sh` installs everything under `$ROOT` (default `~/rybakina3d`) and leaves the system alone,
because the host is shared with other services:

| Step | What | From |
|---|---|---|
| ffmpeg | static `ffmpeg` + `ffprobe` (n8.1, GPL) in `$ROOT/bin/`, because the host has no ffmpeg | BtbN FFmpeg-Builds release on GitHub |
| venv | `$ROOT/.venv`, Python 3.11 | uv |
| torch | `torch==2.7.1`, `torchvision==0.22.1` | PyTorch cu126 index |
| chumpy | `chumpy==0.70` with `--no-build-isolation`, after `numpy>=1.26,<2.3`, `pip`, `setuptools`, `wheel` | PyPI. chumpy, a legacy smplx dependency, does not build under PEP 517 isolation |
| GVHMR + deps | `gvhmr==1.6.1`, `pycolmap`, `ultralytics`, `lapx`, `yacs`, `scenedetect`, `typer`, `pydantic`, `pyyaml`, `rich`, `scipy`, `librosa`; then `-e ".[gpu]" --no-deps` | PyPI. This is `gvhmr[preproc]` without `cython-bbox`: there is no C compiler, and GVHMR gets our own tracks |
| GVHMR checkpoints | `gvhmr download demo` into `$ROOT/weights/` (`GVHMR_CHECKPOINTS`) | GVHMR's own download command |
| body models | `body_models/smplx/SMPLX_NEUTRAL.npz`, `body_models/smpl/SMPL_NEUTRAL.pkl` (`GVHMR_BODY_MODELS`) | a Hugging Face mirror (`camenduru/SMPLer-X`). SMPL and SMPL-X are distributed under their authors' licence |
| check | prints the torch version and `torch.cuda.is_available()` | |

The YOLO detector (`track.detector`, `yolo11x.pt`) is loaded from `$ROOT/weights/yolo11x.pt` if it exists.
Otherwise Ultralytics fetches it by name on first use.

The host layout after setup and a run:

```
~/rybakina3d/
  src/ configs/ scripts/ tests/ web/ pyproject.toml README.md naomi_style.png   # from scripts/sync.sh
  Elena-Rybakina-Shots.mp4                                                     # copied by hand
  .venv/ bin/ weights/ body_models/                                            # from setup_gpu.sh
  runs/rybakina/                                                               # the host's half of the run folder
  logs/  bin/cloudflared                                                       # from r3d publish, if used
```

### How `r3d remote <stage>` works

`r3d remote <stage> [args]` (`remote.run_stage`) runs four steps. If any step fails, the command stops, so a
failed stage pulls nothing.

1. **Push the code.** `scripts/sync.sh` runs `rsync -az --delete` of `src/`, `configs/`, `scripts/`, `tests/`,
   `web/`, `pyproject.toml`, `README.md` and `naomi_style.png` to `~/<remote.root>/`. Everything else is excluded, and
   excluded files on the host (`.venv`, `bin`, `weights`, `runs`, the video) are not deleted.
2. **Push the run inputs.** It rsyncs `runs/rybakina/` to the host without `PUSH_EXCLUDES` (`source/proxy.mp4`,
   `source/audio16k.wav`, `anim/`, `site/`, partial files, and all workstation-made artwork: repaints, sheets,
   `artwork.json`, painted motion, reels, `motion.json`, `motion_<profile>.json`, and the clip videos of
   `artwork/motion/render*/`, which only the host makes). Of `clips/`, only the workstation's calibrations
   `clips/*/camera.npz` go up (`PUSH_FILTERS`).
3. **Run.** It runs `ssh -o BatchMode=yes $R3D_HOST "cd ~/rybakina3d && export PATH=$HOME/rybakina3d/bin:$PATH
   GVHMR_CHECKPOINTS=… GVHMR_BODY_MODELS=… R3D_BLENDER=<remote.blender> PYTHONUNBUFFERED=1 [R3D_CONFIG=…] &&
   .venv/bin/r3d <stage> [args]"`. Extra arguments such as `--force` and `--clip <id>` pass through, and a
   two-word stage works: `r3d remote artwork render`.
4. **Pull the results.** It rsyncs the host's run folder back without `PULL_EXCLUDES`.

| | Travels |
|---|---|
| Up (workstation → host) | code; `source/probe.json`, `shots.json`, `index/`, `selection.json`, `character/`, `curation.json`, `qa_summary.json`, `signature.json`, `artwork/poses.json`, `artwork/renders/`, `artwork/motion/render/`; `clips/*/camera.npz` |
| Down (host → workstation) | everything else the host made: `clips/<id>/clip.json`, `preview.mp4`, `track.*`, `motion.npz`, `overlay.mp4`, `qa.json`; `curation.json`, `qa_summary.json`; `anim/` (GLB, FBX, reports, `joints/`, `checks/`); `artwork/renders/`; `artwork/motion/render/` |
| Never down | `clips/*/clip.mp4` (model-rate copies), `clips/*/gvhmr/` (GVHMR caches), `source/`, `anim/*.blend*`, `artwork/motion/frames/` (PNG frames), partial files, and every workstation-made file (`clips/*/camera.npz` / `camera.jpg`, `WORKSTATION_ARTWORK` in `remote.py`) |
| Never anywhere | `.env` and API keys (fal and Gemini stages run only on the workstation); the client deck, the screenshot, `assets/` |

The pull excludes cover the files the workstation makes inside folders the host also writes: the camera
calibrations and the workstation's artwork. A pull therefore never puts an older host copy over them. The other
workstation files (`index/`, `selection.json`, `character/`, `signature.json`) are not excluded from the pull;
they are pushed just before the stage runs, so the copies that come back are the same, unless you change one of
them on the workstation while a remote stage is still running. Neither direction uses `--delete` for run data,
so a file deleted on one machine stays on the other.

`r3d remote` forwards `R3D_CONFIG` but not `--config`. To use another config on both machines, set
`R3D_CONFIG` (a path inside the project), not `-c` ([A character variant](#a-character-variant)).

The exclude patterns have no leading `/`, so rsync matches them at any depth: a variant's
`variants/<name>/anim/`, `artwork/motion/frames/` and workstation-made artwork travel by the same rules as the
Rybakina ones. The artwork patterns name the outputs by suffix (`artwork/*_character_sheet*`, `artwork/*_moveset_*`,
`artwork/motion/*_moveset_*`), so they hold for any `<stem>`.

## Running stages

### The whole pipeline

```bash
scripts/run_pipeline.sh                 # every stage, in order
FROM=retarget scripts/run_pipeline.sh   # resume from a stage
```

`FROM` takes one of the names below. An unknown name runs nothing and still prints `done`.

| Name in `run_pipeline.sh` | Runs |
|---|---|
| `ingest`, `shots`, `index`, `select`, `camera`, `signature`, `site` | `r3d <name>` on the workstation |
| `character` | `r3d character refs`, `stylize`, `mesh` |
| `clips`, `track`, `motion`, `curate`, `qa`, `retarget` | `r3d remote <name>` |
| `curate2`, `qa2` | `r3d remote curate`, `r3d remote qa` (second passes: with the QA gate; on the retargeted character) |
| `artwork` | `r3d artwork poses`, `r3d remote artwork render`, `r3d artwork paint`, `r3d artwork compose`, then the moving artwork at the `web` profile: `r3d remote artwork motion-render`, `r3d artwork motion-paint`, `r3d artwork motion-compose` |

To run only the moving artwork:

```bash
.venv/bin/r3d remote artwork motion-render
.venv/bin/r3d artwork motion-paint
.venv/bin/r3d artwork motion-compose
.venv/bin/r3d site
```

### Single stages, re-runs and `--force`

Every stage is `.venv/bin/r3d <stage>`, or `r3d remote <stage>` for the GPU ones. Each stage either skips work
whose outputs exist (and recomputes it with `--force`) or rewrites its outputs on every run:

| Stage | Default on re-run | `--force` |
|---|---|---|
| `ingest` | re-probes; keeps an existing `audio16k.wav` and `proxy.mp4` | no flag (delete the files to rebuild them) |
| `shots`, `curate`, `qa`, `signature`, `retarget`, `site`, `character refs`, `artwork poses`, `artwork compose`, `artwork motion-compose` | always rewritten. `curate` is idempotent: it re-derives contact from the contact `select` chose | no flag |
| `index` | reuses `index/chunk_*.json`, and reuses the uploaded proxy (`index/upload.json`) while the Gemini File API still holds it | deletes the chunk caches and labels again |
| `select` | skipped if `selection.json` exists | recomputes. The hitter-box cache `index/locate.json` is kept even with `--force` |
| `clips`, `track`, `camera`, `motion` | skip a clip whose output exists (`clip.json` + `clip.mp4`, `track.npz`, `camera.npz`, `motion.npz`) | recomputes. `camera` first deletes the old calibration so a stale one cannot survive; `motion` deletes `gvhmr/` |
| `character stylize` | keeps `hero.png` and `turnaround.png`; always re-cuts the views | regenerates both |
| `character mesh` | keeps each generator that already has a rig | regenerates both |
| `artwork render`, `artwork motion-render` | re-render only what changed: per figure / video fingerprint of the GLB, pose, camera, size, samples and a version constant (`RENDER_VERSION`, `MOTION_VERSION`, plus the whole profile and `PROFILE_VERSION` for `--profile quest3` / `draft`). When only the encoding changed and the frames are still on the host under the same render key, `motion-render` re-encodes them without Blender | renders everything |
| `artwork paint`, `artwork motion-paint` | cached by fingerprint (render + matte bytes, model, prompt). A changed render or prompt repaints only that figure or clip, and a failed call is retried on the next run. `motion-paint` also re-applies its acceptance thresholds to cached outputs without a call | repaints everything |

`--clip <id>` (repeatable) limits `clips`, `track`, `camera`, `motion` and `artwork motion-render` to those clips.
For `retarget` it limits only the check renders, since the moveset always holds every kept clip. Clip ids are
`<stroke>_<MM>m<SS>s`, from the contact time in the source, e.g. `serve_16m13s`.

```bash
.venv/bin/r3d camera --clip serve_16m13s --force
.venv/bin/r3d remote motion --clip serve_16m13s --force
```

### A character variant

A variant is a second character on the same captured motion, built from its own config (`configs/naomi.yaml`,
which `extends` the Rybakina config; see [Other configs](#other-configs-extends-and-character-variants)). Its
`character/`, `anim/` and `artwork/` live under `runs/rybakina/variants/<name>/` with `<name>_*` file names, while
the footage, clips, motion, curation and QA at the run root are shared. Pick the config per machine:

- **Workstation stages:** `-c configs/naomi.yaml` (or `R3D_CONFIG=configs/naomi.yaml`).
- **`r3d remote`:** `R3D_CONFIG=configs/naomi.yaml` only. `r3d remote` forwards `R3D_CONFIG` to the host as a
  project-relative path and does not forward `-c`, so with `-c` the host would run the default config.
- **`r3d site`:** the default config. The site lists every variant in the run folder's `variants/` on its own
  ([review-site.md](review-site.md#character-variants)).

```bash
# workstation: the character (character.md, section 5)
.venv/bin/r3d character refs -c configs/naomi.yaml
.venv/bin/r3d character stylize -c configs/naomi.yaml
.venv/bin/r3d character mesh -c configs/naomi.yaml
# GPU host: the variant's moveset on the shared motion
R3D_CONFIG=configs/naomi.yaml .venv/bin/r3d remote retarget
# the Quest 3 set (docs/artwork.md, section 4): render, compose and publish in batches
scripts/render_quest3.sh
.venv/bin/r3d site
```

Do not re-run the shared stages with a variant's config. `qa` in particular writes the shared
`clips/<id>/qa.json` and `qa_summary.json` at the run root, and with a variant's config its character pass would
read the variant's joint tracks.

`scripts/render_quest3.sh` takes three environment variables:

| Variable | Default | Effect |
|---|---|---|
| `CONFIG` | `configs/naomi.yaml` | exported as `R3D_CONFIG` for every command it runs |
| `PROFILE` | `quest3` | the render profile (`quest3`, `draft`; `artwork/layout.py` `PROFILES`) |
| `BATCH` | `4` | clips per batch |

It orders the moveset like `motion-render` (`motion_render.FIRST`, then curation order). Per batch it runs
`r3d remote artwork motion-render --profile $PROFILE --clip …`, then `r3d artwork motion-compose --profile
$PROFILE` on the workstation, then `r3d publish`, so the shared site shows the reel growing at the same URL. An
interrupted run resumes: finished clips match their fingerprint and are skipped. For a quick look first:
`PROFILE=draft BATCH=13 scripts/render_quest3.sh`.

With `DETACH=1` the script pushes code and run, starts itself on the GPU host (`ON_HOST=1`, detached with
`setsid nohup`, log in `~/<remote.root>/logs/quest3.log`) and returns. There every batch renders, composes and
rebuilds the site locally (`r3d site` under the default config), so the workstation may sleep or shut down
during the hours of a `quest3` run. The host loop first waits for any `r3d artwork motion-render` already
running there, so it never renders the same clips twice. Its reel and `motion_<profile>.json` stay on the host:
pulls never overwrite workstation artwork (`remote.WORKSTATION_ARTWORK`).

### Review

```bash
.venv/bin/r3d site      # rebuild site/ from the run folder (safe while stages are still running)
.venv/bin/r3d serve     # http://127.0.0.1:8765 (site.port); builds the site first if needed
.venv/bin/r3d publish   # serve it from the GPU host behind a public Cloudflare quick-tunnel URL
```

[review-site.md](review-site.md) describes the pages and what `r3d publish` does. Anyone with the published URL
can open the site, which shows client reference material and broadcast previews.

## Configuration: `configs/rybakina.yaml`

Paths are relative to the project root. `config.py` holds the typed schema (`Config`) and its defaults.

| Section | Keys (current value) | What they control |
|---|---|---|
| `run_dir` | `runs/rybakina` | every output; the host uses the same project-relative path under `remote.root` |
| `athlete` | `name`, `handedness: right`, `backhand: two_handed`, `height_m: 1.84` | `name` goes into the Gemini prompts. `handedness` goes into the hitter-box prompt and picks the hitting wrist in `curate`, `qa`, `signature` and `artwork poses`. `backhand` shapes the hitter-box prompt and the repaint grip note. `height_m` is shown on the site but not read by any stage (the rig uses `character.height_m`) |
| `source` | `video`, `youtube` | the compilation's path (see [Private inputs](#private-inputs)); `youtube` is only displayed |
| `remote` | `host: ""`, `root: rybakina3d`, `blender: /usr/local/bin/blender` | empty `host` means `R3D_HOST` from the environment or `.env`; `root` is relative to the remote `$HOME`; `blender` is the host's Blender |
| `models` | `index`, `locate`: `gemini-3.8-flash`; `image_hero`, `image_views`: `gemini-3-pro-image`; `mesh`: Meshy 7.1; `mesh_alt`: Tripo H3.1; `rig`: Meshy rigging; `image_paint`: `openai/gpt-image-2.5/sunburst/edit`; `video_paint`: `fal-ai/bernini-r/reference-edit-video` | model ids per call site. `video_paint` must stay Bernini-R: `motion-paint` builds its request for that model and raises on any other value |
| `shots` | `adaptive_threshold: 3.0`, `min_shot_s: 0.6` | PySceneDetect `AdaptiveDetector` sensitivity and minimum shot length |
| `index` | `proxy_height: 360`, `chunk_s: 240`, `fps: 2.0` | upload proxy height; length of each Gemini window; frames per second Gemini samples |
| `select` | `oversample: 2.0`; `strokes` (serve 6, forehand 6, backhand 6, slice 2, volley 2, overhead 1, celebration 2); `window_s` (serve 2.2 s before / 1.0 s after contact, celebration 0.6 / 2.4, default 1.3 / 1.0); `allowed_views: [main_broadcast, behind_low, side]`; `allow_slow_motion: false`; `min_quality: 3`; `onset_search_s: 0.5` | which shots qualify, how many clips per stroke (quota × `oversample` are cut), the clip window and the audio search radius. Fixed in code: an audio onset counts when its local z-score is above 2.5; a window covering less than 60 % of the wanted length scores 0; at most one clip per shot and stroke; contacts at least 1 s apart |
| `clips` | `model_fps: 30`, `time_mode: retime`, `preview_height: 720` | `retime` keeps every source frame and stamps it at GVHMR's 30 fps (no duplicated frames); the preview keeps the source rate |
| `track` | `detector: yolo11x.pt`, `conf: 0.25`, `min_coverage: 0.7` | YOLO weights and confidence. Coverage below `min_coverage` warns in `track` and `qa`; below 0.5 (`qa.COVERAGE_FAIL`) `qa` fails the clip |
| `motion` | `static_motion_px: 1.5`, `default_hfov_deg: 32`, `flip_test: true` | for clips without a calibration: background motion below `static_motion_px` counts as a static camera, and the lens comes from `VIEW_HFOV` in `motion/recover.py` (main_broadcast 38°, behind_low 42°, side 42°). `default_hfov_deg` applies only to views missing from that table, which `select` never passes. `flip_test` is GVHMR's test-time flip |
| `character` | `height_m: 1.84`, `target_tris: 24000`, `views: [front, left, back, right]` | rig height for Meshy; `target_tris / 2` quads for Meshy's `target_polycount` and Tripo's `face_limit`; turnaround views |
| `curate` | `keep_all: true` | `true`: every clip goes into the moveset and failed stroke checks are only flagged. `false` (the code default): the best `select.strokes[type]` label-valid clips per stroke are kept |
| `retarget` | `ground_from_contacts: true`, `smooth_window: 5` | re-ground each clip from GVHMR's foot contacts (so `qa`'s ground rules only warn); Savitzky–Golay rotation smoothing window in frames (0 = off) |
| `site` | `port: 8765` | `r3d serve` and `r3d publish` port (`--port` overrides) |

Many thresholds are module constants rather than config keys, each next to the measurement that motivated it.
Examples: `camera.MIN_OK_FRACTION = 0.90`, `curate.OVERHEAD_MIN_RISE_M = 0.25`, `qa.SKATE_MEAN_CM = (1.5, 3.0)`,
`paint.ACCEPT_IOU = 0.85`, `motion_paint.ACCEPT_MEDIAN = 0.85` / `ACCEPT_P10 = 0.75`.
[motion-pipeline.md](motion-pipeline.md) and [artwork.md](artwork.md) cover them.

`motion-paint` accepts at most 121 frames per clip (`VIDEO_MAX_FRAMES`, 4.84 s at 25 fps). A `select.window_s`
longer than that makes the stage raise for those clips.

### Other configs: `extends` and character variants

A config can start from another with `extends: <file>`, a path relative to the config itself. `Config.load`
reads the base first and deep-merges the file over it: mappings merge key by key, any other value (a list, a
number, a string) replaces the base's. `configs/naomi.yaml` is `extends: rybakina.yaml` plus:

| Key | Value | What it does |
|---|---|---|
| `character.variant` | `naomi` | makes this a variant: `RunLayout.out` is `runs/rybakina/variants/naomi/` and `RunLayout.stem` is `naomi` (file names `naomi_*`) |
| `character.name` | `Naomi Osaka` | `Config.display_name`, the name on the artwork (default: `athlete.name`) |
| `character.height_m` | `1.80` | the rig height |
| `character.refs_dir` | `artexamplesfortennisnaomiosaka` | the art team's renders (project-relative); `character refs` and `stylize` then take the art-team path (`character/art.py`) |
| `retarget.smooth_limb_weights` | `20` | Laplacian passes over the limb weights before retargeting (`blender/retarget.py` `smooth_limb_weights`); `0` (the default) leaves the auto-rig's weights |
| `retarget.weld_m` | `1.0e-5` | skinned-mesh vertices closer than this (metres) are merged before retargeting (`blender/retarget.py` `weld_meshes`); the default `0` keeps the mesh as shipped (Rybakina) |
| `retarget.leg_weights` | `band` | the leg skin weights, replaced after the limb smoothing (`blender/retarget.py` `LEG_WEIGHTS`): `band` rebuilds each knee's thigh/shin split centred on the knee joint; `meshy` (the default) keeps the auto-rig's ([character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant) says why Naomi needs `band`) |
| `retarget.knee_band_cm` | `5.0` | the band's half-width either side of the knee plane (the code default too) |
| `character.mesh_primary` | `hi3d` | the generator whose rig becomes `character/rigged.glb` (`meshy`, `tripo`, `hunyuan` or `hi3d`) |
| `character.racket_colors` | `frame`, `grip`, `strings` as linear RGB | the racket's colours, passed to `blender/racket.py` through the retarget job |
| `models.mesh` | `""` | no Meshy generator for this variant |
| `models.mesh_alt` | `hitem3d/hi3d/v3.0/multi-view-to-3d` | the A/B generator (`hunyuan-3d` and `tripo3d` endpoints are recognised too), rigged by `fal-ai/meshy/rigging` |

Everything else (athlete, source, remote, the other models, thresholds, `run_dir`) comes from `configs/rybakina.yaml`, so
both configs share one run folder and one GPU-host layout.

## Time and cost

The code and the run reports record only some of these figures. Gemini and image-model prices are not recorded
anywhere in the repository.

| Step | Figure | Source |
|---|---|---|
| `index` | 8 Gemini requests (240 s windows over 30.8 min); the 360p proxy is uploaded once and reused while the File API holds it (files expire after about 48 h) | `index/chunk_*.json`, `gemini.upload_video` |
| `select` | one Gemini image request per candidate contact frame, 8 in parallel, cached in `index/locate.json` | `selection.py` |
| `retarget` | 17.3 s for the Blender build of 39 clips (plus the re-import verification) | `anim/retarget.json` `seconds` |
| `artwork render` | 13.7 s Cycles for the hero figure at 128 samples on the A100 | `artwork/renders/blender.json` |
| `artwork paint` | 33–66 s per GPT Image call (median 46 s): 11 calls (13 samples, since the strict retry asks for 3) for 10 figures, 4 in parallel | `artwork/painted/paint.json` |
| `artwork motion-render` | 2,681 frames (39 clips + 150-frame turntable): 3,554 s at 48 samples on one A100. At `layout.MOTION.samples = 64` the same frames take about 65 min (about 1.3 s a frame), so `motion_render.SAMPLES` caps it at 48. After OIDN, a 48-sample frame differs from a 64-sample one by 0.7/255 on average | `artwork/motion/render/renders.json`, `motion_render.py` |
| `artwork motion-render --profile quest3` | 2064 × 2208 at 60 fps, 16 samples. Measured at 32 samples, before the profile was set to 16: 4.2 s Cycles + 0.64 s matte a frame on one A100 (Rybakina's `backhand_14m19s`, 139 frames). The moveset is 6,065 frames at 60 fps, about 8.2 h at that rate. A 2.3 s clip encodes to about 1.35 MB | `artwork/motion/render_quest3/renders.json`, `layout.PROFILES` |
| `artwork motion-render --profile draft` | 516 × 552 at 60 fps, 16 samples: 0.41 s Cycles + 0.05 s matte a frame on the same clip, about 47 min for the moveset. Naomi's full draft on the earlier Hunyuan character (39 clips, 6,065 frames) took 4,240 s of Blender time (about 71 min): 0.53 s + 0.09 s a frame and about 12 s of setup per clip | `artwork/motion/render_draft/renders.json`, `archive/naomi_hunyuan/artwork/motion/render_draft/renders.json` |
| `artwork motion-paint` | Bernini-R takes about 10 min a clip, 8 clips in flight (`WORKERS`). List price `VIDEO_PRICE_USD_S = 0.08` per output second: the 39 clips go out as 2,615 frames (each padded to 4k+1), 104.6 s, about USD 8.4 for the video model alone. The frame-0 repaints and BiRefNet matting cost extra, and a clip whose anchor fails costs no video call. Per-clip `cost_usd` and the total go to `artwork/motion/painted/paint.json` | `motion_paint.py` |

Disk for the Quest 3 set:

- **GPU host.** A `quest3` frame is about 3.2 MB of PNGs (RGBA + matte; 438.5 MB for the 139-frame backhand at 32 samples),
  about 19 GB for the moveset. `quest3` sets `keep_frames=False`, so each clip's PNGs are deleted after its video
  is encoded, when the run's Blender job has finished. One run over the whole moveset therefore peaks near 19 GB;
  `scripts/render_quest3.sh`'s batches of 4 clips peak at 4 clips' frames (about 0.6 GB per 194-frame serve). The
  cost is that a changed `quest3` encoding renders again.
  `draft` keeps its frames, about 0.26 MB a frame (1.6 GB for the Hunyuan draft's 6,065), and so does `web`. All of them sit
  in `artwork/motion/frames/` (`frames/<profile>/` for `quest3` and `draft`), which is never pulled.
- **Workstation.** Only the videos come back: about 1.35 MB per 2.3 s `quest3` clip, 0.12 to 0.39 MB per draft
  clip (9.7 MB for the Hunyuan draft's 39), plus the reel (3.7 MB for that draft) and the 336 × 360 site copies that
  `motion-compose` writes.

## Troubleshooting

Failure modes that came up in this project or that the code guards against, with the fix.

| Symptom | Cause and fix |
|---|---|
| `ffmpeg not found (install ffmpeg or put a static build in bin/)` | The GPU host has no ffmpeg. `setup_gpu.sh` puts a static build in `~/rybakina3d/bin/`, which `media._bin` prefers and `r3d remote` adds to `PATH`. On the workstation, install ffmpeg |
| `missing secret: one of ('FAL_KEY', 'FAL_API') (env or .env)` (or `GEMINI_API_KEY`, `R3D_HOST`) | `.env` must be in the project root; `util.secret` does not look in the current directory |
| `scripts/sync.sh`: `R3D_HOST: set R3D_HOST=user@gpu-host` | run by hand, the script reads only the environment: `export R3D_HOST=…`. `r3d remote` sets it for you |
| `r3d remote` fails at ssh with a permission error | ssh runs with `BatchMode=yes` and never prompts; set up key authentication for `R3D_HOST` |
| `clips` on the host cannot open the source video | nothing syncs it; copy `Elena-Rybakina-Shots.mp4` to `~/rybakina3d/` once |
| chumpy fails to build, or `cython-bbox` needs a compiler, when installing GVHMR by hand | install chumpy with `--no-build-isolation` and leave out `cython-bbox` (`gvhmr` without `[preproc]` + the listed deps), as `setup_gpu.sh` does |
| `retarget` / `artwork render` cannot start Blender on the host | `r3d remote` always sets `R3D_BLENDER=remote.blender`; Blender 5.2 must exist at that path |
| A camera calibration or repaint seems to "come back" or be overwritten after a remote stage | It cannot. Pulls exclude the calibrations and the workstation's artwork (`clips/*/camera.*`, `WORKSTATION_ARTWORK` in `remote.py`), so an older host copy never replaces them. Without `--delete`, a file removed on one machine stays on the other ([How `r3d remote` works](#how-r3d-remote-stage-works)) |
| Old clip folders from an earlier selection linger | `RunLayout.clip_ids()` only uses folders listed in the current `selection.json`; the rest are ignored |
| `artwork paint` logs `moderated by the provider` (`content_policy_violation`) | fal's output moderation rejects some samples of a figure in a short dress lunging. `paint` moves on to its strict retry (`RETRY_CANDIDATES = 3` samples), and `motion-paint` tries each call up to `MODERATION_TRIES = 3` times. A figure whose every attempt fails is recorded with an `error` and retried on the next run |
| `motion-paint` logs `… is unavailable … clips not started, run again later` | fal answered `api access is currently unavailable` (`CLOSED`): the endpoint is closed to API callers, as happened to Kling v4 edit on 2026-10-22. The stage cancels the clips not yet started; run it again later |
| `compose` ignores a repaint | its fingerprint no longer matches the current render (the render changed); run `r3d artwork paint` again |
| `artwork paint`: `no renders in …: run r3d remote artwork render first` | the renders come from the host; run `r3d remote artwork render` |
| `camera` writes nothing for a clip | Not an error. A calibration is written only when ≥ 90 % of frames hold (`MIN_OK_FRACTION`) with a mean reprojection ≤ 1.5 px (`MAX_CLIP_REPROJ_PX`), and only `main_broadcast` / `behind_low` views are tried. `motion` then falls back to the view's typical lens |
| Index times look off by a window's start | Gemini sometimes answers relative to the window. `labeler._fix_time` maps times back, and events still outside their shot ±0.6 s are dropped (68 in this run) |
| Gemini image generation refuses the hero | the stylize prompts never name the athlete, because the image filters key on names; the photos carry the identity. Keep it that way when editing prompts |
| Videos do not seek in the review site | use `r3d serve`, which answers HTTP Range requests with 206; `python -m http.server` does not |
| The page stays on its loader | the page imports three.js 0.169.0 from `cdn.jsdelivr.net` through an import map, so it needs network access; offline, the whole module graph fails to load |
| A copied `site/` folder shows nothing | `site/media/` holds relative symlinks into the run folder; copy with `cp -RL` or `rsync -L` to share it |
| `r3d remote … -c configs/naomi.yaml` runs the stage for the Rybakina character | `-c` is local only: the host loads its default config. Use `R3D_CONFIG=configs/naomi.yaml .venv/bin/r3d remote …` |
| `character face`: `no …/weights/yunet/face_detection_yunet_2023mar.onnx: download … there` | the YuNet weights are not in the repository; download them as in [Workstation](#workstation) |
| `character refs` stalls on Wikimedia Commons | HTTP 429 rate limiting; `refs` waits 5, 10, 20, 40 and 80 s between at most 5 download attempts, then raises |
