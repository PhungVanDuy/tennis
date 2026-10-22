# Review site

The review site is a local, static single-page website over one run folder. A reviewer uses it to check every stage's result side by side: the footage index, each stroke clip as broadcast video, GVHMR overlay and retargeted 3D character in sync, the character and its meshes, the character sheet and atlas, the moveset in motion, any character variant's Quest 3 renders, and the run's limitations. This document explains how `r3d site` builds it (`src/rybakina3d/site.py`), how `r3d serve` serves it, how `r3d publish` shares it from the GPU host, what each tab shows and which files it reads, how the front-end in `web/` is organised, how to check it headlessly, and how to extend it. For the run folder and the stages that produce the data, see [architecture.md](architecture.md). For what the artwork itself is, see [artwork.md](artwork.md). For running the pipeline, see [setup.md](setup.md).

## Build and serve

```bash
.venv/bin/r3d site                 # (re)build runs/rybakina/site/ from the run folder
.venv/bin/r3d serve                # http://127.0.0.1:8765/  (site.port; --port to change)
```

`r3d serve` builds the site first if `site/data/site.json` does not exist yet. After that, rebuild with `r3d site` whenever a stage has written something new, then reload the page. A running server does not need a restart.

The site is safe to build at any point, including while stages are still running. A missing or unreadable output is written as `null` and shown as pending, so a half-finished run still renders.

## What `r3d site` writes

```
runs/rybakina/site/
  index.html app.js player.js viewer.js style.css   copied from web/ on every build
  data/site.json                                     everything the page shows, one file
  data/motion/<clip>.json                            GVHMR world joints of one clip, for the 3D skeleton
  media/...                                          relative symlinks into the run folder
```

| Content | Copied or linked | How |
|---|---|---|
| Front-end files from `web/` | copied (`_copy_web`) | `shutil.copy2` on every build; files in `site/` that `web/` no longer has are deleted, except under `data/` and `media/` |
| `data/site.json` | generated | aggregated from every stage's reports (next section) |
| `data/motion/<clip>.json` | generated | `{fps, n, contact_index, parents, joints}`: the 24 SMPL joints of `motion.npz` (`joints_world`), rounded to millimetres; files of clips that are gone are deleted after the new ones are written |
| Every image, video and model the page shows | symlinked (`_Linker`) | `media/<rel>` → a **relative** link to the file in the run folder |

`_Linker` is the single gateway from the run folder to the page. `link(src, rel)` returns the page URL `media/<rel>`, or `None` when `src` does not exist, so a missing file turns into a pending box rather than a broken link. Each link is created under a temporary name and moved over the old one with `os.replace`, which is atomic. Links that the current build did not write are pruned only at the end. As a result, a running `r3d serve` never answers 404 for a file that exists while the site is being rebuilt. Because the links are relative, the whole run folder can be moved without breaking them.

Nothing heavy is copied: the workstation keeps one copy of every video. There is one deliberate exception. The In motion grid would choke a browser if it played dozens of 1024 px videos at once, so `artwork motion-compose` writes light 360 px copies (`TILE_WEB = 360`) into `artwork/motion/rybakina_moveset_tiles/`. The site links those copies when they are at least as new as the clip's render or repaint. Otherwise it links the full video (`site._fresh`). A character variant's section works the same way with `<variant>_moveset_tiles_<profile>/` (336 x 360 copies of the portrait renders), except that a tile older than its render is still played and counted as stale ([Character variants](#character-variants)). Its files are linked under `media/variants/<name>/…`.

For videos, `_video()` also runs `ffprobe` to record `start` (where the first frame sits on the media timeline, because previews with audio start one frame late), the frame count and the size. The player uses `start` to line the videos up.

At the time of writing (2026-10-22), `site/` holds about 3 MB of real files (`data/site.json` is 0.75 MB) and about 450 links that resolve to about 200 MB of media.

### `data/site.json`

| Key | Built from | Used by |
|---|---|---|
| `generated` | build time | top bar ("run built …"), Notes |
| `athlete`, `config` | the loaded config (models, clips, track, motion, character, retarget) | labels, model names |
| `footage` | `source/probe.json`, `shots.json`, `index/index.json`, clip metas | Overview KPIs, Footage |
| `selection` | `selection.json` (without its clip rows) and the `select:` config | Overview |
| `clips[]` | per clip: `clip.json`, the media links, summaries of `track.npz`, `motion.npz` and `camera.npz`, `qa.json`, the curation row with a sentence saying why it was kept or flagged, the signature row, the animation name in the moveset GLB, the `retarget.json` entry, the check renders | Moveset, Footage, Artwork |
| `character` | `character/refs/refs.json`, `sheet/stylize.json`, `mesh/mesh.json`, `glb_stats` of every GLB, `rigged.glb` | Overview, Character |
| `moveset` | `anim/rybakina_moveset.glb` / `.fbx` (links + `glb_stats`), `anim/retarget.json`, `anim/checks/` | Character, Moveset, Artwork |
| `artwork` | `artwork/artwork.json`, sheet PNGs and their JPEG previews, `artwork/poses.json`; `motion`: `artwork/motion/render/renders.json`, `motion/painted/<clip>.json` and `paint.json`, `motion/motion.json` | Artwork |
| `variants[]` | per folder in `variants/`: `artwork/motion/motion_<profile>.json` and `render_<profile>/renders.json` for `quest3` and `draft`, the tile copies, the variant's moveset GLB / FBX (links + `glb_stats`), and the `configs/*.yaml` that names the variant | Artwork |
| `qa_summary` | `qa_summary.json` | Notes |
| `curation` | `curation.json`, with each clip's current QA gate | Overview, Moveset, Footage |
| `signature` | `signature.json` | Overview, Moveset |
| `pipeline[]` | one node per stage (see Overview) | Overview, Notes |
| `notes` | `{limitations, next}`, sentences generated from all of the above | Notes |

NaN and infinity are written as `null` (`_finite`), because browsers refuse bare `NaN` in JSON.

## How it is served

`r3d serve` (`site.serve`) is a small threaded static server over `site/`:

- It binds `127.0.0.1` only, on `site.port` (8765) or `--port`.
- It follows the `media/` symlinks like ordinary files.
- **HTTP Range.** Python's `SimpleHTTPRequestHandler` ignores `Range`, and Chrome needs `206 Partial Content` to seek inside a `<video>`. The moveset player seeks frame by frame. `_Handler` answers a `Range` request with `206`, `Content-Range` and only the requested bytes. It honours the first range of a multi-range request, supports suffix ranges (`bytes=-N`), and answers `416` with `Content-Range: bytes */<size>` when the range cannot be satisfied.
- Every response carries `Accept-Ranges: bytes` and `Cache-Control: no-cache`. The browser revalidates, so a rebuild shows up on reload. The server speaks HTTP/1.1 keep-alive, because a video issues many small range requests.
- It sets MIME types for the 3D files (`.glb` → `model/gltf-binary`), video, modules and images.
- It logs only responses with status 400 and above. A video drops its connection every time it seeks, so `ConnectionError`s are not reported.

`r3d serve` is for local review. To let other people open the site, use `r3d publish` (next section).

## Sharing the site: `r3d publish`

`r3d publish [--port N]` (`remote.publish`) serves the review site from the GPU host behind a Cloudflare quick
tunnel and prints its public `https://<random>.trycloudflare.com` URL. It runs from the workstation, uses the
same host as `r3d remote` (`remote.host` or `R3D_HOST`) and takes the port from `site.port` (8765) unless
`--port` is given. In order:

1. **Code up.** `scripts/sync.sh`, which also carries `web/`.
2. **Results up.** The workstation's run folder is rsynced to the host with `--update`, so a file the host holds
   in a newer version is never replaced. `PUBLISH_EXCLUDES` leaves out `site/`, partial files and the paint
   stages' scratch (`artwork/painted/attempts/` and `inputs/`, `artwork/motion/painted/sent/`, `attempts/`,
   `first/attempts/` and `first/inputs/`).
3. **On the host**, in `~/<remote.root>/`: `r3d site` rebuilds the site with links valid on that machine (log in
   `logs/site.log`). `r3d serve --port <port>` and `bin/cloudflared tunnel --url http://127.0.0.1:<port>` are
   started as background processes, each with a pid file and a log in `logs/`, and only when they are not already
   running. `cloudflared` (the latest Linux amd64 release) is downloaded into `bin/` on first use.
4. **URL.** The command waits up to 60 s for the tunnel URL in `logs/tunnel.log` and prints it.

Running `r3d publish` again after new results rebuilds the site and keeps the same URL, because the running
server and tunnel are reused. There is no stop command: kill the processes whose ids are in `logs/serve.pid`
and `logs/tunnel.pid` on the host. A new tunnel gets a new URL.

Anyone who has the URL can open the site. It shows the client's reference material, the broadcast previews and
the character, so share the URL only with people who may see them.

## Front-end

There is no build step. The browser loads ES modules directly.

| File | Role |
|---|---|
| `web/index.html` | the shell: top bar with the six tabs (hash links), one `<section class="view">` per tab, the lightbox and the tooltip, and an **import map** that loads three.js `0.169.0` (`three` and `three/addons/`) from `cdn.jsdelivr.net` |
| `web/app.js` | loads `data/site.json`, renders all six views up front, routes on the URL hash, and runs the lightbox and tooltips; exposes `window.r3dReview` for scripted checks |
| `web/player.js` | `Timeline`: one frame clock for the moveset's two videos and its 3D view |
| `web/viewer.js` | three.js helpers: `Stage` (renderer, lights, a fading ground and grid, orbit controls, camera fly-to, follow target; renders only while its tab is active), `loadAsset` (GLB, or FBX with gzip detection), `fitUnits` / `placeOnGround` / `findRig` / `bodyFrame`, and `JointSkeleton` (the GVHMR joints as thick lines: left chain amber, right chain and spine blue, the same colours as the overlay video) |
| `web/style.css` | dark theme defined as custom properties on `:root`, responsive breakpoints from 1200 px down to 420 px |

Routes: `#/overview` (and `#/overview/signature` to jump to a section), `#/character`, `#/moveset/<clip id>`, `#/artwork`, `#/footage/<shot id>`, `#/notes`. Fonts (Inter, JetBrains Mono) come from Google Fonts, with system fonts as the fallback.

## The tabs

### Overview

- A hero with the stylized 2D hero image next to a render of the primary mesh.
- A KPI row: footage minutes, shots, her strokes labelled, stroke clips, motion recovered, triangles, bones, animations.
- The **stroke signature**: per stroke, mean ± sd of each metric over the clips whose motion matches their label, with one dot per clip that links to it.
- The **pipeline status** graph in two lanes. The motion lane runs footage → index → select → clips → track → camera → GVHMR → QA → curate → retarget → signature. The character lane runs references → stylize → mesh + rig → artwork. Each node shows a status dot, a one-line result, a breakdown and the models used.
- A "Models and tools" table with each stage's last update.

Sources: `pipeline`, `footage`, `character`, `signature`.

The status is computed when `r3d site` runs, by `site._pipeline` reading the run folder. Each node counts outputs against what is expected: `pending` when nothing exists, `done` when everything does, `partial` in between. For example, `track` counts `track.npz` files against the clips, and `camera` counts calibrations against the clips in a calibrated view. "Updated" is the newest modification time of the stage's outputs. "Seconds" comes from an optional `timings.json` (`{stage: seconds}`) in the run folder; `retarget` reports its own time through `retarget.json`. The page does **not** poll. It is live only in the sense that you can rebuild at any time and reload.

At the time of writing, every one of the 15 nodes is done: 30.8 min, 283 shots, 255 of her strokes, 39 of 124 candidates selected, 38/39 clips court-calibrated (the side-view clip by design), 39/39 recovered, QA 7 pass, 22 warn, 10 fail, all 39 kept with 15 flagged, 39 animations, and a 24,588-triangle, 24-bone rig.

### Character

- A 3D viewer with three controls:
  - **generator**: Meshy (primary) or Tripo (A/B);
  - **variant**: *Moveset* (the moveset GLB, with every animation), *Rigged* or *Raw mesh*;
  - **animation**: a dropdown with the rest pose or any clip.
  
  It also has *Bones* and *Turntable* toggles, a *Reset view* button, and a "Loaded model" panel showing triangles, vertices, bones, textures, animations, height and file size.
- A Meshy vs Tripo table: triangles, vertices, bones, textures, rigged file size and endpoint.
- The style sheet: the hero, the turnaround and the four cut-out views. Views are labelled by the side of her they show, as recorded in `mesh.json`, with a note when the image model drew the side views in swapped positions. The image prompts sit in a collapsible block.
- The references, grouped by role: identity, style and sheet layout.

Sources: `character.*` and `moveset.url`. At the time of writing, the rig is 24,588 triangles, 24 bones, 3 textures and 7.2 MB. The moveset GLB is 25,004 triangles (the racket included), 39 animations and 8.4 MB.

### Moveset

On the left is the clip list, grouped by stroke (serve, forehand, backhand, slice, volley, overhead, celebration). Kept clips come first, in curation order. Clips that curate rejected sit under a collapsible "Rejected by curate" list; with `curate.keep_all: true` there are none, and failed checks show as *flagged* instead. Each card shows the `track.jpg` thumbnail, duration, view, source time, static or moving camera, curation and QA badges, an `anim` mark when the clip is in the moveset GLB, a low-coverage mark when track coverage is below 0.7, and the short reason for a flag.

The review area for the selected clip contains:

- A head with the stroke, the view, the time in the source, the camera, and the curation and QA badges, plus a banner explaining a flag or rejection. Previous / next buttons walk the kept clips.
- **Three synced views:**

  | View | Source | Notes |
  |---|---|---|
  | Broadcast source | `clips/<id>/preview.mp4` | sound optional (the ball strike) |
  | GVHMR skeleton overlay | `clips/<id>/overlay.mp4` | muted |
  | 3D | the moveset GLB's action named by the clip id, plus `data/motion/<id>.json` | layers *Character* / *GVHMR*. The GVHMR joints are scaled by the retarget's body scale and pinned to the character's hips at the contact frame, so away from contact the vertical gap is exactly what the retarget's re-grounding corrected. |

  The 3D view has three camera presets:
  - *Follow*: orbits her hips.
  - *Broadcast*: from where the TV camera stood. The position comes from `retarget.json`'s `camera_at_contact`, else from the camera pose recovered from GVHMR's world and camera joints (`motion.camera_rel` in `site.json`). The elevation is clamped to 12°–35°, the distance to 8–30 m, and the vertical field of view follows the broadcast.
  - *Side*: from her right side.
- **Transport:** play / pause, frame step, a scrubber with ticks every 5 frames and a contact marker, a readout (frame, time, offset from contact), speed 0.25× / 0.5× / 1×, loop, jump to contact, and sound. Keys: `Space` play, `←` / `→` one frame (`Shift` five), `C` contact, `↑` / `↓` previous / next clip, `Home` / `End`, `1` / `2` / `3` speed, `L` loop.
- **Cards:**
  - Clip: from `clip.json`, with a YouTube link at the source time and a link to the shot in Footage.
  - Curate: the motion-vs-label check against its threshold, the score and rank, the QA gate now and at curate time, and the reason.
  - Stroke signature: this clip next to her mean ± sd, with a z-score.
  - Motion: from `motion.npz`.
  - Capture: track coverage, box height, camera motion, court calibration share and reprojection error, whether GVHMR used the calibration, the lens.
  - QA: metrics and reasons from `qa.json`.
  - Retarget: from `retarget.json`: scale to the rig, ground offset, planted frames, foot skate before → after planting, lowest foot, FK error, warnings.
  - Review renders: the `anim/checks/<id>_<frame>.jpg` sheets, each with a "Show frame N" button, and `camera.jpg`.

**How the sync works** (`player.js`). Clip frames are 1:1 across all three views. While playing, the first video that can play is the master clock, because its decoder paces playback. The other video is pulled back whenever it drifts by more than two frames, and with no video at all a wall clock drives the 3D view. When paused, every view sits on the middle of the current frame (`(f + 0.5) / fps`), so stepping is exact. The 3D pose is set by writing the action's time directly and updating the mixer by zero; `mixer.setTime()` would run three.js's end-of-clip logic and pin the action to frame 0. If an exporter has offset or resampled the animation's timeline, the page rescales it and says so in the 3D pane.

### Artwork

- A header with download buttons for the moveset GLB and FBX.
- **One section per character variant**, when the run folder has any ([Character variants](#character-variants)). With a variant present, the In motion header names the athlete ("In motion — Elena Rybakina").
- **In motion** (from `artwork motion-render`, `motion-paint` and `motion-compose`):
  - a header with the counts (clips rendered, painted, flagged) and a *Painted* / *3D render* toggle (painted is disabled until at least one clip is painted);
  - the moveset reel, centred;
  - a grid of every clip grouped by stroke, using the 360 px tiles, with curate-flagged clips marked.
  
  An `IntersectionObserver` plays only the videos on screen, and leaving the tab pauses them all. Clicking a tile opens the full video in the lightbox, with its frame count, contact frame, pose-check numbers (silhouette IoU median and 10th percentile for a painted clip) and a link to its review in the Moveset tab. A note appears when the reel predates the latest renders or repaints, or when it holds fewer clips than are rendered. The turntable video (`artwork/motion/render/turntable.mp4`) is rendered but not shown on this tab.
- **Character sheet** with a *Painted* / *3D render* / *Side by side* toggle. The page shows the 1600 px JPEG previews; clicking opens the full PNG in the lightbox, where a click toggles 1:1 zoom. Below the sheet:
  - one chip per painted slot with its silhouette IoU and whether the repaint was accepted;
  - the paint-check rule (`ACCEPT_IOU`, read from `artwork.json`; a miss is retried once, best of three, and otherwise the render stays);
  - "Where each pose comes from": every posed slot with its clip (linked to the Moveset tab) and its frame relative to contact.
- **Moveset atlas**: every clip at its contact frame, with curate-flagged clips marked.

Sources: `artwork` (from `artwork.json`, `poses.json` and the motion reports). At the time of writing:

- The sheet is 3840 × 2560 px, with 10 painted figures, 6 silhouettes cut from the render mattes and none left as a plain render.
- The atlas holds 39 clips, 15 of them flagged.
- All 39 clip videos and the turntable are rendered: 1024 × 1024 px, 25 fps, Cycles at 48 samples, about 59 minutes of rendering on one A100.
- The reel is 1920 × 2114 px and loops every 5.2 s, with every contact on the frame at 3.0 s.
- **Painted motion:** `artwork motion-paint` was still running when this was written, and the reel had not been rebuilt since, so it showed the 3D render on every tile ([artwork.md](artwork.md#motion-paint) has a snapshot). *Maintainer: replace this sentence with the final count of accepted painted clips and their IoU range from `runs/rybakina/artwork/motion/painted/paint.json`.*

#### Model comparison

Side-by-side videos in `<run>/compare/` (`site._comparisons`, newest first) open the Artwork tab when there are any: one clip rendered with two characters or settings, everything else equal. A `<name>.json` beside a video may give its `title` and `note`; a `<name>.jpg` is its poster. They are linked under `media/compare/` like any other run file, so a site rebuild keeps them.

#### Character variants

A variant is another character on the same captured motion, under `variants/<name>/` in the run folder ([character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant)). `site._variants` lists every folder in `variants/` that has a moveset GLB, an FBX or an `artwork/motion/` folder; a variant whose character is still being built has no section yet. The section shows its moveset in motion at the Quest 3 render profile, render only ([artwork.md](artwork.md#4-render-profiles-the-quest-3-set)).

**Which profile is live.** `VARIANT_PROFILES = ("quest3", "draft")`. The live profile is the first of the two whose reel report (`motion_<profile>.json`) holds every moveset clip, else the first with any reel report, else the first with any rendered clip. A complete draft reel therefore keeps playing while `quest3` renders batch by batch, and the Quest 3 reel takes over once it holds the whole moveset. Meanwhile every clip whose `quest3` video already exists links that video instead of its draft (`profile: "quest3"` on the clip, `n_hires` on the variant); its grid link reads "2064×2208 ↗" in the accent colour, the others "draft ↗".

What the section shows:

- A header "<name> — Quest 3" with the counts (clips rendered of the moveset, the render size, "per eye" on `quest3`, the frame rate, the clips flagged by curate), a chip for the live profile (*Quest 3 renders*, *Draft preview* or *Pending*), and download buttons for the variant's moveset GLB and FBX with their size and number of animations.
- A short text: the variant playing the athlete's moveset, rendered in Cycles at 2064 × 2208 and 60 fps, render only, every contact on one reel frame.
- The reel, which plays while it is on screen. Its caption gives size, frame rate, loop length, a download link with the file size and a full-size link.
- **Every clip**, grouped by stroke in the atlas order, as portrait tiles at the render's aspect. Hovering a tile plays it (a tap on a touch screen), one at a time, because even the light copies of a 2064 × 2208 grid are dozens of decoders. A click, `Enter` or `Space` opens the full-resolution video in the lightbox with its stroke, size, frame rate, frame count and contact frame and a link to the clip in the Moveset tab. "full ↗" opens the full-resolution video directly. The grid itself never loads the full-resolution videos.

**Partial results.** Renders arrive clip by clip, so the section links whatever exists:

- The clips come from the live profile's reel report merged with its `renders.json`. A clip rendered after the last compose appears in its place in the atlas order, with its poster.
- Before the first compose of the live profile, the grid shows every rendered clip's poster and a note gives the `motion-compose` command.
- A composed clip without a tile yet shows its poster ("poster · tile pending"). A clip not rendered yet shows "render pending".
- With the draft live and `quest3` renders already back, a note says how many Quest 3 clips are rendered and which command makes their reel.
- The commands shown carry `R3D_CONFIG=<the variant's config>`, found by scanning `configs/*.yaml` for one with that `character.variant` (`site._variant_config`).

**Stale flags.**

| Flag | Set when | Shown as |
|---|---|---|
| `reel_stale` | the reel is older than any linked clip render, or it holds fewer clips than are rendered (`n_in_reel` against `n_rendered`) | a note "The reel predates the latest renders (it holds N of the M rendered clips)" with the `motion-compose` command |
| `tile_stale` (per clip), `n_tiles_stale` | a tile copy is older than its clip's render | the tile still plays; a note says how many tiles predate a re-render and that the full-resolution link is current |

The data is `variants[]` in `site.json`: per variant `variant`, `name`, `motion_from`, `config`, `profile` (live), `composed`, `size`, `fps`, `reel` (`url`, `poster`, `size_mb`, `size`, `frames`, `fps`, `loop_s`), `reel_stale`, `groups`, `clips` (`id`, `stroke`, `profile`, `flagged`, `mp4`, `mb`, `poster`, `tile`, `tile_poster`, `tile_stale`, `size`, `fps`, `frames`, `contact_frame`), the counts `n_clips`, `n_rendered`, `n_hires`, `n_tiles`, `n_tiles_stale`, `n_flagged`, `n_in_reel`, `progress` (per profile: `rendered`, `composed`, `size`), `downloads`, `made` and `updated`. `r3d site` logs each variant's progress, for example `Naomi Osaka: 39/39 clips (draft)`.

### Footage

- A timeline of every shot of the compilation, coloured by camera view (main broadcast, behind low, side, close-up, other), with a marker under each clip's contact. Markers distinguish kept and rejected clips and link to the Moveset tab. Hovering a shot shows its label.
- Filters:
  - by view and by stroke;
  - her strokes only;
  - hide replays;
  - *usable for capture*, using the same rules as `select`; `site._shot_usable` mirrors `selection._candidates`;
  - moveset clips only.
- A shot table whose rows expand into the shot's frames, flags, why it was skipped for capture, the clips cut from it, and every Gemini event with time, hitter and confidence.

Sources: `footage` (from `probe.json`, `shots.json`, `index/index.json` and the clip metas). At the time of writing: 283 shots over 30.8 minutes, 68 usable for capture, 255 of her strokes and 186 opponent strokes labelled.

### Notes

- **Limitations** and **next steps**. These are not hand-written. `site._notes` generates each sentence from the run's own outputs: shot exclusions, label mismatches found by curate, calibration coverage, the depth share of the remaining foot skate, foot planting before and after retarget, racket and hand rig limits, and QA counts. The text therefore changes when the run changes.
- The stage status table (the same nodes as the Overview).
- The QA summary per stroke from `qa_summary.json`: pass / warn / fail, median skate, jitter, reprojection error and |Δt| to contact.

## Checking it headlessly

None of these checks write to the run folder, except `r3d site` itself, which only writes `site/`.

```bash
.venv/bin/r3d site
.venv/bin/r3d serve &                                   # or use one that is already running

# data and Range support
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8765/data/site.json        # 200
curl -s -D - -o /dev/null -H 'Range: bytes=0-99' \
  http://127.0.0.1:8765/media/clips/serve_05m21s/preview.mp4 | head -1                # HTTP/1.1 206 Partial Content

# every media link resolves (prints nothing when none is broken)
find -L runs/rybakina/site/media -type l

# stage status straight from the data
.venv/bin/python -c "import json; d = json.load(open('runs/rybakina/site/data/site.json')); \
print([(s['id'], s['status']) for s in d['pipeline']])"

# each variant: live profile, clips rendered, clips in the moveset, stale reel
.venv/bin/python -c "import json; d = json.load(open('runs/rybakina/site/data/site.json')); \
print([(v['variant'], v['profile'], v['n_rendered'], v['n_clips'], v['reel_stale']) for v in d['variants']])"
```

To check that the page boots and renders, use headless Chrome. Set `CHROME` to `google-chrome` or `chromium` on Linux, or to the binary inside the app bundle on macOS. The `--virtual-time-budget` gives the modules, `site.json` and the GLB time to load.

```bash
"$CHROME" --headless=new --virtual-time-budget=15000 --dump-dom 'http://127.0.0.1:8765/#/overview' > dom.html
grep -c 'id="boot"' dom.html                   # 0: the loader was removed, the page booted
grep -o 'pnode status-[a-z]*' dom.html | sort | uniq -c   # pipeline nodes by status

"$CHROME" --headless=new --window-size=1440,1000 --virtual-time-budget=20000 \
  --screenshot=moveset.png 'http://127.0.0.1:8765/#/moveset/serve_05m21s'
```

If `site.json` cannot be loaded, the `#boot` element stays and shows "Could not load the run data". The screenshot command renders the three synced views, WebGL included, in current headless Chrome.

For scripted checks, from DevTools or a browser automation tool, the page exposes `window.r3dReview`. It returns `data()` (the parsed `site.json`), `state` (the current tab and clip), `moveset()` (the moveset player, for example `r3dReview.moveset().tl.frame`), `character()` (the character viewer) and `variants()` (the variant sections' playback state: the reels on screen and the tile playing).

The Python side is covered by `tests/test_artwork_motion_compose.py::test_site_motion_block`, which builds the In motion block over a temporary run, checks the URLs it produces, and checks that every linked file really exists in the site, and by `tests/test_site_variants.py`, which does the same for a fake variant: what exists is linked, the draft is live until the Quest 3 reel is composed, clips rendered after a partial compose keep the atlas order, and `build()` writes `variants` into `site.json`. The front-end has no automated tests.

## Extending the site

- **New data.** Add a helper in `site.py` that reads with `_maybe_json` / `_load_npz`, which turn missing or unreadable files into `None` with a warning, and exposes files only through `link(src, rel)`. Add its result to the dict in `build()`. Keep `site.json` small, because the page fetches it in one request. Put bulky per-clip arrays in separate files, as `data/motion/<id>.json` does, and fetch them on demand with `fetchJSON` in `app.js`, which caches.
- **New stage on the pipeline graph.** Add a tuple to `stages` in `site._pipeline`: id, lane (`motion` or `character`), name, models, status, detail, breakdown, output paths and a one-line description. Add an edge to `EDGES` in `app.js` if it should be connected in the graph. A sentence in `site._notes` can report its limitations.
- **New tab.**
  1. Add `<a href="#/x" data-view="x">` to the nav and `<section class="view" id="view-x" data-view="x" hidden></section>` to `web/index.html`.
  2. Add `'x'` to `VIEWS` in `app.js`.
  3. Write `renderX()` and call it from `boot()`.
  4. If the tab owns a WebGL stage or videos, start and stop them in `route()`, as the Moveset, Character and Artwork tabs do.
- **Front-end conventions:**
  - Pass every string from `site.json` through `esc()` before it goes into `innerHTML`. Labels, prompts and notes are model output or file content.
  - Show missing data with `pendingBox(...)` and the command that produces it.
  - Give video-heavy views `preload="none"` and play only what is on screen (see `wireMotion`).
  - Show large images as a preview, with the full file in `data-full` for the lightbox.
- **Test it** the way `test_site_motion_block` does: build a small run in `tmp_path`, call the helper with `site._Linker(tmp_path / "site")`, and assert the URLs and that the linked files exist. Then check the page headlessly as above.

## Limitations

- **It needs a network connection.** three.js comes from a CDN through the import map. `app.js` imports `viewer.js`, which imports three.js, so offline the module graph fails to load and the page stays on its loader. The fonts fall back to system fonts.
- **It must be served over HTTP.** ES modules and `fetch` do not work from `file://`. Use `r3d serve`, or any static server that follows symlinks and supports `Range`.
- **It is not live.** Statuses and notes are as of the last `r3d site`. Rebuild and reload to see new results.
- **`r3d serve` is local only.** It binds to `127.0.0.1`, and `media/` holds symlinks, so copying `site/` alone copies dangling links (`rsync -aL` would dereference them, about 200 MB). The site shows client reference material and broadcast previews, which is why `site/` and `runs/` are gitignored. Share it through `r3d publish` only with people who may see that material.
- **The 3D views need WebGL.** Without it, the 3D panes show a message while the videos and every other view still work.
- **One run per site.** The site shows the run folder of the loaded config (`--config` or `R3D_CONFIG`). Build it with the default config: the variants in that run folder's `variants/` get their sections whichever config is loaded, but a variant's config would put the variant's character, moveset and artwork paths in the main tabs.
- **A variant shows only its moveset in motion.** Its section has the Quest 3 or draft reel, the clips and the GLB / FBX downloads. Its character, references, sheet and atlas are not on the site.
