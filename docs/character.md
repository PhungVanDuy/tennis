# Character: references to a rigged game character

This document covers the `r3d character` stages, which turn identity photos and the client's style references into the stylized, rigged, game-ready character: `refs` (collect references), `stylize` (a hero image in the target style and an A-pose turnaround), `mesh` (image-to-3D and auto-rig through fal, with an A/B generator). It also covers the procedural racket that `retarget` attaches to the rig. It says what each step sends to which model, what comes back, where the files land, the numbers from the current run (`runs/rybakina`) and the known limitations. How the moveset is retargeted onto this rig is in [motion-pipeline.md](motion-pipeline.md#11-retarget-smpl-motion-onto-the-character-rig-in-blender), and how the character is rendered and painted for the character sheet is in [artwork.md](artwork.md). Paths of outputs are relative to the run folder `runs/rybakina/`; code paths are relative to `src/rybakina3d/`.

## At a glance

| Step | Command | Model / tool | Output (under `runs/rybakina/`) |
|---|---|---|---|
| References | `r3d character refs` | Wikimedia Commons API, the client's deck (PDF) and a screenshot of a reference sheet | `character/refs/*.png\|jpg`, `refs.json` |
| Stylize | `r3d character stylize [--force]` | `gemini-3-pro-image` (`models.image_hero`, `models.image_views`) | `character/sheet/hero.png`, `turnaround.png`, `views/*.png`, `stylize.json` |
| Mesh + rig | `r3d character mesh [--force]` | fal `fal-ai/meshy/v7.1/multi-image-to-3d` (primary), `tripo3d/h3.1/multiview-to-3d` + `fal-ai/meshy/rigging` (A/B); `gemini-3.8-flash` names the side views | `character/mesh/{meshy,tripo}/`, `mesh.json`, `character/rigged.glb` |
| Racket | built inside `r3d remote retarget` | `blender/racket.py` (procedural) | part of `anim/rybakina_moveset.glb` / `.fbx` |

All three character stages run on the workstation (they call hosted APIs; keys stay there). `scripts/run_pipeline.sh` runs them in this order as its `character` stage.

## Inputs you provide

None of the client material is in the repository; `.gitignore` excludes it. [setup.md](setup.md#private-inputs) lists every private input. To run `character refs` you need, at the project root:

| File | What it is | Read by |
|---|---|---|
| `Intelligent Internet Mail - Task - 3D tennis.pdf` | the client's brief as a PDF; `refs` extracts five embedded images from it: three identity photos and two style references | `character/refs.py` (`CLIENT_PDF`) |
| `Screenshot 2026-10-21 at 14.05.59.png` | a screenshot that shows part of a character sheet in the target style | `character/refs.py` (`SHEET_SCREENSHOT`) |
| `naomi_style.png` | the client's style reference sheet (not used here, but needed by the [artwork](artwork.md) stages) | `artwork/paint.py` (`reference`, `paper_colour`), for `paint`, `compose`, `motion-render`, `motion-paint` and `motion-compose` |

The PDF images are picked by page and image index, with an optional crop (`PDF_REFS` in `refs.py`), and the screenshot is cropped with `SHEET_CROP`. A different deck needs those tables edited. `assets/client/` (also ignored) holds images extracted from the deck by hand; no code reads it.

Secrets go in `.env` at the project root (read by `util.secret`): `GEMINI_API_KEY` and `FAL_KEY` (or `FAL_API`). The Tripo branch of `mesh` also needs a local Blender to convert FBX to GLB: `$R3D_BLENDER_LOCAL`, else `/Applications/Blender.app/Contents/MacOS/Blender`, else `blender` on the `PATH`.

## 1. References: `r3d character refs`

`refs.py` writes every reference to `character/refs/` and lists it in `refs.json` (`name`, `role`, `source`, `path`, `note`). The stage rewrites everything on each run; it has no cache and no `--force`.

| Name | Role | Source | Used by |
|---|---|---|---|
| `client_avatar_card` | style | client deck (cropped) | stylize: hero (identity and target style in one image) |
| `osaka_render` | style | client deck | stylize: hero |
| `osaka_character_sheet` | sheet | the screenshot, cropped | stylize: hero and turnaround |
| `photo_racket_ready` | identity | client deck | stylize: hero |
| `photo_smile`, `photo_serve_prep` | identity | client deck | collected only; no model receives them at present |
| `commons_full_body` | identity | Wikimedia Commons | stylize: hero (full-body proportions) |
| `commons_face_front` | identity | Wikimedia Commons | stylize: hero (face) |
| `commons_upper_body` | identity | Wikimedia Commons | stylize: hero (face and build) |
| `commons_back_hair` | identity | Wikimedia Commons | stylize: turnaround (hair from behind) |

How it fetches from Wikimedia Commons (`_commons`):

- One MediaWiki API query per file (`prop=imageinfo`, `iiprop=url|size|extmetadata`, `iiurlwidth=1600`). If the original is wider than 1600 px it downloads the thumbnail rendition, otherwise the original; tracking query strings are dropped from the URL. Commons may serve the nearest standard thumbnail size; the stored files are 1370 to 1920 px wide.
- Every request sends a descriptive `User-Agent` that names the tool and its purpose (`UA` in `refs.py`), as Wikimedia's User-Agent policy asks.
- An HTTP 429 (rate limited) download is retried with exponential backoff: waits of 5, 10, 20, 40 and 80 s, five attempts at most, then the error is raised. The stage pauses 1 s between files.
- The file page URL goes into `source`, and the licence short name (from `extmetadata`) is appended to `note`. The artist field is fetched too but not stored, so full attribution is on each file page. The four files in use are CC BY-SA 4.0 (two), CC0 and CC BY-SA 2.0.

## 2. Stylize: `r3d character stylize`

`stylize.py` makes one front "hero" image of the athlete in the target style, then a four-view turnaround conditioned on that hero, and cuts the turnaround into the square views that image-to-3D wants.

Model choice. This stage runs on Gemini (`gemini-3-pro-image` for both `models.image_hero` and `models.image_views` in `configs/rybakina.yaml`). The character was generated with it earlier in the project. The artwork stages use GPT Image 2.5 Sunburst on fal (`models.image_paint`). The character has not been regenerated with Sunburst. Regenerating it would change the mesh, the rig, every retargeted animation and every artwork figure, because `hero.png` is also the identity reference for the artwork repaints.

### Hero

- Input: labelled image parts, then the prompt. The parts are two STYLE references (`osaka_render`, `osaka_character_sheet`, each marked "a different person"), the client avatar marked as IDENTITY plus style, and four IDENTITY photos (`photo_racket_ready`, `commons_face_front`, `commons_upper_body`, `commons_full_body`).
- Prompt (`HERO_PROMPT`, built from `STYLE`, `OUTFIT`, `POSE` and `STAGE` in `stylize.py`): keep her real facial structure and tall athletic build; a stylized, semi-realistic sports-game render; the white dress with dark-green trim, visor, wristbands, socks and shoes, and the low ponytail; a neutral A-pose for rigging (arms about 40° from the body, palms to the thighs, fingers relaxed); flat even light, a plain light-grey background, the whole body in frame, no racket, ball, text or logos.
- The prompts never name the athlete. The image filters key on names, so the photos carry the identity.
- Image config: aspect `3:4`, size `2K`, which gives 1792 x 2400 in this run. A call that returns no image is retried, three calls at most, and the finish reason is logged.

### Turnaround

- Input: the hero, `osaka_character_sheet` (style) and `commons_back_hair`, then `SHEET_PROMPT`: the same character in four views side by side (front, left side, back, right side), all at the same height and scale with the feet on one ground line, all in the hero's A-pose.
- Image config: aspect `21:9`, size `4K`, which gives 6336 x 2688 in this run.

### Splitting the turnaround into views

`_split_sheet` separates the figures from the background:

1. The background colour is the median of the top and bottom 8 pixel rows. A pixel is foreground when its summed RGB difference from that colour is over 40, followed by a 25 x 25 morphological close.
2. Runs of foreground columns are the figures. Runs separated by less than 1% of the image width are merged (a hand's width), and the four widest runs are kept, ordered left to right. If the stage does not find exactly four, it falls back to four equal slices.
3. Each figure is cropped with a 6% margin and centred on a square canvas of the background colour.

The hero is the sharpest front view, so it replaces the turnaround's front figure (also centred on a square canvas). The hero and the turnaround are cached and re-generated only with `--force`; the split runs every time. `stylize.json` records both models, both prompts and the output paths.

## 3. Mesh and rig: `r3d character mesh`

`mesh.py` sends the four views to two image-to-3D generators in parallel, rigs both, and copies the primary result to `character/rigged.glb`.

### Naming the side views

The image model does not always draw the side views where they were asked for. `ordered_views` asks `models.locate` (`gemini-3.8-flash`) which way the nose points in each side view (`image_left`, `image_right`, `camera` or `away`). A figure facing image-right shows her right side. If both side views come back the same, the sheet order is kept. The result is cached in `character/sheet/views/facing.json`. In this run the image model swapped them: `left.png` shows her right side and `right.png` her left, and `mesh.json` records the views in the order they were actually sent (front, left, back, right).

### Generators

The views are uploaded to fal's CDN once and both generators get the same four URLs in the order front, left, back, right.

| | Meshy 7.1 (primary) | Tripo H3.1 (A/B) |
|---|---|---|
| Endpoint | `fal-ai/meshy/v7.1/multi-image-to-3d` | `tripo3d/h3.1/multiview-to-3d`, then `fal-ai/meshy/rigging` |
| Topology | `should_remesh: true`, `topology: quad`, `target_polycount` = `character.target_tris // 2` = 12000 quads, `symmetry_mode: auto` | `quad: true`, `face_limit` = 12000, `geometry_quality: detailed` |
| Texture | `should_texture: true`, `enable_pbr: true` | `texture: true`, `pbr: true`, `texture_quality: detailed`, `texture_alignment: original_image` |
| Pose / rig | `pose_mode: a-pose`, `enable_rigging: true`, `rigging_height_meters: 1.84` | rigged afterwards by Meshy's rigging endpoint with `height_meters: 1.84` |
| Safety checker | off | off |
| Files kept | `rigged_character_glb.glb`, `rigged_character_fbx.fbx`, `model_glb.glb` (unrigged), `thumbnail.png`, plus any `basic_animations` | `model.fbx`, `model.glb`, `rigged_character.glb`, `rigged_character.fbx`, `thumbnail.webp` |

Tripo returns its quad mesh as FBX, and the rigging endpoint wants GLB. `_fbx_to_glb` therefore runs a local headless Blender with factory settings (import FBX, export GLB) before it uploads the GLB for rigging.

Behaviour worth knowing:

- The two generators run in two threads. If one fails, its error is recorded in `mesh.json` and the other's result is kept.
- The primary is Meshy if it produced a rigged GLB, otherwise Tripo. Its rigged GLB is copied to `character/rigged.glb` (only when the bytes differ).
- Without `--force`, a generator that already has a rigged GLB is skipped. If Tripo's mesh exists but its rig does not, only the rigging call is repeated.
- `mesh.json` holds the endpoints, the downloaded files and GLB statistics from `gltf.glb_stats` (triangles, vertices, meshes, materials, textures, skins, joints, animations, size).
- Only the primary rig is used downstream. The Tripo result is there for comparison on the review site.

### Result in this run

| | Meshy (primary, `character/rigged.glb`) | Tripo + Meshy rigging |
|---|---|---|
| Triangles | 24,588 | 21,386 |
| Vertices | 20,520 | 18,347 |
| Meshes / materials / textures | 1 / 1 / 3 | 1 / 1 / 3 |
| Skeleton | 24 joints | the same 24 joints |
| Rigged GLB size | 7.23 MB | 10.99 MB |

The Meshy GLB has one double-sided PBR material with three 2048 x 2048 maps: base colour (JPEG), metallic-roughness (JPEG) and normal (PNG). It has no morph targets. Its armature has a 0.01 scale (centimetre bones); `retarget` applies that scale on import, so the moveset is at unit scale. The character is 1.84 m tall (`character.height_m`).

### Skeleton

The rig has 24 joints. `retarget` maps 22 of them to SMPL joints (`anim/retarget.json`, `mapping`):

| Bone | Parent | SMPL joint |
|---|---|---|
| `Hips` | none | pelvis |
| `Spine02`, `Spine01`, `Spine` | chain from `Hips` | spine1, spine2, spine3 |
| `neck` | `Spine` | neck |
| `Head` | `neck` | head |
| `head_end`, `headfront` | `Head` | none (end and helper bones; they keep their rest pose) |
| `LeftShoulder`, `RightShoulder` | `Spine` | left / right collar |
| `LeftArm`, `RightArm` | shoulder | left / right shoulder |
| `LeftForeArm`, `RightForeArm` | arm | left / right elbow |
| `LeftHand`, `RightHand` | forearm | left / right wrist |
| `LeftUpLeg`, `RightUpLeg` | `Hips` | left / right hip |
| `LeftLeg`, `RightLeg` | up-leg | left / right knee |
| `LeftFoot`, `RightFoot` | leg | left / right ankle |
| `LeftToeBase`, `RightToeBase` | foot | left / right foot |

The rig has no finger bones, no face bones and no blendshapes. The rest pose is an A-pose.

## 4. The racket

The racket is not generated. `blender/racket.py` builds it procedurally inside Blender during `retarget`, and `blender/retarget.py` (`attach_racket`) parents it to `RightHand`.

| Property | Value (constant in `racket.py`) |
|---|---|
| Length | 68.5 cm (`LENGTH = 0.685`) |
| Head | oval, 33 x 26 cm (`HEAD_LENGTH`, `HEAD_WIDTH`), open throat |
| Grip | octagonal, about 3.2 cm across (`GRIP = 0.032`), 19 cm handle |
| Materials | red frame (`RED`), black shaft and grip (`BLACK`), semi-transparent strings drawn as a packed 256 px grid texture (16 mains x 19 crosses), so they survive glTF and FBX export |
| Size | one mesh, three materials, 416 triangles |

The grip is an eastern forehand grip in the rest pose. The racket axis is turned `GRIP_ANGLE_DEG = 25` from the hand's long axis towards the thumb, the palm centre is 7.5 cm along the hand (`PALM_OFFSET_M`) and 1.5 cm off the handle axis (`PALM_DEPTH_M`), and the hand holds the handle 7 cm from the butt cap (`GRIP_FROM_BUTT_M`). The racket is a rigid child of the hand bone, so it follows the recovered wrist. Its face angle is not measured from the footage. With the racket, the moveset GLB has 2 meshes, 4 materials and 25,004 triangles.

The red frame is the only saturated red in a render. The painted-motion racket check in [artwork.md](artwork.md#motion-paint) relies on that.

## Inspecting the outputs

Run `.venv/bin/r3d site` and `.venv/bin/r3d serve`, then open http://127.0.0.1:8765/#/character. The page has a 3D viewer (switch between the Meshy and Tripo results and their variants, pick an animation, show the bones, turntable), statistics for the loaded model, and a Meshy-vs-Tripo table. Below that are the style sheet (hero, turnaround, the four views labelled by the side of her they show, and both prompts with their model) and every reference grouped by role.

Files:

| Path (under `runs/rybakina/`) | Content |
|---|---|
| `character/refs/refs.json` | every reference: role, source (with licence for Commons files), file |
| `character/sheet/stylize.json` | models and exact prompts of the hero and the turnaround |
| `character/sheet/hero.png`, `turnaround.png` | the generated images |
| `character/sheet/views/{front,left,back,right}.png`, `facing.json` | the square views, named by position on the turnaround; `facing.json` maps each side to the file that shows it |
| `character/mesh/mesh.json` | endpoints, files, GLB statistics of both generators, the primary |
| `character/mesh/meshy/`, `character/mesh/tripo/` | everything each generator returned |
| `character/rigged.glb` | the primary rigged character (input to `retarget`) |

Statistics of any GLB from the command line:

```bash
.venv/bin/python -c "from rybakina3d.gltf import glb_stats; print(glb_stats('runs/rybakina/character/rigged.glb'))"
```

The GLB opens in Blender with File > Import > glTF 2.0. Meshy's FBX export of the same rig is `character/mesh/meshy/rigged_character_fbx.fbx`.

## Limitations

- **Face texture resolution.** The whole character shares one 2048 x 2048 base-colour map. Measured from the GLB's UVs, the head (everything above the chin line) covers about 9% of the mapped UV area. That is roughly 0.17 megapixels, about a 415 x 415 texel patch for the whole head, and the face is a fraction of it. Head close-ups and the bust on the character sheet magnify this. The texture also has generator artefacts: skin-coloured patches on the dress, seams across the face and stringy hair. The artwork repaint prompt asks the image model to paint over them (see [artwork.md](artwork.md#paint)); the raw render shows them.
- **No finger bones.** Meshy's auto-rig stops at `LeftHand` / `RightHand`. The hands stay open and stiff, so the grip on the racket is not articulated. On two-handed backhands the left hand follows the recovered left wrist and is not constrained to the handle.
- **No face controls.** The rig has no face bones and no blendshapes, so it cannot show expressions. The sheet's four head close-ups are four camera angles instead.
- **No secondary motion.** Hair, ponytail and dress are skinned to the body. There is no cloth or hair simulation.
- **Identity is the image model's interpretation** of the photos. Nothing checks likeness numerically.
- **Not reproducible bit for bit.** The hosted models are non-deterministic. Re-running `stylize` or `mesh` with `--force` produces a different character, and everything downstream (retarget, artwork) has to be redone.
- **Client files required.** `character refs` cannot run without the deck PDF and the screenshot under the exact names above. There is no option to skip them.
