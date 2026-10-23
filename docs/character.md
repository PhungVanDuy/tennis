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

The rig has no finger bones, no face bones and no blendshapes. The rest pose is an A-pose. (The hand settings in section 4 add finger bones for a character whose hand mesh has separate fingers.)

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

**Hands (off by default; on in `configs/naomi.yaml`).** With those offsets the handle sits inside the open palm and the butt inside the wristband. `character/hands.py` finds the finger joints, finger weight bands, the palm frame and a racket seat on the rigged character's own welded hand mesh and writes `character/hand_rig.json` (+ `.npz`); a hand it cannot read (Rybakina's fused fingers) gets `"ok": false`, and retarget keeps the fixed grip there. Three retarget settings use it ([setup.md](setup.md#other-configs-extends-and-character-variants)): `grip: palm` seats the handle on the palm skin with the butt at the heel of the hand (`blender/fingers.py` `seat_frame`) and turns the right hand by a constant offset in every frame, so the racket's world direction stays what the fixed grip gives (fixing it needs the racket measured from the footage); `fingers` adds 15 finger bones per hand (54 bones instead of 24), splits each hand's weight over them after the other skin fixes, keys the right hand in a grip solved against the seated handle and the left from relaxed to grip; `left_hand_ik` puts the left palm on the handle against the right hand wherever the wrists are close (`blender/twohand.py`), the ready position included (higher up, the closed hand would reach into the open throat). The hand goes on when GVHMR's wrists come within 20 cm and stays on until they are 30 cm apart (they stay 20-30 cm apart through some two-handed strokes). Coming and going, the palm goes straight between the motion's palm and a point 5 cm out from its grip spot, turning into the grip on the way (the wrist bent at most 70° off the forearm), then straight down onto the handle, closing the fingers as it comes down (their closing changes by at most 0.35 a frame, 1 being the grip); the left hand's skin (`fingers.hand_points`) is pushed out of the racket wherever it would go in, the motion's own hand included. Each ramp slows only over the part of the blend that has far to go, so the palm never steps more than 1.4 cm, and the elbow 2.5 cm, beyond the motion's and the full IK's own steps. The IK elbow bends the way the motion's own elbow does off its shoulder-wrist line (taken off the shoulder-target line instead, it flipped sides where the target crossed a nearly straight arm: a 15 to 20 cm jump in one frame on three serves). (A path round the handle at the hover's distance was two to three times longer than the straight one; with ramps that much slower the hand rarely reached the handle in a stroke.) `anim/retarget.json` then records per clip the grip metrics on every frame (`clips.<id>.grip`), and a clip warns where more than 30 of one hand's vertices are inside the racket. Measured on Naomi's 5 s clip (`configs/naomi_5s_grip.yaml` against `configs/naomi_5s.yaml`, the same build with the fixed grip), the butt moves from 1.2 to 5.7 cm past the wrist joint, the worst frame's vertices inside the racket go from 396 to 4 of 3,645 on the right hand and from 479 to 25 of 3,770 on the left, the left palm's skin sits a median 1.4 cm from the handle axis instead of 6.5 cm, and the racket's elevation around contact is unchanged. On this hand the seat reaches two of its limits (the 15° tilt and the butt 5 cm past the wrist), which the build logs. The constant right-hand offset (28°) bends the wrist visibly in the take-backs until the racket is measured from the footage.

## 5. A character drawn by the art team: the Naomi variant

The same 39 captured strokes can drive a second character. `configs/naomi.yaml` `extends` the Rybakina config and sets `character.variant: naomi`, so footage, clips, curation and the SMPL motions stay shared, while `character/`, `anim/` and `artwork/` live under `runs/rybakina/variants/naomi/` with `naomi_*` file names (`RunLayout.out` and `RunLayout.stem` in `config.py`). Every stage takes `-c configs/naomi.yaml`; `r3d remote` needs `R3D_CONFIG=configs/naomi.yaml` instead so the host runs the same config. The run order for a variant is in [setup.md](setup.md#a-character-variant), and its moveset rendered for Meta Quest 3 in [artwork.md](artwork.md#4-render-profiles-the-quest-3-set).

The art team already designed this character, so nothing is invented from photos. Their renders (`character.refs_dir`, kept out of the repository) are the references, listed with the outfit text the view prompts quote in a manifest beside them, `<refs_dir>/refs.yaml` (`{outfit: str, refs: [{name, role, file, note}]}`, read by `character/art.py` `manifest`), so neither their file names nor the design's description are in the code:

| Reference | Role | Used for |
|---|---|---|
| `art_front` | front | the A-pose front render (1715 x 2515, alpha cut-out): the front view itself; Meshy's texture guide |
| `art_action` | style | the character in action: outfit and hair from another angle (side and back views) |
| `art_head_front/left/right/back` | head | 2000 x 2000 head renders: face and skull in profile; the front one is the face stage's source |

The head renders show the face with a different hairstyle and top than the full-body art; they only contribute the face.

| Step | Command (`-c configs/naomi.yaml`) | Model / tool | Output (under `variants/naomi/character/`) |
|---|---|---|---|
| References | `r3d character refs` | copy (`character/art.py`, `ART_REFS`) | `refs/*.png`, `refs.json` |
| Views | `r3d character stylize` | GPT Image 2.5 Sunburst (`models.image_paint`), 2 candidates per view | `sheet/attempts/`, `sheet/{back,left,right}.png`, `sheet/views/*.png`, `stylize.json` |
| Mesh + rig | `r3d character mesh` | Hitem3D Hi3D v3.0 (`models.mesh_alt`) rigged by `fal-ai/meshy/rigging`, primary (`character.mesh_primary: hi3d`); no Meshy generator (`models.mesh: ""`) | `mesh/hi3d/`, `mesh.json`, `rigged.glb` |
| Face (optional, not used for Naomi) | `r3d character face` | Blender (local) + OpenCV YuNet | `face/*`, `rigged.glb` with the new base colour |

**Views.** The art team's front is the front view. Sunburst draws the back first, then each side, on the front's canvas size (1648 x 2416) with a transparent background, from the front, the action art and two head renders; the back joins the references of both sides so hair and outfit agree around the figure. `PICK` in `art.py` overrides the first candidate. Each figure is cut out by alpha, scaled to the front's figure height, feet on one line, and centred on a square light-grey canvas; `facing.json` is written directly, so the Gemini side-view check is skipped.

**Mesh.** Three generators were tried on the same four views, each rigged by the same Meshy rigging endpoint (`_rig`, the 24-joint skeleton `retarget` already maps), and compared under the artwork's studio light:

| Generator | Call | Result |
|---|---|---|
| Meshy 7.1 | `fal-ai/meshy/v7.1/multi-image-to-3d`, as for Rybakina plus `geometry_resolution: 2k`, `texture_image_url` = the art front, 50,000 quads | 98,392 triangles, 2048 x 2048 maps, ~8 min on fal. Reads as clay: lumpy hair, ridged visor, a pale blotchy face |
| Hunyuan3D 3.1 Pro | `fal-ai/hunyuan-3d/v3.1/pro/image-to-3d`: `input/back/left/right_image_url`, `enable_pbr`, `face_count` 200,000 (`_hunyuan`) | 199,986 triangles, 4096 x 4096 base colour, metal-roughness and normal maps, <3 min. Crisp pleats and ruffle, clean visor, real hair volume; a generic face |
| **Hi3D v3.0** (primary) | `hitem3d/hi3d/v3.0/multi-view-to-3d`: `front/back/left/right_image_url`, `resolution: 2048quality`, `enable_texture`, `enable_pbr`, `face_count` 280,000 (`_hi3d`; Hi3D advises 2M, Meshy rigging takes ~300k at most) | 280,000 triangles, 8192 x 8192 base colour and metal-roughness, ~9.5 min on fal plus ~3 min rigging. The face, brows and hair closest to the art, the shoe logos legible |

The deciding test rendered `backhand_14m19s` at the `quest3` profile with the Hunyuan and the Hi3D rig and put them side by side (full frame and a 1:1 window that follows the figure). Hi3D was chosen for the face and hair; Hunyuan's skin is smoother. The earlier outputs are kept out of the run's `variants/` in `runs/rybakina/archive/naomi_hunyuan/` (Hunyuan: draft set, 8 `quest3` clips) and `archive/naomi_hi3d_test/` (the one-clip test).

**Face** (`character/face.py`, `blender/face_project.py`), written for the Meshy rig, whose face came out pale and blotchy from a ~250 px face lit from the front. On Meshy it helps; on Hunyuan it did not: Hunyuan's face shape differs from the art's, so the projected features landed beside its own (doubled lips, a streak along the cheek). Naomi's `rigged.glb` is Hi3D's rig as rigged, without this stage. What the stage does:
1. evens the skin (`even_skin`): skin texels lighter than the body's median skin lightness are pulled towards it (Lab L 178 on the face against 134 on the body in this run), keeping `SKIN_KEEP` = 35 % of their excess;
2. renders the rest-pose head unlit through a front orthographic camera at 2048 px;
3. finds eyes, nose tip and mouth corners with YuNet (`weights/yunet/face_detection_yunet_2023mar.onnx`, from opencv_zoo) in that render and in the front headshot, and maps the headshot onto it with a similarity transform (landmark residual 10.5 px at a 263 px eye distance);
4. keeps only the headshot's detail (`delight`): its low frequencies below 0.2 x the eye distance are replaced by the rig's own shading blurred at 0.6 x, so their studio light and skin tone leave no seam;
5. masks a face oval (brows to chin) to the render's skin pixels and to the headshot's own skin inset by 0.2 x the eye distance (its face outline would otherwise project as an edge), feathers it, and bakes the projection (front-facing corners only) into a base colour of at least 4096 px with Cycles, which replaces the original image in the GLB (`gltf.replace_image`).

Re-running `character mesh` puts the untouched rig back, so when the stage is used, run it after `mesh` and before `retarget`.

**Racket.** `character.racket_colors` (linear RGB `frame`, `grip`, `strings`) reaches `blender/racket.py` through the retarget job: the art direction's charcoal frame with lime strings instead of Rybakina's red.

**Hands.** `configs/naomi.yaml` switches the hand settings of [section 4](#4-the-racket) on: the moveset has 54 bones (the auto-rig's 24 and 15 finger bones per hand), the racket sits on the right palm and the left hand goes onto the handle in two-handed strokes and ready positions. Over the 39 clips (2,531 frames, `anim/retarget.json`): the left hand is on the racket (weight over 0.5) on 1,382 frames, at full weight on 1,025, with its fingers closed past halfway on 1,155; 36 clips hold it at full weight somewhere, in `forehand_09m57s` it only hovers in the grip beside the handle (13 frames over 0.5, the full IK short of it), and in two (`forehand_27m36s`, `overhead_14m57s`, where the running median of GVHMR's wrist distance never comes within 20 cm) it never goes on. The worst frame has 89 of the left hand's 3,770 vertices inside the racket (`serve_12m47s` frame 57, mid-approach), more than 30 on 5 frames in 4 clips; the right hand's worst is 4 of 3,645. The left palm never steps more than 1.4 cm, and the left elbow 2.45 cm, beyond the motion's and the full IK's own steps, the wrist bends at most 70° on the way to the grip and 73° in the last centimetres onto it, and the push moves the hand off the racket on 720 frames, by up to 10 cm. The left-hand IK and its push take about 6 s a clip, so the 39-clip build runs about 5 min on the GPU host. When the hand takes hold is read from GVHMR's wrist distance, not from the footage; GVHMR's left palm sits a median 10 cm from its grip spot while the wrists are close, so a ramp on or off takes about half a second (a median 12 frames from off to full weight), and where the full IK is 2 to 4 cm short of the handle the hand hovers in the grip beside it with its fingers open.

**Knees.** Hi3D ships every triangle with its own three vertices (839,998 vertices for 280,000 triangles). Skinned like that the surface shades in facets and a bending knee tears into folds behind the joint, on Hunyuan's mesh too once rigged by the same endpoint. `retarget` therefore welds coincident vertices of every skinned mesh before anything else (`weld_meshes`, `retarget.weld_m` = 1e-5 m: 139,998 vertices here, UVs kept on the face corners, smooth shading), so the moveset GLB and FBX carry the welded mesh. Naomi's config also smooths the limb weights (`retarget.smooth_limb_weights: 20`: 20 Laplacian passes at factor 0.5 over the thigh, shin, foot, upper-arm, forearm and hand groups, then normalised), which spreads a bend over more of the limb; the `quest3` and `draft` renders add dual-quaternion skinning and a Corrective Smooth ([artwork.md](artwork.md), `smooth_skin`). Each step was judged on close-ups of three bent-knee frames of `backhand_14m19s`: welding plus the render-time smoothing removed the faceting and most of the fold, the weight smoothing softened the bend further. Moving the rig's knee joints to the middle of the leg (they sit 2.7 and 3.7 cm towards the kneecap) made the shin notch on the inside and was dropped.

The back of the calf still collapsed in deep bends. The cause is where the auto-rig hands each knee from thigh to shin: below the joint, not on it. In the rest pose, behind the knee, the shin's share of the weight reaches 50 % 5.4 cm (left) and 5.5 cm (right) below the joint (1.7 and 2.5 cm in front) and climbs from 10 to 90 % over 16.6 and 17.6 cm. In a deep bend the upper calf therefore turns only part of the way with the shin and lags behind it. Naomi's config sets `retarget.leg_weights: band`, which rebuilds each knee's thigh/shin split (`blender/retarget.py` `knee_band`) as a smoothstep across the plane through the knee joint that bisects the thigh and shin directions (`knee_plane`): all thigh `retarget.knee_band_cm` = 5 cm above the plane, all shin 5 cm below it, half and half on it. Every vertex keeps its thigh + shin total, so the hip and ankle blends stay as the auto-rig made them. The band is applied after the limb smoothing, which would blur it again, and `anim/retarget.json` records it under `skin_fix`.

Measured on the moveset posed as the `quest3` studio deforms it, against the same build with the auto-rig's leg weights, the 50 % point sits within 0.1 cm of the joint, front and back. The upper calf (vertices behind the leg 3-8 cm below the knee, compared with where a rigid shin would carry them) is off by a median 1.0 mm instead of 62.9 mm at the worst frame (`backhand_14m19s` frame 40, left knee bent 115°; p95 16.9 against 93.4 mm), and by at most 2.1 mm on the seven deep-bend frames, where the auto-rig's weights gave 13 to 79 mm. The band changes the weights of 10,196 of the 139,998 vertices. On 15 test frames, the vertices whose weights it leaves alone do not move on the skirt or the upper body (arms and face included) and move at most 0.15 cm elsewhere. The bend is now concentrated in a narrower band, so more knee triangles are squeezed below 30 % of their rest area at the worst frame (1,126 against 843). A 7 cm band left 8.6 mm of upper-calf deviation there.

Two SMPL-X variants were tried and dropped. Grafting the leg weights of an SMPL-X body fitted to this mesh onto the bare legs handed the knee over within 2 cm of the joint, but it changed 34,743 vertices, only halved the upper-calf deviation (33.7 mm at the worst frame) and amplified Hi3D's seam on the inside of the leg; moving each knee bone to the fitted body's knee, with either set of weights, moved the skirt by up to 2.3 cm in the hips' frame.

Known issues: the face is the generator's reading of a ~250 px face, close to the art but not its exact features (the art team's own model file would settle that); on Hi3D's mesh the white of the top's armhole stretches onto the upper arm when the arm swings forward (Meshy's auto-weights at the armhole); the ponytail and skirt have no bones of their own and follow the head and thighs.

**Handing it to a game team.** The moveset GLB and FBX are what a game engine imports; the Quest 3 videos only preview them. [game-handoff.md](game-handoff.md) lists what the files hold (280,416 triangles with two 8192 x 8192 maps, 54 joints with `Hips` as the root, 39 actions at 25 fps, the racket as a node under `RightHand`), what a game team usually needs that they lack (a Quest-weight mesh, a root bone, the racket as a prop on a socket, per-shot files, a shots metadata file, clean starts and ends), the caveats to pass on and the questions to ask the team.

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
| `character/hand_rig.json`, `.npz` | with a hand setting on: finger joints, weight bands, palm frame, racket seat and grip per hand (`character/hands.py`; `"ok": false` and the reason for a hand it cannot read) |

Statistics of any GLB from the command line:

```bash
.venv/bin/python -c "from rybakina3d.gltf import glb_stats; print(glb_stats('runs/rybakina/character/rigged.glb'))"
```

The GLB opens in Blender with File > Import > glTF 2.0. Meshy's FBX export of the same rig is `character/mesh/meshy/rigged_character_fbx.fbx`.

## Limitations

- **Face texture resolution.** The whole character shares one 2048 x 2048 base-colour map. Measured from the GLB's UVs, the head (everything above the chin line) covers about 9% of the mapped UV area. That is roughly 0.17 megapixels, about a 415 x 415 texel patch for the whole head, and the face is a fraction of it. Head close-ups and the bust on the character sheet magnify this. The texture also has generator artefacts: skin-coloured patches on the dress, seams across the face and stringy hair. The artwork repaint prompt asks the image model to paint over them (see [artwork.md](artwork.md#paint)); the raw render shows them.
- **No finger bones.** Meshy's auto-rig stops at `LeftHand` / `RightHand`. The hands stay open and stiff, so the grip on the racket is not articulated. On two-handed backhands the left hand follows the recovered left wrist and is not constrained to the handle. The hand settings in [section 4](#4-the-racket) change that for a character whose hand mesh has separate fingers (Naomi's, not Rybakina's); they are on in `configs/naomi.yaml` and off for Rybakina.
- **No face controls.** The rig has no face bones and no blendshapes, so it cannot show expressions. The sheet's four head close-ups are four camera angles instead.
- **No secondary motion.** Hair, ponytail and dress are skinned to the body. There is no cloth or hair simulation.
- **Identity is the image model's interpretation** of the photos. Nothing checks likeness numerically.
- **Not reproducible bit for bit.** The hosted models are non-deterministic. Re-running `stylize` or `mesh` with `--force` produces a different character, and everything downstream (retarget, artwork) has to be redone.
- **Client files required.** `character refs` cannot run without the deck PDF and the screenshot under the exact names above. There is no option to skip them.
