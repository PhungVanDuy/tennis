# Game handoff: the moveset in a game engine

This document is for handing Naomi's moveset (`configs/naomi.yaml`, see [character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant)) to a team that will put it in a game engine. It lists what the moveset files hold, measured from the files themselves; what a game engine team usually needs that the files do not hold yet, with the work each gap would take; the caveats to pass on with the files; and the questions to ask the team before that work starts. The Quest 3 videos ([artwork.md](artwork.md#4-render-profiles-the-quest-3-set)) are previews of the moveset, not engine assets. How each clip was captured and retargeted is in [motion-pipeline.md](motion-pipeline.md#11-retarget-smpl-motion-onto-the-character-rig-in-blender). Output paths are relative to the variant's folder `runs/rybakina/variants/naomi/`, except `curation.json` and `clips/`, which are shared at the run root `runs/rybakina/`.

## What would be sent

| File | Size | What it is |
|---|---|---|
| `anim/naomi_moveset.glb` | 17.41 MB | glTF 2.0 binary from Blender's glTF exporter (Blender 5.2.2 LTS): the skinned character, the racket and the 39 animations; Y up, metres, unit scale |
| `anim/naomi_moveset.fbx` | 33.10 MB | binary FBX 7.4 of the same scene: every action baked, no leaf bones, textures embedded, `-Z` forward, `Y` up |
| `anim/retarget.json` | 0.20 MB | the build report: the bone mapping, the racket seat, the skin fixes, and per clip `fps`, `frames`, `contact_index`, the foot planting and the hand metrics |
| `anim/verify.json` | 0.01 MB | the re-import check in a fresh Blender: action names, joint replay error, the racket's parent and drift, the finger bones in the GLB and the FBX |
| `artwork/motion/render_quest3/<clip>.mp4` | 0.91 to 2.99 MB each, rendered with the hands | previews only: one camera, one eye, render only ([artwork.md](artwork.md#4-render-profiles-the-quest-3-set)) |

The GLB's 17.41 MB are 10.97 MB of mesh data, 3.51 MB of textures, 1.55 MB of animation and 1.37 MB of glTF JSON. The animations are the small part: what makes the files heavy is the mesh and its maps.

## What the files hold

Measured with `gltf.glb_stats` and from the GLB's JSON chunk ([Checking the files](#checking-the-files)).

### Mesh, materials, textures

| | Character (node `char1`, mesh `geometry_0`) | Racket (node `Racket`) |
|---|---|---|
| Triangles | 280,000 | 416 |
| Vertices in the GLB | 145,913 (139,998 after the weld in Blender; the exporter splits vertices at UV seams) | 781 |
| Primitives, materials | 1, 1 (`pbr_material`, double-sided) | 3, 3 (`Racket_Frame`, `Racket_Grip`, `Racket_Strings`, the strings alpha-blended) |
| Textures | base colour and metallic-roughness, each 8192 x 8192 JPEG (2.09 and 1.42 MB in the GLB); no normal map | the strings, a 256 x 256 PNG grid; frame and grip are flat colours (`character.racket_colors`) |
| Skinning | one skin over all 54 joints, at most 4 joints per vertex (`JOINTS_0` / `WEIGHTS_0` only) | not skinned: a child node of the `RightHand` joint with a fixed local transform and no keys |
| Morph targets | none | none |

In all: 2 meshes, 4 materials, 3 images, 1 skin, 280,416 triangles and 146,694 vertices. The skin weights carry the fixes that make the knees bend as one skin (welded vertices, smoothed limb weights, the knee band) and the finger weights; they were computed on this mesh ([character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant), `retarget.json` `weld` and `skin_fix`).

### Skeleton

54 joints: the Meshy auto-rig's 24 and 15 finger bones per hand (`Thumb`, `Index`, `Middle`, `Ring`, `Pinky`, three each, `LeftHandIndex1` and so on). `Hips` is the skeleton's root; it sits directly under the scene's `Armature` node, which carries no transform and is not a joint. There is no root bone at ground level.

| Chain | Bones (parent to child) |
|---|---|
| Spine and head | `Hips` → `Spine02` → `Spine01` → `Spine` → `neck` → `Head` → `head_end`, `headfront` (the spine is numbered downwards: `Spine02` is the lowest) |
| Arms | `Spine` → `LeftShoulder` → `LeftArm` → `LeftForeArm` → `LeftHand` → fingers; the same on the right |
| Legs | `Hips` → `LeftUpLeg` → `LeftLeg` → `LeftFoot` → `LeftToeBase`; the same on the right |

The armature's centimetre scale was baked into the bones on import, so the skeleton is at unit scale, standing on the floor with the hips 1.03 m up in the rest pose, which is an A-pose. `head_end` and `headfront` are end and helper bones that keep their rest pose. There are no face bones and no blendshapes; the ponytail and the skirt have no bones of their own.

### Animations

39 actions named by clip id (`<stroke>_<MM>m<SS>s`, the label and the source time of the contact): 12 serves, 12 forehands, 12 backhands, 1 slice, 2 overheads. 2,531 frames at 25 fps, the broadcast rate: 58 frames per forehand, backhand, slice and overhead, 75 to 81 per serve. Every action has translation, rotation and scale channels on all 54 joints (162 channels): rotations keyed linearly at every frame, except where a bone holds one pose through the clip and carries a single key (the right hand's fingers in their grip in every clip, the left hand's in the 6 clips where they never close, `head_end` and `headfront` at rest); the translation of `Hips` keyed linearly at every frame; every other translation and every scale constant, stored as steps. `verify.json`: every clip replays its joint track within 0.008 mm in the re-imported GLB, and every clip keys all 30 finger bones.

**Root motion is in `Hips`.** Each clip starts with the hips over the origin (x = z = 0 at frame 0, Y up) and moves her with the stroke: 0.37 to 4.49 m horizontally within a clip (median 1.75 m). The heading is the clip's own, because GVHMR's world follows the broadcast camera (`blender/retarget.py`): the hips' heading at frame 0 spans 127° across the 39 clips (-102° to +25° from the GLB's +Z). Where on the court she stands is not in the moveset.

The contact frame of each clip is `retarget.json` `clips.<id>.contact_index` (the frame of racket-ball contact, refined by `curate`). The Quest 3 videos play the same keys interpolated to 60 fps; the files hold the 25 fps keys only.

### The racket

`blender/racket.py` builds it inside the retarget: 68.5 cm long, 416 triangles. In the GLB its local frame has the origin at the butt cap, the shaft along -Z to the tip and the string-face normal along +Y (`blender/racket.py`'s +Y and +Z, turned by the exporter's Y up). It is seated on the right palm (`retarget.grip: palm`) and parented to `RightHand` at a fixed local offset of (-4.1, 5.7, -4.0) cm in the GLB's `RightHand` node frame; the right hand's keys carry a constant 28° offset (`retarget.json` `hands.seat.offset_deg`) so the racket points where the fixed grip pointed. `verify.json` records its parent as `BONE:RightHand`, its rest pose within 0.02° and no drift on the hand in any clip.

## What a game team usually needs

| Need | What the files hold | Proposed work |
|---|---|---|
| A Quest-weight character | 280,416 triangles and two 8192 x 8192 maps. Quest 3 standalone hero characters are typically tens of thousands of triangles with 2K textures | Decimate the mesh to the team's budget, bake the high mesh's detail into a normal map and re-bake base colour and metallic-roughness at 2K, then skin the low mesh to the same 54 joints (transfer the weights from this mesh and redo the knee band and the finger weights on it). The animations key only the joints, so they play unchanged on any mesh skinned to this skeleton and rest pose. A PC VR build can keep more of the mesh |
| A root bone and a root-motion choice | `Hips` is the root; the clip's travel and heading are in its translation and rotation | Add a `root` bone on the floor under `Armature` as the parent of `Hips`, then either extract root motion (the horizontal path and the heading move onto `root`, `Hips` keeps the rest) or make the clips in place (zero the travel and give it in the shots file). Unreal reads root motion from the skeleton's root bone; Unity's Humanoid derives it from the body's centre of mass and can work without one. Either way, turn every clip to one start heading |
| The racket as a prop with a socket | the racket is a node of the character file under `RightHand` and imports as part of it | Export the racket as its own file in its local frame, and the hand offset above as a named socket on `RightHand` (an Unreal skeletal-mesh socket, a Unity child transform); the character file then carries no racket |
| Per-shot files, or clean take names | one GLB and one FBX hold the character, the racket and all 39 actions. The GLB's animations are named by clip id; the FBX's takes carry Blender's object prefix (a re-import in Blender reads them as `Armature\|Armature\|<clip>`, `verify.json` `fbx_actions`) | One character FBX (mesh, skin, skeleton, no animation) and one animation-only FBX per shot, named by clip id or the team's convention, or the single FBX with the prefix taken off the take names. A glTF pipeline can keep the single GLB |
| A shots metadata file | spread over reports: `retarget.json` (`fps`, `frames`, `contact_index`), `curation.json` (`label_ok`, `reason`), the clip id (stroke label, source time). Nothing records the racket face at contact or where on the court a clip starts and ends | One file per moveset (named apart from the run's `shots.json`, which lists the footage's camera cuts): per clip the id, stroke, flagged and why, frames, fps, contact frame and time, the racket face's centre, normal and velocity at contact (read off the GLB), the start and end court positions and headings (mapped through each clip's court calibration, `clips/<id>/camera.npz`; the side-view clip has none) and the caveats below |
| Clean starts and ends | each clip is a window of broadcast footage around one contact (2.2 s before to 1.0 s after for serves, 1.3 s before to 1.0 s after otherwise, `select.window_s`), so it starts and stops mid-motion. There is no ready or idle pose, no transition and no loop in the moveset | Author a ready/idle loop and the blends (or short transition clips) from each clip's first and last frames into it, and trim or extend where the game's timing needs it. This is animation authoring, not capture |
| Frame rate | 25 fps keys | none: engines interpolate between keys at their own rate |

## Caveats to pass on

- **The racket's direction is not from the footage.** The right hand's grip orientation is a constant offset that keeps the racket where the fixed grip aims it; nothing measures the racket in the video. On the 5 s experiment's clip its 2D direction differed from the video's by 40 to 48°. Racket face and forearm roll are not measured either.
- **Some stroke labels contradict the motion.** 15 of the 39 clips fail the stroke check in 3D (the 6 flagged "serves" are most likely returns of serve), [README](../README.md#results) and `curation.json` (`label_ok`, `reason`). They stay in the moveset under their label; the shots file should flag them.
- **The left hand's hold is inferred.** When it takes the handle is read from GVHMR's wrist distance, and the grip is one pose solved on the handle, not measured per stroke. It is on the racket on 1,382 of the 2,531 frames, never in `forehand_27m36s` and `overhead_14m57s`; the worst frame has 89 of the left hand's 3,770 vertices inside the racket (`serve_12m47s` frame 57), more than 30 on 5 frames in 4 clips ([character.md](character.md#5-a-character-drawn-by-the-art-team-the-naomi-variant)).
- **Two clips end on a guess.** The player was not tracked on frames 40 to 57 of `forehand_26m50s` and 42 to 57 of `backhand_10m16s`; the motion there is GVHMR's (`retarget.json` clip warnings). Both clips are among the flagged ones.
- **Feet.** Planted frames are pinned through the root, without heel/toe IK, so the lowest foot can dip about 3 cm into the court (5th percentile).
- **Build-machine paths.** `anim/retarget.json` records absolute paths of the build machine in `rig.path`, `outputs` and `settings.hand_rig`; `anim/job.json` and `anim/blender.log` hold more, and the FBX's texture file-name fields name the build folder (the textures themselves are embedded). The GLB holds none. Strip them, or re-export, before anything leaves the project.

## Questions for the game team

1. **Engine:** Unity, Unreal or another (and which version)? This decides FBX or glTF, the take naming and how root motion is set up.
2. **Platform:** Quest 3 standalone or PC VR? This decides the triangle and texture budget, and so how far the mesh is decimated.
3. **Root motion or in place?** With root motion, which bone and axis convention do they expect?
4. **Their target skeleton.** Unity's Humanoid can take this rig as it is (`Spine02`, `Spine01`, `Spine` map to Spine, Chest and Upper Chest; the fingers map too); its avatar setup can enforce a T-pose over the A-pose. Unreal needs an IK Retargeter from this skeleton to theirs, or uses this skeleton as its own.
5. **Budgets:** triangles, texture size, materials and joints per vertex for a hero character.
6. **The racket:** this one as a prop on a socket, or theirs?
7. **Which clips:** all 39, or only the 24 that pass the stroke check?

## Checking the files

```bash
.venv/bin/python -c "from rybakina3d.gltf import glb_stats; print(glb_stats('runs/rybakina/variants/naomi/anim/naomi_moveset.glb'))"
.venv/bin/python -c "import json; d=json.load(open('runs/rybakina/variants/naomi/anim/retarget.json')); [print(k, c['fps'], c['frames'], c['contact_index']) for k, c in d['clips'].items()]"
```

The GLB opens in Blender with File > Import > glTF 2.0, and in any glTF viewer; the review site's Artwork tab has download buttons for both files ([review-site.md](review-site.md#character-variants)).
