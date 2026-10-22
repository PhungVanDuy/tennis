"""Stage `character face`: the art team's face, projected onto an art-team character's rig (workstation).

Image-to-3D textures the head from the front view, where the face is ~250 px tall and lit from the front, so the
rig's face comes out pale and blotchy next to the art. Their head renders show the same face at ~900 px. This
stage evens out the rig's skin, then aligns that front headshot to the rig's own face and bakes it in:

0. `even_skin`: the front light baked into the texture leaves face, neck and chest far lighter than the arms and
   legs (Lab L ~178 against ~130 here). Skin texels (hue near the rig's skin, soft edge) lighter than the body's
   median skin are pulled towards it, keeping a SKIN_KEEP share of their own contrast.
1. blender/face_project.py (render): the rest-pose head from the front, orthographic and unlit, at SIZE px.
2. YuNet (OpenCV FaceDetectorYN, weights/yunet/) finds eyes, nose tip and mouth corners in that render and in the
   headshot; a similarity transform (rotation, scale, shift) maps the headshot onto the render.
3. The mask: an oval from those landmarks (brows to chin, cheek to cheek), minus the render's visor, hair and
   background (only skin-coloured pixels of the render take the projection), feathered so it fades into the
   rig's own skin. Only the headshot's detail is projected (`delight`): its low frequencies (their studio light,
   the broad skin tone) are replaced by the rig's own, so the features change and no light or tone seam shows.
4. blender/face_project.py (bake): that image through that camera onto the mesh (front-facing corners only),
   baked into a base-colour texture of at least BAKE_MIN px, which replaces the original in the GLB.

Outputs: character/face/{evened.glb, render.png, camera.json, headshot_aligned.png, mask.png, basecolor.png,
face.json};
character/rigged.glb gets the new texture (the untouched rig stays in character/mesh/<generator>/).
Run after `character mesh` (which puts the untouched rig back) and before `retarget`.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np

from ..blender import SCRIPTS_DIR
from ..config import PROJECT_ROOT, Config
from ..gltf import base_color_images, replace_image
from ..util import log, read_json, step, write_json
from .mesh import _blender_local

SIZE = 2048                     # px of the face render: its eye distance ~ the headshot's (~280 px)
BAKE_MIN = 4096                 # px: a smaller base colour is baked up to this size (2048 -> 4096), larger ones kept
YUNET = PROJECT_ROOT / "weights" / "yunet" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
HEADSHOT = "art_head_front"
# the oval, in eye distances from the eye-mouth midpoint: half-width, up to the brows, down past the chin
OVAL_HALF_W, OVAL_UP, OVAL_DOWN = 0.95, 1.05, 1.0
FEATHER = 0.18                  # of the eye distance: the mask's soft edge
ART_INSET = 0.2                 # of the eye distance: the mask stays this far inside the headshot's own skin, so
                                # its face outline (an edge the detail pass would carry over) never projects
DELIGHT_SIGMA = 0.2             # of the eye distance: the art's detail is what lies below this scale
BASE_SIGMA = 0.6                # of the eye distance: the rig's shading under it (wider: its blotches smooth out)
SKIN_AB_RADIUS = (12.0, 20.0)   # Lab a/b distance from the rig's skin hue: full weight inside, none beyond
SKIN_KEEP = 0.35                # of a pale skin texel's lightness above the body median that survives


def landmarks(img: np.ndarray) -> np.ndarray:
    """(5, 2) right eye, left eye, nose tip, right and left mouth corner (image coordinates) of the best face."""
    if not YUNET.exists():
        raise FileNotFoundError(f"no {YUNET}: download {YUNET_URL} there")
    s = 1000 / max(img.shape[:2])
    small = cv2.resize(img[..., :3], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (small.shape[1], small.shape[0]), 0.3)
    _, faces = det.detect(np.ascontiguousarray(small))
    if faces is None or not len(faces):
        raise RuntimeError("no face found")
    best = faces[np.argmax(faces[:, -1])]
    return best[4:14].reshape(5, 2) / s


def oval(points: np.ndarray, shape: tuple[int, int], scale: float = 1.0) -> np.ndarray:
    """Filled face oval (float 0/1) from the landmarks, `scale` times its size."""
    eyes, mouth = points[:2].mean(0), points[3:5].mean(0)
    d = np.linalg.norm(points[0] - points[1])
    centre = (eyes + mouth) / 2
    up = (eyes - mouth) / np.linalg.norm(eyes - mouth)
    angle = np.degrees(np.arctan2(-up[0], up[1]))      # the oval's vertical axis along mouth -> eyes
    shift = up * d * (OVAL_UP - OVAL_DOWN) / 2
    axes = (int(OVAL_HALF_W * d * scale), int((OVAL_UP + OVAL_DOWN) / 2 * d * scale))
    m = np.zeros(shape, np.uint8)
    cv2.ellipse(m, tuple(int(v) for v in centre + shift), axes, angle, 0, 360, 1, -1)
    return m.astype(np.float32)


def skin(render: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Pixels of the render whose colour is close to the face's own skin (no visor, hair or background)."""
    lab = cv2.cvtColor(render[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[sample > 0], axis=0)
    dist = np.linalg.norm((lab - ref) * (0.6, 1.0, 1.0), axis=-1)   # lightness counts less than hue
    ok = (dist < 28) & (render[..., 3] > 128 if render.shape[-1] == 4 else True)
    ok = cv2.morphologyEx(ok.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.morphologyEx(ok, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)).astype(np.float32)


def delight(art: np.ndarray, base: np.ndarray, base_mask: np.ndarray, sigma: float, base_sigma: float) -> np.ndarray:
    """The art's detail on the base's broad shading: base low-pass at `base_sigma` (taken over `base_mask` only, so
    visor and hair do not bleed in) + (art - art low-pass at `sigma`), in float RGB."""
    a, b = art.astype(np.float32), base.astype(np.float32)
    m = base_mask.astype(np.float32)
    weight = np.maximum(cv2.GaussianBlur(m, (0, 0), base_sigma), 1e-3)[..., None]
    low_b = cv2.GaussianBlur(b * m[..., None], (0, 0), base_sigma) / weight
    out = low_b + a - cv2.GaussianBlur(a, (0, 0), sigma)
    return np.clip(out, 0, 255).astype(np.uint8)


def even_skin(tex: np.ndarray) -> tuple[np.ndarray, dict]:
    """The skin's baked front light taken out: skin texels lighter than the body skin's median lightness are
    compressed towards it (hair, lips and shadows, darker than that, keep their values; cloth is another hue)."""
    lab = cv2.cvtColor(tex, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, ab = lab[..., 0], lab[..., 1:]
    # the skin hue: the most common warm, mid-light chroma in the texture (a, b > 132: away from white cloth)
    warm = (ab[..., 0] > 135) & (ab[..., 1] > 135) & (L > 60) & (L < 235)
    hue = np.median(ab[warm], axis=0)
    dist = np.linalg.norm(ab - hue, axis=-1)
    w = np.clip((SKIN_AB_RADIUS[1] - dist) / (SKIN_AB_RADIUS[1] - SKIN_AB_RADIUS[0]), 0, 1) * (L > 60)
    body_l = float(np.median(L[(dist < SKIN_AB_RADIUS[0]) & (L > 60)]))
    excess = np.maximum(L - body_l, 0)
    lab[..., 0] = L - w * excess * (1 - SKIN_KEEP)
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out, {"skin_hue_ab": [round(float(v), 1) for v in hue], "body_lightness": round(body_l, 1),
                 "skin_share": round(float((w > 0.5).mean()), 3)}


def _image(glb: Path, index: int) -> bytes:
    from ..gltf import _read_glb

    g, binary = _read_glb(glb)
    view = g["bufferViews"][g["images"][index]["bufferView"]]
    return binary[view.get("byteOffset", 0) : view.get("byteOffset", 0) + view["byteLength"]]


def _blender(job: dict, path: Path) -> None:
    write_json(path, job)
    cmd = [_blender_local(), "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(SCRIPTS_DIR / "face_project.py"), "--", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"blender face_project failed:\n{res.stdout[-3000:]}\n{res.stderr[-2000:]}")


def run(cfg: Config) -> dict:
    run_ = cfg.run
    report = read_json(run_.mesh_report)
    primary = report["primary"]
    source = run_.mesh_dir / primary / report["generators"][primary]["files"]["rigged_character_glb"]
    refs = {r["name"]: run_.refs_dir / r["path"] for r in read_json(run_.refs)}
    out = run_.character_dir / "face"
    out.mkdir(parents=True, exist_ok=True)
    images = base_color_images(source)
    if len(images) != 1:
        raise RuntimeError(f"expected one base-colour image in {source}, found {images}")
    tex = cv2.imdecode(np.frombuffer(_image(source, images[0]), np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    evened, skin_stats = even_skin(np.ascontiguousarray(tex))
    ok, png = cv2.imencode(".png", evened[..., ::-1])
    evened_glb = out / "evened.glb"
    replace_image(source, evened_glb, images[0], png.tobytes())
    log(f"skin evened: body lightness {skin_stats['body_lightness']}, {skin_stats['skin_share']:.0%} of the texture")

    with step("face render (Blender, rest pose, unlit)"):
        _blender({"mode": "render", "glb": str(evened_glb), "size": SIZE, "png": str(out / "render.png"),
                  "out": str(out / "camera.json")}, out / "job_render.json")
    render = cv2.cvtColor(cv2.imread(str(out / "render.png"), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
    head = cv2.cvtColor(cv2.imread(str(refs[HEADSHOT])), cv2.COLOR_BGR2RGB)
    p_render, p_head = landmarks(cv2.cvtColor(render, cv2.COLOR_RGBA2BGR)), landmarks(head[..., ::-1])
    M, _ = cv2.estimateAffinePartial2D(p_head, p_render, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("could not align the headshot to the render")
    residual = float(np.abs(cv2.transform(p_head[None], M)[0] - p_render).max())
    aligned = cv2.warpAffine(head, M, (SIZE, SIZE), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)

    d = float(np.linalg.norm(p_render[0] - p_render[1]))
    face = oval(p_render, (SIZE, SIZE))
    own_skin = skin(render, face * (render[..., 3] > 128))
    art_skin = skin(np.dstack([aligned, np.full(aligned.shape[:2], 255, np.uint8)]), face)
    inset = int(ART_INSET * d) | 1
    art_skin = cv2.erode(art_skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset, inset)))
    aligned = delight(aligned, render[..., :3], own_skin, DELIGHT_SIGMA * d, BASE_SIGMA * d)
    own_skin = own_skin * art_skin
    blur = int(FEATHER * d) | 1
    mask = cv2.GaussianBlur(face * own_skin, (blur, blur), 0)
    mask_png = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "headshot_aligned.png"), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "mask.png"), mask_png)

    with step(f"face bake (Cycles, base colour >= {BAKE_MIN} px)"):
        _blender({"mode": "bake", "glb": str(evened_glb), "size": SIZE, "image": str(out / "headshot_aligned.png"),
                  "mask": str(out / "mask.png"), "min_size": BAKE_MIN, "png": str(out / "basecolor.png")},
                 out / "job_bake.json")
    replace_image(source, run_.rig, images[0], (out / "basecolor.png").read_bytes())

    result = {"source": str(source.relative_to(run_.root)), "rig": str(run_.rig.relative_to(run_.root)),
              "headshot": str(refs[HEADSHOT].relative_to(run_.root)), "size": SIZE, "bake_min": BAKE_MIN,
              "eye_distance_px": round(d, 1), "landmark_residual_px": round(residual, 1),
              "delight_sigma_px": round(DELIGHT_SIGMA * d, 1), "skin": skin_stats, "transform": np.round(M, 5).tolist(),
              "mask_area_px": int((mask > 0.5).sum())}
    write_json(out / "face.json", result)
    log(f"face projected: landmark residual {residual:.1f} px at eye distance {d:.0f} px → {run_.rig}")
    return result
