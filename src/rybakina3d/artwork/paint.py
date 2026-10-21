"""Stage `artwork paint`: repaint each render in the reference sheet's style with GPT Image 2.5 Sunburst (fal).

Every paintable slot of the layout goes to the edit model as three images: the render composited over the paper
colour (POSE / COMPOSITION source), a crop of the same kind of figure from naomi_style.png (STYLE) and the
approved A-pose hero (IDENTITY / OUTFIT). The model returns the figure on a transparent background: compared on
the hero slot, the transparent and the paper-coloured (opaque) variants kept the pose about equally well (aligned
IoU 0.94 vs 0.92), but only the transparent one gives a clean cut-out; on paper the white visor, dress and shoes
barely separate from the background and the mask tears there.

The pose is the point of the sheet (it is her captured motion), so a repaint is only accepted when its silhouette
matches the render matte: IoU >= ACCEPT_IOU after aligning the two by their bounding boxes. A miss is retried once
with a stricter prompt (RETRY_CANDIDATES samples, the best match counts); after that the slot falls back to the render.
Misses are mostly a slightly bulkier body or a racket a little longer than the render's, which costs thin poses
(arms out, a serve) far more IoU than compact ones. fal's output moderation rejects the odd sample of a figure in
a short dress lunging (content_policy_violation); that attempt is simply made again.

The rig has no finger bones, so racket poses also ask for hands closed on the grip (GRIP; on the two-handed
backhand both hands on the handle), a change inside the silhouette. The cache key (`fingerprint`) is the render +
matte bytes, the model and the slot's own prompt text: editing one kind's prompt repaints only that kind, and
`compose` refuses a repaint whose fingerprint no longer matches the current render. A change to how the cut-out is
made (CUT_VERSION) re-cuts the accepted repaints without calling the model.

  painted/<slot>.png        the accepted repaint as the model returned it
  painted/<slot>_cut.png    that repaint on the render's canvas (a whole figure scaled onto the render's feet line,
                            see `similarity`) over the render's soft ground shadow; this is what `compose` lays out
  painted/attempts/         every attempt; painted/inputs/ the exact images sent
  painted/<slot>.json       per slot: accepted, IoU, attempts, model, prompt kind + grip, fingerprint (the record)
  painted/paint.json        all slots in one report

The renders index (renders/renders.json, written by `artwork render`) and the paper colour are shared with
`compose` from here.
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .. import fal
from ..config import PROJECT_ROOT, Config
from ..util import log, read_json, step, write_json
from .layout import SLOTS, STYLE_REFERENCE, Slot

ACCEPT_IOU = 0.85
WORKERS = 4
RETRY_CANDIDATES = 3         # the strict retry samples this many repaints and keeps the best match
BACKGROUND = "transparent"
MIN_PIXELS = 1_100_000        # GPT Image wants roughly 0.66-8.3 MP; small slots (faces) are sent upscaled
MAX_PIXELS = 4_000_000
MAX_EDGE = 3840

# Where the same kind of figure sits on the reference sheet (normalised x0, y0, x1, y1).
STYLE_CROPS = {
    "standing": (0.058, 0.029, 0.293, 0.457),    # front + back turnaround
    "hero": (0.336, 0.035, 0.617, 0.692),        # the big two-handed backhand
    "portrait": (0.746, 0.035, 0.996, 0.400),    # 3/4 head-and-shoulders fading into the paper
    "action": (0.578, 0.400, 1.000, 0.768),      # the two smaller "Details" poses
    "head": (0.613, 0.807, 0.967, 0.956),        # the row of head close-ups
}

STYLE = (
    "a semi-realistic, hand-painted game character illustration: soft diffuse daylight, gentle painterly brushwork "
    "with visible soft strokes, warm natural skin tones with soft shading, soft cloth folds, refined facial features, "
    "a slightly muted warm palette"
)
IDENTITY = (
    "her face, light-blonde hair tied back in a low ponytail, white visor, white sleeveless tennis dress with "
    "dark-green trim at the collar and hem, white wristbands, white socks and white tennis shoes"
)
KIND_NOTES = {
    "standing": "It is a turnaround view: if Image 1 shows her from behind, paint her back (ponytail, back of the "
                "dress, calves, heels), not her face.",
    "hero": "She wears white tennis shorts under the dress.",
    "portrait": "It is a head-and-shoulders portrait cut by the bottom edge of the frame: keep exactly that crop.",
    "action": "",
    "head": "It is a head close-up cut by the frame: keep exactly that crop, head size and angle.",
}
# The rig has no finger bones: its hands are stiff open paddles, and on a two-handed backhand the left one only rests
# near the right wrist. The repaint closes them on the grip, a change inside the silhouette.
GRIP = {
    "two_handed": "The 3D rig has no finger bones, so Image 1's hands are stiff and open. Paint them gripping: this "
                  "is a two-handed backhand, so both hands hold the racket handle, the right hand at the bottom of the "
                  "grip and the left hand directly above it, fingers wrapped around the handle.",
    "racket": "The 3D rig has no finger bones, so Image 1's hands are stiff and open. Paint the right hand's fingers "
              "wrapped around the racket grip; the free hand stays where it is, relaxed and natural.",
}
STRICT = (
    "STRICT: a previous attempt drifted from Image 1's pose. Trace Image 1's outline exactly: the same head "
    "position and tilt, the same arm and leg angles, the same hand and foot positions, the same racket angle and "
    "length, the same crop and scale. Change only how it is painted, nothing about the shape."
)


def prompt(kind: str, strict: bool = False, grip: str = "") -> str:
    return "\n".join(filter(None, [
        "Repaint Image 1 as a finished hand-painted character illustration.",
        "Image 1 is the POSE AND COMPOSITION SOURCE: a 3D render of our game character. Keep exactly its pose, body "
        "proportions, camera angle, perspective, silhouette, framing, position and size in the frame, and the "
        "racket's position, angle and shape (if there is one). Every limb, the head and the racket must stay "
        "exactly where they are in Image 1.",
        f"Image 2 is the STYLE REFERENCE (a different person): copy only its painting style, {STYLE}. Do not copy "
        "that person's face, hair, outfit or logos.",
        f"Image 3 is the IDENTITY AND OUTFIT REFERENCE: the same character as Image 1: {IDENTITY}.",
        "Image 1 shows texture glitches of the 3D model (skin-coloured patches on the dress, seams across the face, "
        "stringy hair): paint them as they should be, a clean white dress, smooth skin, soft hair, without changing "
        "any outline.",
        KIND_NOTES[kind],
        GRIP.get(grip, ""),
        STRICT if strict else "",
        "Output: only this one figure on a fully transparent background. No ground, no cast shadow, no text, no "
        "logos, no frame, no extra objects or people.",
    ]))


def kind_of(slot: Slot) -> str:
    if slot.framing == "head":
        return "head"
    if slot.framing == "bust":
        return "portrait"
    if slot.pose == "idle":
        return "standing"
    return "hero" if slot.name == "hero" else "action"


def grip_of(slot: Slot, backhand: str = "two_handed") -> str:
    """Which GRIP note the slot's repaint gets: none without a racket (the idle poses)."""
    if slot.pose == "idle" or not isinstance(slot.pose, dict):
        return ""
    return "two_handed" if slot.pose.get("stroke") == "backhand" and backhand == "two_handed" else "racket"


# --- shared with compose -------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def reference() -> Image.Image:
    return Image.open(PROJECT_ROOT / STYLE_REFERENCE).convert("RGB")


PAPER_FALLBACK = (251, 246, 240)   # what paper_colour() measures on the client's sheet, which is not in the repo


@lru_cache(maxsize=1)
def paper_colour() -> tuple[int, int, int]:
    """The reference's paper: the most common colour among its bright, low-chroma pixels (it fills the sheet).
    Without the reference (a fresh clone: it is the client's file and kept out of git) its measured value."""
    if not (PROJECT_ROOT / STYLE_REFERENCE).exists():
        return PAPER_FALLBACK
    px = np.asarray(reference()).reshape(-1, 3).astype(int)
    bright = px[(px.min(1) > 200) & (px.max(1) - px.min(1) < 24)]
    bins = bright // 4
    key = bins[:, 0] * 4096 + bins[:, 1] * 64 + bins[:, 2]
    values, counts = np.unique(key, return_counts=True)
    return tuple(int(v) for v in np.median(bright[key == values[np.argmax(counts)]], axis=0))


@dataclass
class Figure:
    """One rendered figure (renders.json row): RGBA with soft shadow, figure-only matte, where she stands."""

    name: str
    image: Path
    matte: Path | None = None
    size: tuple[int, int] = (0, 0)
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)    # figure (matte) bbox in px, x1/y1 exclusive
    ground_y: float = 0.0                              # px: the floor under her feet (below them in a jump)
    px_per_m: float | None = None                      # render px per metre at the figure (camera lens/distance)
    meta: dict = field(default_factory=dict)


def mask_of(path: Path) -> np.ndarray:
    """Alpha (or grey) of an image as float32 0..1."""
    im = Image.open(path)
    a = np.asarray(im.getchannel("A") if "A" in im.getbands() else im.convert("L"), np.float32)
    return a / 255.0


def _bbox(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _figure(name: str, row: dict, run_root: Path, png: Path, matte: Path) -> Figure | None:
    png = run_root / row["png"] if row.get("png") else png
    matte = run_root / row["matte"] if row.get("matte") else matte
    if not png.exists():
        return None
    matte = matte if matte.exists() else None
    with Image.open(png) as im:
        w, h = im.size
    # extent from the pixels: the matte (else the opaque part of the RGBA, which leaves out the soft shadow)
    m = mask_of(matte) if matte else mask_of(png)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    bb = _bbox(m > 0.5) or (0, 0, w, h)
    ground = row.get("ground_y", row.get("feet_y"))
    cam = row.get("camera") or {}
    ppm = None
    if all(isinstance(cam.get(k), (int, float)) and cam[k] > 0 for k in ("lens_mm", "sensor_height_mm", "distance_m")):
        ppm = cam["lens_mm"] / cam["sensor_height_mm"] * h / cam["distance_m"]  # vertical sensor fit
    return Figure(name, png, matte, (w, h), bb, float(ground) if ground is not None else float(bb[3]), ppm, row)


SHADOW_FLOOR = 0.05     # render alpha outside the figure below this is denoiser haze, not shadow
CUT_VERSION = 4         # bump when `cutout` changes: accepted repaints are re-cut without calling the model again


def render_rgba(fig: Figure) -> Image.Image:
    """The render with a clean background: the figure (matte + its rim) as rendered, and outside it only the
    contact shadow, i.e. alpha above SHADOW_FLOOR in the lower part of a whole figure. Cycles leaves faint
    blotches of alpha all over the transparent film, which would print as grey boxes on the paper."""
    im = Image.open(fig.image).convert("RGBA")
    if fig.matte is None:
        return im
    rgba = np.asarray(im).copy()
    h, w = rgba.shape[:2]
    m = mask_of(fig.matte)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    near = cv2.dilate((m > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    a = rgba[..., 3].astype(np.float32) / 255
    shadow = np.clip((a - SHADOW_FLOOR) / (1 - SHADOW_FLOOR), 0, 1)
    if fig.meta.get("framing", "full") == "full":
        top, ground = fig.bbox[1], max(fig.ground_y, fig.bbox[3])
        y0, ramp = top + 0.55 * (ground - top), 0.06 * (ground - top) + 1
        shadow *= np.clip((np.arange(h, dtype=np.float32)[:, None] - y0) / ramp, 0, 1)
    else:  # bust / head: no floor
        shadow[:] = 0
    rgba[..., 3] = np.where(near, rgba[..., 3], (shadow * 255).round()).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def load_renders(cfg: Config) -> tuple[dict[str, Figure], dict[str, Figure]]:
    """renders.json ({"slots": {slot: row}, "atlas": {clip: row}}) → Figures; files on disk fill any gaps."""
    root, run_root = cfg.run.artwork_renders, cfg.run.root
    data = read_json(root / "renders.json") if (root / "renders.json").exists() else {}
    slot_rows, atlas_rows = data.get("slots") or {}, data.get("atlas") or {}
    slots: dict[str, Figure] = {}
    for s in SLOTS:
        f = _figure(s.name, slot_rows.get(s.name, {}), run_root, root / f"{s.name}.png", root / f"{s.name}_matte.png")
        if f:
            slots[s.name] = f
    atlas: dict[str, Figure] = {}
    on_disk = sorted(p.stem for p in (root / "atlas").glob("*.png") if not p.stem.endswith("_matte"))
    for cid in list(atlas_rows) + [c for c in on_disk if c not in atlas_rows]:
        f = _figure(cid, atlas_rows.get(cid, {}), run_root, root / "atlas" / f"{cid}.png",
                    root / "atlas" / f"{cid}_matte.png")
        if f:
            atlas[cid] = f
    return slots, atlas


# --- silhouette check ----------------------------------------------------------------------------------
def clean_mask(m: np.ndarray) -> np.ndarray:
    """Binary figure mask without specks or pinholes: open, close, keep the parts near the largest, fill holes."""
    m = m.astype(np.uint8)
    k = max(3, int(round(min(m.shape) / 300)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3 * k, 3 * k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.02 * big]
        m = np.isin(lab, keep).astype(np.uint8)
    # holes = background not connected to the image border (racket head, the gap of joined hands)
    pad = np.pad(m, 1)
    flood = pad.copy()
    cv2.floodFill(flood, np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return (m.astype(bool) | holes)


def painted_mask(img: Image.Image) -> np.ndarray:
    """Foreground of a repaint: its alpha when it is a cut-out, else distance from the background colour."""
    a = np.asarray(img.convert("RGBA"))
    alpha = a[..., 3]
    if (alpha < 128).mean() > 0.02:
        return clean_mask(alpha >= 128)
    rgb = a[..., :3].astype(np.int32)
    border = np.concatenate([rgb[:3].reshape(-1, 3), rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    return clean_mask(np.abs(rgb - bg).sum(-1) > 36)


def aligned_iou(mask: np.ndarray, ref: np.ndarray) -> float:
    """IoU of two figure masks after mapping mask's bounding box onto ref's (position + scale drift removed)."""
    bm, br = _bbox(mask), _bbox(ref)
    if bm is None or br is None:
        return 0.0
    crop = mask[bm[1]:bm[3], bm[0]:bm[2]].astype(np.uint8)
    warped = cv2.resize(crop, (br[2] - br[0], br[3] - br[1]), interpolation=cv2.INTER_NEAREST).astype(bool)
    r = ref[br[1]:br[3], br[0]:br[2]]
    union = (warped | r).sum()
    return float((warped & r).sum() / union) if union else 0.0


def _shadow_layer(render: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """The render's ground shadow alone (RGBA float), continued under the figure.

    Only pixels clearly outside the figure count: the matte's anti-aliased rim would otherwise read as a pale halo
    wherever the repaint does not cover the render exactly. Under the figure each pixel takes the shadow of the
    nearest pixel outside it: just above a sole that is the contact shadow below it, so a repainted foot that sits
    a little higher or is smaller than the rendered one still stands in its shadow (a fill blended from the whole
    outline mixes in the empty floor beside the shoe and leaves a paper-white band under the sole); up the legs it
    is the empty floor beside them, so a slimmer repaint shows no ghost of the rendered body.
    """
    outside = cv2.erode((matte < 0.02).astype(np.uint8), np.ones((5, 5), np.uint8))
    a = render[..., 3] * outside
    out = np.zeros_like(render)
    if a.max() < 0.03:
        return out
    h, w = a.shape
    iy, ix = ndimage.distance_transform_edt(outside == 0, return_indices=True)[1]
    filled = cv2.GaussianBlur(a[iy, ix], (0, 0), max(1.0, min(h, w) / 400))  # soften the nearest-pixel seams
    out[..., :3] = np.median(render[..., :3][a > 0.05], axis=0) if (a > 0.05).any() else 0.0
    out[..., 3] = np.clip(np.where(outside > 0, a, filled), 0, 1)
    return out


def similarity(bm: tuple[int, int, int, int], br: tuple[int, int, int, int],
               size: tuple[int, int]) -> tuple[float, float, float]:
    """Uniform scale + shift (s, tx, ty) putting the repaint's figure where the render's stands, undistorted.

    A whole figure (head and feet clear of the frame on both) is scaled by height and set on the render's feet
    line, centred on it; a bust or head is cut by the frame, which already fixes it: identity.
    """
    H = size[1]
    if not (bm[1] > 1 and br[1] > 1 and bm[3] < H - 1 and br[3] < H - 1):
        return 1.0, 0.0, 0.0
    s = (br[3] - br[1]) / max(bm[3] - bm[1], 1)
    return float(s), float((br[0] + br[2]) / 2 - s * (bm[0] + bm[2]) / 2), float(br[3] - s * bm[3])


def cutout(painted: Image.Image, mask: np.ndarray, render: Figure, size: tuple[int, int]) -> Image.Image:
    """The repaint aligned onto the render's canvas (at `size`) by `similarity`, over the render's soft shadow."""
    W, H = size
    rgba = np.asarray(painted.convert("RGBA")).astype(np.float32) / 255
    if (rgba[..., 3] < 0.5).mean() <= 0.02:  # opaque repaint: the cleaned mask is the alpha
        rgba[..., 3] = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 0.8)
    rnd = np.asarray(render_rgba(render).resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
    matte = mask_of(render.matte) if render.matte else (rnd[..., 3] > 0.5).astype(np.float32)
    matte = cv2.resize(matte, (W, H), interpolation=cv2.INTER_LINEAR)
    bm, br = _bbox(mask), _bbox(matte > 0.5)
    canvas = _shadow_layer(rnd, matte)
    if bm and br:
        s, tx, ty = similarity(bm, br, (W, H))
        rgba[..., :3] *= rgba[..., 3:4]  # warp premultiplied so edges do not pick up the transparent colour
        fig = cv2.warpAffine(rgba, np.float32([[s, 0, tx], [0, s, ty]]), (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fig = np.clip(fig, 0, 1)
        a = fig[..., 3:4]
        out_a = a + canvas[..., 3:4] * (1 - a)
        canvas[..., :3] = (fig[..., :3] + canvas[..., :3] * canvas[..., 3:4] * (1 - a)) / np.maximum(out_a, 1e-6)
        canvas[..., 3:4] = out_a
    return Image.fromarray((canvas * 255).round().clip(0, 255).astype(np.uint8), "RGBA")


# --- model calls ---------------------------------------------------------------------------------------
def request_size(w: int, h: int) -> tuple[int, int]:
    """Render size scaled into the model's pixel budget, multiples of 16."""
    s = max(1.0, (MIN_PIXELS / (w * h)) ** 0.5)
    s = min(s, (MAX_PIXELS / (w * h)) ** 0.5, MAX_EDGE / max(w, h))
    return max(16, int(w * s) // 16 * 16), max(16, int(h * s) // 16 * 16)


def _sha(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def fingerprint(cfg: Config, slot: Slot, fig: Figure) -> str:
    """What a repaint was made from: the render + matte bytes, the model and the slot's prompts (the cache key: a
    prompt change repaints only the slots whose prompt it changes; a repaint whose fingerprint no longer matches is
    stale and must not be laid out over the current render)."""
    kind, grip = kind_of(slot), grip_of(slot, cfg.athlete.backhand)
    text = hashlib.sha1((prompt(kind, False, grip) + prompt(kind, True, grip)).encode()).hexdigest()[:8]
    return f"{_sha(fig.image, *([fig.matte] if fig.matte else []))}-{cfg.models.image_paint}-p{text}"


def _over_paper(fig: Figure, size: tuple[int, int], dest: Path) -> Path:
    im = render_rgba(fig).resize(size, Image.LANCZOS)
    bg = Image.new("RGBA", size, (*paper_colour(), 255))
    bg.alpha_composite(im)
    dest.parent.mkdir(parents=True, exist_ok=True)
    bg.convert("RGB").save(dest)
    return dest


def _style_crop(kind: str, dest: Path) -> Path:
    ref = reference()
    x0, y0, x1, y1 = STYLE_CROPS[kind]
    W, H = ref.size
    dest.parent.mkdir(parents=True, exist_ok=True)
    ref.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H))).save(dest)
    return dest


def _paint_slot(cfg: Config, slot: Slot, fig: Figure, refs: dict[str, str], key: str) -> dict:
    out = cfg.run.artwork_painted
    kind, grip = kind_of(slot), grip_of(slot, cfg.athlete.backhand)
    size = request_size(*fig.size)
    quality = "xhigh" if kind in ("hero", "portrait") else "high"
    src = _over_paper(fig, size, out / "inputs" / f"{slot.name}.png")
    matte = mask_of(fig.matte) if fig.matte else mask_of(fig.image)
    ref_mask = clean_mask(cv2.resize(matte, size, interpolation=cv2.INTER_LINEAR) > 0.5)
    urls = [fal.upload(src), refs[kind], refs["identity"]]
    entry: dict[str, Any] = {"kind": kind, "grip": grip or None, "model": cfg.models.image_paint, "quality": quality,
                             "background": BACKGROUND, "size": list(size),
                             "source": str(fig.image.relative_to(cfg.run.root)), "fingerprint": key,
                             "attempts": [], "accepted": False, "iou": None}
    best: tuple[float, Path, np.ndarray] | None = None
    for attempt in (1, 2):
        strict = attempt > 1
        n = RETRY_CANDIDATES if strict else 1
        t0 = time.perf_counter()
        try:
            res = fal.subscribe(cfg.models.image_paint, {
                "image_urls": urls, "prompt": prompt(kind, strict, grip), "quality": quality, "background": BACKGROUND,
                "image_size": {"width": size[0], "height": size[1]}, "output_format": "png", "num_images": n,
            })
        except Exception as e:  # noqa: BLE001 - the output moderation fires on some samples only: try again
            if "content_policy" not in str(e):
                raise
            log(f"[yellow]{slot.name}: attempt {attempt} moderated by the provider, trying again[/]")
            entry["attempts"].append({"strict": strict, "error": "content_policy_violation"})
            continue
        seconds = round(time.perf_counter() - t0, 1)
        for i, image in enumerate(res["images"]):
            suffix = f"{attempt}{'abcdefghij'[i]}" if n > 1 else f"{attempt}"
            dest = fal.download(image["url"], out / "attempts" / f"{slot.name}_{suffix}.png")
            img = Image.open(dest)
            if img.size != size:
                img = img.resize(size, Image.LANCZOS)
            mask = painted_mask(img)
            iou = aligned_iou(mask, ref_mask)
            entry["attempts"].append({"file": str(dest.relative_to(out)), "strict": strict, "iou": round(iou, 4),
                                      "seconds": seconds})
            log(f"{slot.name}: attempt {suffix} IoU {iou:.3f}")
            if best is None or iou > best[0]:
                best = (iou, dest, mask)
        if best[0] >= ACCEPT_IOU:
            break
    if best is None:
        raise RuntimeError("every attempt was moderated by the provider (content_policy_violation)")
    iou, dest, mask = best
    entry["iou"] = round(iou, 4)
    entry["accepted"] = iou >= ACCEPT_IOU
    for stale in (out / f"{slot.name}.png", out / f"{slot.name}_cut.png"):
        stale.unlink(missing_ok=True)
    if entry["accepted"]:
        img = Image.open(dest)
        img.save(out / f"{slot.name}.png")
        cutout(img.resize(size, Image.LANCZOS) if img.size != size else img, mask, fig, size) \
            .save(out / f"{slot.name}_cut.png")
        entry["output"] = f"{slot.name}.png"
        entry["cutout"] = f"{slot.name}_cut.png"
        entry["cut_version"] = CUT_VERSION
    else:
        log(f"[yellow]{slot.name}: best IoU {iou:.3f} < {ACCEPT_IOU}: the sheet uses the render[/]")
    return entry


def _recut(cfg: Config, slot: Slot, fig: Figure, entry: dict) -> dict:
    """The accepted repaint (painted/<slot>.png) cut onto the render's canvas again, with the current `cutout`."""
    out = cfg.run.artwork_painted
    size = tuple(entry["size"])
    img = Image.open(out / entry["output"])
    img = img.resize(size, Image.LANCZOS) if img.size != size else img
    cutout(img, painted_mask(img), fig, size).save(out / entry["cutout"])
    log(f"{slot.name}: re-cut (cut-out v{CUT_VERSION})")
    return {**entry, "cut_version": CUT_VERSION}


def slot_entries(cfg: Config) -> dict[str, dict]:
    """Per-slot paint results: painted/<slot>.json, else the slot's row of paint.json. The per-slot files are the
    record (paint.json is their summary): a run-folder sync from the GPU host can put an older paint.json back
    while a paint run is writing, but it never removes a file the host did not have."""
    out = cfg.run.artwork_painted
    summary = (read_json(out / "paint.json") if (out / "paint.json").exists() else {}).get("slots") or {}
    entries = {}
    for slot in SLOTS:
        side = out / f"{slot.name}.json"
        e = read_json(side) if side.exists() else summary.get(slot.name)
        if e:
            entries[slot.name] = e
    return entries


def run(cfg: Config, force: bool = False) -> dict:
    run_ = cfg.run
    out = run_.artwork_painted
    run_.ensure(out / "inputs", out / "attempts")
    slots, _ = load_renders(cfg)
    if not slots:
        raise RuntimeError(f"no renders in {run_.artwork_renders}: run `r3d remote artwork render` first")
    report_path = out / "paint.json"
    report = {"model": cfg.models.image_paint, "accept_iou": ACCEPT_IOU, "background": BACKGROUND,
              "paper": list(paper_colour()),
              "slots": {} if force else slot_entries(cfg)}

    todo: list[tuple[Slot, Figure, str]] = []
    for slot in SLOTS:
        if not slot.paint:
            continue
        fig = slots.get(slot.name)
        if fig is None:
            log(f"[yellow]{slot.name}: no render, skipped[/]")
            continue
        key = fingerprint(cfg, slot, fig)
        old = report["slots"].get(slot.name) or {}
        done = old.get("fingerprint") == key and "error" not in old and (
            not old.get("accepted") or (out / old.get("cutout", "-")).exists())
        if force or not done:
            todo.append((slot, fig, key))
        elif old.get("accepted") and old.get("cut_version") != CUT_VERSION:
            report["slots"][slot.name] = _recut(cfg, slot, fig, old)
            write_json(out / f"{slot.name}.json", report["slots"][slot.name])
    log(f"{len(todo)} slots to paint ({sum(s.paint for s in SLOTS)} paintable, the rest cached)")
    if todo:
        hero = run_.sheet_dir / "hero.png"
        refs = {"identity": fal.upload(hero)}
        for kind in {kind_of(s) for s, _, _ in todo}:
            refs[kind] = fal.upload(_style_crop(kind, out / "inputs" / f"style_{kind}.png"))
        with step(f"paint {len(todo)} slots ({cfg.models.image_paint})"), ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(_paint_slot, cfg, s, f, refs, key): s.name for s, f, key in todo}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    entry = fut.result()
                except Exception as e:  # noqa: BLE001 - one failed call must not lose the other slots
                    log(f"[red]{name}: {e}[/]")
                    entry = {"error": str(e), "accepted": False}
                report["slots"][name] = entry
                write_json(out / f"{name}.json", entry)
                write_json(report_path, report)
    write_json(report_path, report)
    acc = [n for n, e in report["slots"].items() if e.get("accepted")]
    log(f"painted {len(acc)}/{len(report['slots'])} slots: "
        + ", ".join(f"{n} {e.get('iou') if e.get('iou') is not None else 'error'}" for n, e in report["slots"].items()))
    return report
