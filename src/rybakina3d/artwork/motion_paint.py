"""Stage `artwork motion-paint`: each clip's render carried into the reference sheet's painted look (fal, workstation).

Per clip, from `motion-render`'s artwork/motion/render/<clip>.mp4 (the character over the sheet's paper with her
contact shadow, 25 fps, frames 1:1 with the footage) and <clip>_matte.mp4 (her coverage, no shadow):

  1. Anchor. Frame 0 is repainted exactly like a sheet figure in `paint`: models.image_paint (GPT Image 2.5
     Sunburst) gets the frame, the reference's "action" crop (STYLE) and the approved hero (IDENTITY) with
     `paint.prompt` for an action figure plus the stroke's GRIP note, returns a transparent cut-out, accepted at
     silhouette IoU >= ANCHOR_IOU against the matte (one strict best-of-paint.RETRY_CANDIDATES retry;
     moderated samples are tried again). The cut-out is set on the render's feet line (`paint.similarity`) over the
     frame's own paper and shadow with the rendered figure taken out (`plate`). No anchor: the clip keeps its render.
  2. Video. Bernini-R reference video edit (models.video_paint, VIDEO_MODEL; the only model this stage drives)
     re-renders the render following the prompt, with the reference images in order: the anchor, the style crop,
     the hero and an accepted sheet repaint (when there is one); the images steer the look, nothing pins frame 0.
     Of the fal video models tried on a test forehand it kept the motion best (per-frame IoU median 0.96, p10 0.90,
     racket recall 1.0; Kling v4 edit was closed to API callers on 2026-10-22). It takes 4k+1 frames (5-121) at
     the clip's own 25 fps, so a clip is sent with frame 0 held at the head up to the next 4k+1 (`plan`: the
     58-frame groundstrokes as 61, the 81-frame serves as they are). The output is mapped back so that painted
     frame i shows render frame i (`frame_map`, by time at the output's rate; recorded per clip) and resized to
     the render's size.
  3. Pose check, per frame: the painted figure's mask against the render's matte, aligned like
     `paint.aligned_iou`. A white dress on off-white paper defeats a colour key, so masks come from a video matting
     model (MATTER, BiRefNet on fal) run on every output frame; the same matting on the render itself is the
     control (median 0.98 against the matte on a test forehand: what part of a miss is the matting's). Accepted
     when the model's frame 0 still has the anchor's silhouette (FIRST_MATCH_IOU), the IoU holds throughout
     (median >= ACCEPT_MEDIAN, 10th percentile >= ACCEPT_P10) and the racket is still there (`racket_recall`,
     RACKET_*: a racket dissolving into the swing barely moves the silhouette); otherwise the clip keeps its render.
     Recorded, not gated: how much of the anchor's look survives (`look_carried`), flicker (temporal second
     difference, painted vs render) and colour drift from frame 0 (CIELAB, painted vs render).
  4. The accepted clip is the painted figure, cut out by its soft matte, over the render's own paper and contact
     shadow (`composite`, BACKGROUND), like the sheet's repaints over the render's shadow: every clip sits on the
     same clean paper whatever the model did to the background.

Outputs (artwork/motion/painted/):
  <clip>.mp4        the accepted painted clip: H.264 yuv420p BT.709, 25 fps, the render's frame count and size
  <clip>.json       model, request summary, sent / output timing, frame map, IoU per frame + median / p10 / min,
                    control, first-frame match + look, racket, flicker, drift, accepted + why, cost, latency,
                    fingerprint (paint.json: all clips without the per-frame lists)
  first/<clip>.png  the painted frame 0 (anchor) on the render's canvas, first/<clip>_mask.png its figure,
                    first/<clip>.json its record + the matting control; first/attempts/, first/inputs/
  attempts/<clip>.raw.mp4   the model's output as downloaded; <clip>.mp4 mapped to the render's frames (no composite);
                    <clip>.masks.npz the matting of every output frame; <clip>.alpha.mp4 its soft matte
  sent/<clip>.mp4, sent/<clip>.first.png   exactly what the model got

The cache key is the render + matte bytes, both models, the request (prompt included), the matting model and
BACKGROUND: a changed render or prompt repaints that clip, --force repaints all; thresholds are re-applied to cached
clips without a call. A clip whose call fails keeps its render and is retried on the next run (its anchor is kept).
`paint_clip` paints one clip into any folder; `reassess` re-runs the checks on a saved output.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .. import fal
from ..config import Config
from ..media import _bin, probe
from ..util import log, read_json, step, write_json
from . import paint

VIDEO_MODEL = "fal-ai/bernini-r/reference-edit-video"
VIDEO_PRICE_USD_S = 0.08      # fal list price (USD per output second), for the report
VIDEO_MIN_FRAMES, VIDEO_MAX_FRAMES = 5, 121   # what Bernini-R takes, as 4k+1 frames
VIDEO_STEPS = 40              # denoising steps (the model's default is 30)
VIDEO_SIZE = 1024             # long edge of what it works at (the renders' size; it takes up to 1280)
FPS = 25.0                    # the footage's real-time rate: render frame i = footage frame i
WORKERS = 8                   # clips in flight (each holds one video request at a time; Bernini-R takes ~10 min a clip)
ACCEPT_MEDIAN = 0.85          # per-frame aligned silhouette IoU, median over the clip
ACCEPT_P10 = 0.75             # ... and its 10th percentile: a few frames may dip (fast racket, motion blur)
FIRST_MATCH_IOU = 0.85        # the model's frame 0 against the anchor it was given (same canvas, unaligned)
# The anchor only sets the look (the motion is judged frame by frame afterwards), so it passes a little below the
# sheet's paint.ACCEPT_IOU: serve stances seen side-on scored 0.82-0.84 with the pose right (a fuller painted body,
# a thin racket), and were turned away at 0.85.
ANCHOR_IOU = 0.80
RACKET_MIN_PX = 60            # render racket pixels for a frame to count (edge-on or behind her: skipped)
RACKET_MEDIAN = 0.7           # share of the render's racket pixels still painted red nearby, median over the clip
RACKET_P10 = 0.5              # ... and its 10th percentile (a racket gone at contact fails here)
MATTER = "fal-ai/birefnet/v2/video"
MATTER_ARGS = {"model": "General Use (Light)", "operating_resolution": "1024x1024", "output_mask": True,
               "refine_foreground": False, "video_output_type": "X264 (.mp4)", "video_quality": "maximum"}
SEND_CRF = 10                 # what the model gets: visually lossless
OUT_CRF = 14
MODERATION_TRIES = 3          # provider moderation fires on some samples of a figure in a short dress only
CLOSED = "api access is currently unavailable"   # what fal answers for an endpoint closed to API callers
FIRST_VERSION = 1             # bump when the anchor's making (not its prompt) changes
TWO_HANDED = "two_handed"
# The accepted clip's figure (cut out by the matting) over the render's own paper and contact shadow, like the
# sheet's repaints: every clip on the same clean paper, no model-made stains or boiling paper. "model" keeps the
# model's own background.
BACKGROUND = "render"


# --- video io --------------------------------------------------------------------------------------------
def read_video(path: Path, keep: Callable[[int], bool] | None = None,
               fn: Callable[[np.ndarray], np.ndarray] | None = None) -> list[np.ndarray]:
    """Frames of a video as RGB uint8, optionally only the indices `keep` accepts, each passed through `fn`.
    Decoded by ffmpeg with the stream's own colour matrix (the renders are tagged BT.709) and accurate rounding: the
    default conversion is a few levels off on the near-white paper."""
    p = probe(path)
    size = p.width * p.height * 3
    proc = subprocess.Popen([_bin("ffmpeg"), "-v", "error", "-i", str(path), "-vf", "scale=flags=accurate_rnd+"
                             "full_chroma_int", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    frames, i = [], 0
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            if keep is None or keep(i):
                rgb = np.frombuffer(buf, np.uint8).reshape(p.height, p.width, 3).copy()
                frames.append(fn(rgb) if fn else rgb)
            i += 1
    finally:
        proc.stdout.close()
        proc.wait()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return frames


def count_frames(path: Path) -> tuple[int, float, tuple[int, int]]:
    """(frames, fps, (w, h)) by decoding: container frame counts of generated videos are not always right."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, float(fps), size


def write_video(frames: list[np.ndarray] | np.ndarray, dest: Path, fps: float = FPS, crf: int = OUT_CRF) -> Path:
    """RGB frames → H.264 yuv420p (BT.709) mp4 with faststart, written to a temp name and renamed when complete."""
    h, w = frames[0].shape[:2]
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.stem}.part{dest.suffix}")
    cmd = [_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv:flags=accurate_rnd+full_chroma_int,format=yuv420p",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart", str(tmp)]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed writing {dest}")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def fit(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A frame resized to (w, h) (the model returns its own resolution of the same square)."""
    if frame.shape[1] == size[0] and frame.shape[0] == size[1]:
        return frame
    up = size[0] > frame.shape[1]
    return cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA)


# --- masks + measures ------------------------------------------------------------------------------------
def matte_mask(frame: np.ndarray) -> np.ndarray:
    """Figure mask of a matte frame of either polarity (light figure on dark or dark on light): whatever differs
    from the border colour by more than half the range, cleaned like `paint.clean_mask`."""
    g = frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(border))
    fg = 255.0 - bg if abs(255.0 - bg) > bg else 0.0      # the extreme farther from the background
    return paint.clean_mask(np.abs(g - bg) > abs(fg - bg) / 2)


def alpha_mask(frame: np.ndarray) -> np.ndarray:
    """Figure mask of a matting model's mask frame (white = figure)."""
    g = frame.astype(np.float32).mean(axis=2) if frame.ndim == 3 else frame.astype(np.float32)
    return paint.clean_mask(g >= 128)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 0.0


def iou_stats(masks: list[np.ndarray], refs: list[np.ndarray]) -> dict:
    """Per-frame silhouette agreement: aligned (shape, `paint.aligned_iou`) and raw (shape + place)."""
    aligned = np.array([paint.aligned_iou(m, r) for m, r in zip(masks, refs, strict=True)])
    raw = np.array([iou(m, r) for m, r in zip(masks, refs, strict=True)])
    return {"median": round(float(np.median(aligned)), 4), "p10": round(float(np.percentile(aligned, 10)), 4),
            "min": round(float(aligned.min()), 4), "worst_frame": int(aligned.argmin()),
            "raw_median": round(float(np.median(raw)), 4),
            "per_frame": [round(float(v), 4) for v in aligned], "raw_per_frame": [round(float(v), 4) for v in raw]}


def _grey(f: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)


def flicker(frames: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    """Temporal noise: mean |f(t-1) - 2 f(t) + f(t+1)| in grey levels, on the figure (mask at t) and on the paper
    away from it. Smooth motion leaves little; boiling brush texture or popping colours a lot."""
    fig, paper = [], []
    keep_off = [cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) == 0 for m in masks]
    prev, cur = _grey(frames[0]), _grey(frames[1]) if len(frames) > 1 else None
    for t in range(1, len(frames) - 1):
        nxt = _grey(frames[t + 1])
        d2 = np.abs(prev - 2 * cur + nxt)
        if masks[t].any():
            fig.append(float(d2[masks[t]].mean()))
        if keep_off[t].any():
            paper.append(float(d2[keep_off[t]].mean()))
        prev, cur = cur, nxt
    return {"figure": round(float(np.mean(fig)), 3) if fig else None,
            "paper": round(float(np.mean(paper)), 3) if paper else None}


def colour_drift(frames: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    """Identity drift: CIELAB distance (ΔE76) of the figure's mean colour, and of its top sixth (visor, hair,
    face), from frame 0's. The pose moves colour around too; compare with the same measure on the render."""
    def means(f: np.ndarray, m: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        ys = np.nonzero(m.any(axis=1))[0]
        if not len(ys):
            return None
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0] *= 100 / 255
        lab[..., 1:] -= 128
        top = m.copy()
        top[ys[0] + max(1, (ys[-1] - ys[0]) // 6):] = False
        return lab[m].mean(axis=0), lab[top].mean(axis=0)

    ref = means(frames[0], masks[0])
    body, head = [], []
    for f, m in zip(frames, masks, strict=True):
        cur = means(f, m)
        if ref is None or cur is None:
            continue
        body.append(float(np.linalg.norm(cur[0] - ref[0])))
        head.append(float(np.linalg.norm(cur[1] - ref[1])))
    if not body:
        return {"body_median": None, "body_max": None, "head_median": None, "head_max": None}
    return {"body_median": round(float(np.median(body)), 2), "body_max": round(max(body), 2),
            "head_median": round(float(np.median(head)), 2), "head_max": round(max(head), 2)}


def red_mask(frame: np.ndarray, min_sat: int) -> np.ndarray:
    """Red pixels: the racket's frame is the only saturated red in a render (skin stays below 140 saturation)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h <= 10) | (h >= 165)) & (s >= min_sat) & (v >= 60)


def racket_recall(painted: list[np.ndarray], frames: list[np.ndarray]) -> dict:
    """Is the racket still there, where the render has it: the share of the render's racket pixels (saturated red)
    with red paint within 3 px in the painted frame. The silhouette barely notices a racket dissolving into the
    swing; this does."""
    k = np.ones((7, 7), np.uint8)
    rec = []
    for p, f in zip(painted, frames, strict=True):
        ref = red_mask(f, 140)
        if ref.sum() >= RACKET_MIN_PX:
            near = cv2.dilate(red_mask(p, 70).astype(np.uint8), k) > 0
            rec.append(float((near & ref).sum() / ref.sum()))
    if not rec:
        return {"median": None, "p10": None, "min": None, "frames": 0}
    return {"median": round(float(np.median(rec)), 3), "p10": round(float(np.percentile(rec, 10)), 3),
            "min": round(min(rec), 3), "frames": len(rec)}


def look_carried(painted: list[np.ndarray], frames: list[np.ndarray], figures: list[np.ndarray],
                 anchor: np.ndarray, anchor_mask: np.ndarray) -> dict:
    """How much of the anchor's painted look the model kept, on the figure (softened, so a pixel of misalignment
    does not count as style): d(a, b) = mean |a - b| over the figure. carry = 1 - d(frame 0, anchor) / d(render 0,
    anchor): 1 when the model's frame 0 is the anchor, 0 when it is back to the render. restyle = d(painted i,
    render i) / d(anchor, render 0) over the clip (median, and over its last quarter): ~1 painted as strongly as
    the anchor throughout, ~0 reverted to the render."""
    def d(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        if not m.any():
            return 0.0
        diff = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.0) - cv2.GaussianBlur(b.astype(np.float32), (0, 0), 2.0)
        return float(np.abs(diff)[m].mean())

    k = np.ones((5, 5), np.uint8)
    m0 = cv2.erode((anchor_mask & figures[0]).astype(np.uint8), k) > 0
    base = d(frames[0], anchor, m0)
    if base < 1e-6:
        return {"carry": None, "restyle": None, "restyle_end": None, "colour_delta": None}
    restyle = [d(p, f, cv2.erode(m.astype(np.uint8), k) > 0) / base
               for p, f, m in zip(painted, frames, figures, strict=True)]
    tail = restyle[len(restyle) * 3 // 4:] or restyle
    first = d(painted[0], anchor, m0)
    return {"carry": round(1 - first / base, 3), "colour_delta": round(first, 2), "anchor_vs_render": round(base, 2),
            "restyle": round(float(np.median(restyle)), 3), "restyle_end": round(float(np.median(tail)), 3)}


# --- frame 0: the painted anchor -------------------------------------------------------------------------
def plate(frame: np.ndarray, figure: np.ndarray) -> np.ndarray:
    """The render frame without the figure: its paper and contact shadow, continued under the figure from the
    nearest pixel clearly outside it (just above a sole that is the shadow under it, up the legs the empty
    paper beside them), softened so the nearest-pixel seams do not show."""
    h, w = figure.shape
    outside = cv2.erode((~figure).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if not outside.any():
        return frame.copy()
    iy, ix = ndimage.distance_transform_edt(~outside, return_indices=True)[1]
    filled = cv2.GaussianBlur(frame[iy, ix], (0, 0), max(1.0, min(h, w) / 300))
    return np.where(outside[..., None], frame, filled)


def place(painted: Image.Image, mask: np.ndarray, frame: np.ndarray, figure: np.ndarray) -> tuple[np.ndarray,
                                                                                                    np.ndarray]:
    """The repaint (any size, transparent or opaque) set on the render frame's figure by `paint.similarity` (uniform
    scale onto the feet line) over `plate`: (RGB at the frame's size, the placed figure's mask)."""
    h, w = figure.shape
    rgba = np.asarray(painted.convert("RGBA").resize((w, h), Image.LANCZOS)).astype(np.float32) / 255
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    if (rgba[..., 3] < 0.5).mean() <= 0.02:          # opaque repaint: the cleaned mask is the alpha
        rgba[..., 3] = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
    base = plate(frame, figure).astype(np.float32) / 255
    bm, br = _bbox(m), _bbox(figure)
    if bm is None or br is None:
        return frame.copy(), figure.copy()
    s, tx, ty = paint.similarity(bm, br, (w, h))
    A = np.float32([[s, 0, tx], [0, s, ty]])
    rgba[..., :3] *= rgba[..., 3:4]                   # premultiplied: edges do not pick up the transparent colour
    fig = np.clip(cv2.warpAffine(rgba, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0), 0, 1)
    a = fig[..., 3:4]
    out = fig[..., :3] + base * (1 - a)
    placed = cv2.warpAffine(m.astype(np.uint8), A, (w, h), flags=cv2.INTER_NEAREST) > 0
    return (out * 255).round().clip(0, 255).astype(np.uint8), placed


def _bbox(m: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def grip_for(stroke: str, backhand: str) -> str:
    """`paint.GRIP` note for a clip: both hands on the handle on a two-handed backhand, else the racket hand."""
    return TWO_HANDED if stroke == "backhand" and backhand == TWO_HANDED else "racket"


def composite(painted: np.ndarray, alpha: np.ndarray, mask: np.ndarray, frame: np.ndarray,
              figure: np.ndarray) -> np.ndarray:
    """The painted figure cut out by its soft matte (only near its cleaned mask: no specks) over the render frame's
    own paper and contact shadow (`plate`), as `paint` lays a repaint over the render's shadow on the sheet."""
    near = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    a = np.clip((alpha.astype(np.float32) / 255 - 0.05) / 0.9, 0, 1) * near
    base = plate(frame, figure).astype(np.float32)
    out = painted.astype(np.float32) * a[..., None] + base * (1 - a[..., None])
    return out.round().clip(0, 255).astype(np.uint8)


# --- sources ---------------------------------------------------------------------------------------------
@dataclass
class Source:
    """One clip's render as `motion-render` left it."""

    clip: str
    video: Path
    matte: Path
    stroke: str
    frames: int
    size: tuple[int, int]
    contact: int | None = None
    meta: dict = field(default_factory=dict)


def load_source(cfg: Config, clip_id: str) -> Source:
    """The clip's render + matte (its renders.json row if there is one, else the contract's file names)."""
    root = cfg.run.motion_renders
    index = read_json(root / "renders.json") if (root / "renders.json").exists() else {}
    rows = index.get("clips") if isinstance(index, dict) and isinstance(index.get("clips"), dict) else {}
    row = rows.get(clip_id) if isinstance(rows.get(clip_id), dict) else {}

    def path(key: str, default: Path) -> Path:
        v = row.get(key)
        return (cfg.run.root / v if not Path(v).is_absolute() else Path(v)) if isinstance(v, str) else default

    video, matte = path("video", root / f"{clip_id}.mp4"), path("matte", root / f"{clip_id}_matte.mp4")
    if not video.exists() or not matte.exists():
        raise FileNotFoundError(f"{clip_id}: no render {video.name} + {matte.name} in {video.parent} "
                                "(run `r3d remote artwork motion-render`)")
    cur = _curation(cfg).get(clip_id, {})
    p = probe(video)
    return Source(clip_id, video, matte, row.get("stroke") or cur.get("stroke") or clip_id.split("_")[0],
                  int(row.get("n_frames") or p.frames), (p.width, p.height),
                  row.get("contact_frame", cur.get("contact_index")), row)


def _curation(cfg: Config) -> dict[str, dict]:
    if not cfg.run.curation.exists():
        return {}
    return {c["id"]: c for c in read_json(cfg.run.curation).get("clips", [])}


def _render(src: Source) -> tuple[list[np.ndarray], list[np.ndarray]]:
    frames = read_video(src.video)
    figures = read_video(src.matte, fn=matte_mask)
    n = min(len(frames), len(figures))
    if n != len(frames) or n != len(figures):
        log(f"[yellow]{src.clip}: render has {len(frames)} frames, matte {len(figures)}: using {n}[/]")
    return frames[:n], figures[:n]


def _sha(*paths: Path) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


# --- the video model -------------------------------------------------------------------------------------
def video_prompt(stroke: str, sheet: bool) -> str:
    """The edit instruction. Bernini-R names its inputs by order: the source video (the render), then the reference
    images: the anchor, style, identity and a sheet repaint (when there is one)."""
    return " ".join(filter(None, [
        "Repaint the source video as a hand-painted character illustration in motion. The source video is a 3D "
        f"render of our game character playing a tennis {stroke}.",
        "The first reference image is the opening frame of the source video already repainted: carry exactly that "
        "painted look (her face, hair, outfit, colours and brushwork) through every frame.",
        f"Paint it in the style of the second reference image, a different person: copy only the painting style "
        f"({paint.STYLE}), not her face, hair or outfit.",
        f"The third reference image shows the same character: {paint.IDENTITY}.",
        "The fourth reference image is another figure of her already painted in this look." if sheet else "",
        "Keep the motion exactly as in the source video: her pose in every frame, the timing, the head, arm, leg and "
        "racket positions and angles, the red racket's shape and length, the camera, framing, position and size in "
        "the frame. Change only how it is painted.",
        "Background: the same plain warm off-white paper with only her soft contact shadow; no court, no lines, no "
        "text, no other people or objects, no camera motion, no cuts.",
    ]))


def video_args(video_url: str, image_urls: list[str], stroke: str, frames: int) -> dict:
    """Bernini-R's request: the sent clip at its own rate, `frames` long, the reference images in order."""
    return {"video_url": video_url, "reference_image_urls": image_urls, "num_frames": frames,
            "frames_per_second": round(FPS), "max_image_size": VIDEO_SIZE, "num_inference_steps": VIDEO_STEPS,
            "enable_prompt_expansion": False, "prompt": video_prompt(stroke, len(image_urls) > 3)}


def plan(n: int) -> tuple[int, int]:
    """(frames sent, of which frame 0 held at the head) for a clip of n frames: Bernini-R takes 4k+1 frames,
    VIDEO_MIN_FRAMES to VIDEO_MAX_FRAMES (it would snap another count itself and lose the frame alignment)."""
    if n > VIDEO_MAX_FRAMES:
        raise ValueError(f"{n} frames ({n / FPS:.2f} s) is longer than {VIDEO_MODEL} takes ({VIDEO_MAX_FRAMES})")
    total = max(VIDEO_MIN_FRAMES, 1 + 4 * math.ceil((n - 1) / 4))
    return total, total - n


def frame_map(n: int, head: int, sent_frames: int, n_out: int, fps_out: float, stretch: bool = False) -> list[int]:
    """Output frame showing render frame i, for each i: the same moment of the sent clip (frame i sits at head + i),
    in seconds at the output's rate; `stretch` when the model fitted the whole clip into a different duration."""
    t = (head + np.arange(n) + 0.5) / FPS
    if stretch:
        t *= (n_out / fps_out) / (sent_frames / FPS)
    return np.clip(np.floor(t * fps_out), 0, n_out - 1).astype(int).tolist()


def timing(sent_frames: int, n_out: int, fps_out: float) -> bool:
    """Whether to stretch: the output's duration is off the sent one by more than 5 % (more than a frame's slack)."""
    d_in, d_out = sent_frames / FPS, n_out / fps_out
    return abs(d_out - d_in) / d_in > 0.05


def fingerprint(cfg: Config, src: Source) -> str:
    """Cache key of a painted clip: the anchor's key, the video model and its request (the frame count follows the
    render, already keyed), matting and background."""
    req = json.dumps(video_args("VIDEO", ["A", "B", "C", "D"], src.stroke, 0), sort_keys=True)
    return (f"{first_fingerprint(cfg, src)}|{cfg.models.video_paint}-r{hashlib.sha1(req.encode()).hexdigest()[:8]}"
            f"|{MATTER}|{BACKGROUND}")


def first_fingerprint(cfg: Config, src: Source) -> str:
    """What the anchor was made from: render + matte bytes, the image model, its prompts (paint's own text)."""
    grip = grip_for(src.stroke, cfg.athlete.backhand)
    text = hashlib.sha1((paint.prompt("action", False, grip) + paint.prompt("action", True, grip)).encode())
    return f"{_sha(src.video, src.matte)}-{cfg.models.image_paint}-p{text.hexdigest()[:8]}-f{FIRST_VERSION}"


# --- fal calls -------------------------------------------------------------------------------------------
def _moderated(e: Exception) -> bool:
    s = str(e).lower()
    return any(k in s for k in ("content_policy", "content policy", "moderat", "nsfw", "flagged", "safety check",
                                "sensitive content"))


def subscribe(endpoint: str, args: dict, tries: int = MODERATION_TRIES) -> tuple[dict, float, int]:
    """fal call that tries again when the provider's moderation rejects a sample: (result, seconds, tries)."""
    for attempt in range(1, tries + 1):
        t0 = time.perf_counter()
        try:
            return fal.subscribe(endpoint, args), round(time.perf_counter() - t0, 1), attempt
        except Exception as e:  # noqa: BLE001 - moderation fires on some samples only
            if not _moderated(e) or attempt == tries:
                raise
            log(f"[yellow]{endpoint}: moderated by the provider (try {attempt}), trying again[/]")
    raise RuntimeError("unreachable")


def _video_url(res: dict) -> str:
    v = res.get("video")
    v = v[0] if isinstance(v, list) else v
    if isinstance(v, dict) and v.get("url"):
        return v["url"]
    raise RuntimeError(f"no video in the response: {sorted(res)}")


def matte(video: Path, n: int, size: tuple[int, int], keep: Path | None = None) -> tuple[list[np.ndarray], dict]:
    """Figure masks (at `size`) of a video's n frames from MATTER + call info; `keep` saves the soft mask video
    there (for `composite`). A mask video with another frame count is mapped by time."""
    res, seconds, _ = subscribe(MATTER, {"video_url": fal.upload(video), **MATTER_ARGS})
    mask = res.get("mask_video") or {}
    if not mask.get("url"):
        raise RuntimeError(f"{MATTER} returned no mask video")
    with tempfile.TemporaryDirectory() as tmp:
        dest = fal.download(mask["url"], Path(tmp) / "mask.mp4")
        idx = mask_index(dest, n)
        want = set(idx)
        frames = read_video(dest, keep=want.__contains__, fn=lambda f: alpha_mask(fit(f, size)))
        m = count_frames(dest)[0]
        if keep:
            keep.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, keep)
    by_index = dict(zip(sorted(want), frames, strict=True))
    return [by_index[i] for i in idx], {"model": MATTER, "seconds": seconds, "frames": m}


def mask_index(mask_video: Path, n: int) -> list[int]:
    """Mask-video frame for each of the n frames it was made from (by time when the counts differ)."""
    m, fps, _ = count_frames(mask_video)
    return frame_map(n, 0, n, m, fps) if m != n else list(range(n))


def soft_alphas(mask_video: Path, n: int, frames: list[int], size: tuple[int, int]) -> list[np.ndarray]:
    """Soft alpha (uint8, at `size`) of the given frames (of n) from a kept mask video."""
    idx = mask_index(mask_video, n)
    want = {idx[j] for j in frames}
    got = dict(zip(sorted(want), read_video(mask_video, keep=want.__contains__,
                                            fn=lambda f: cv2.cvtColor(fit(f, size), cv2.COLOR_RGB2GRAY)),
                   strict=True))
    return [got[idx[j]] for j in frames]


def save_masks(path: Path, masks: list[np.ndarray]) -> Path:
    """Masks as packed bits (read back with `load_masks`)."""
    stack = np.stack(masks)
    np.savez_compressed(path, bits=np.packbits(stack, axis=-1), shape=np.array(stack.shape))
    return path


def load_masks(path: Path) -> np.ndarray:
    with np.load(path) as z:
        n, h, w = z["shape"]
        return np.unpackbits(z["bits"], axis=-1, count=int(w)).astype(bool).reshape(n, h, w)


# --- one clip --------------------------------------------------------------------------------------------
def _refs(cfg: Config, dest: Path) -> list[str]:
    """Uploaded reference images after the anchor: the style crop of the "action" kind, the hero, and an
    accepted repaint from the sheet when there is one."""
    ref = paint.reference()
    W, H = ref.size
    x0, y0, x1, y1 = paint.STYLE_CROPS["action"]
    crop = dest / "style_action.png"
    if not crop.exists():
        crop.parent.mkdir(parents=True, exist_ok=True)
        ref.crop((round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H))).save(crop)
    urls = [fal.upload(crop), fal.upload(cfg.run.sheet_dir / "hero.png")]
    entries = paint.slot_entries(cfg)
    for name in ("action_a", "action_b", "hero"):
        e = entries.get(name) or {}
        if e.get("accepted") and (cfg.run.artwork_painted / e.get("output", "-")).exists():
            return [*urls, fal.upload(cfg.run.artwork_painted / e["output"])]
    return urls


def _first_frame(cfg: Config, src: Source, frame: np.ndarray, figure: np.ndarray, first_dir: Path,
                 refs: list[str], force: bool) -> dict:
    """The painted anchor for frame 0 (cached in first_dir by `first_fingerprint`)."""
    key = first_fingerprint(cfg, src)
    rec = first_dir / f"{src.clip}.json"
    size = paint.request_size(*src.size)
    if rec.exists() and not force:
        e = read_json(rec)
        if e.get("fingerprint") == key:
            if e.get("accepted") and (first_dir / f"{src.clip}.png").exists() and (
                    first_dir / f"{src.clip}_mask.png").exists():
                return e
            if not e.get("accepted") and (e.get("iou") or 0) >= ANCHOR_IOU and e.get("repaint") and (
                    first_dir / e["repaint"]).exists():  # rejected under a stricter bar: re-cut, no new call
                img = Image.open(first_dir / e["repaint"])
                img = img.resize(size, Image.LANCZOS) if img.size != size else img
                return _accept_anchor(e, img, paint.painted_mask(img), frame, figure, first_dir, src, rec)
            if not e.get("accepted"):
                return e
    grip = grip_for(src.stroke, cfg.athlete.backhand)
    inputs = first_dir / "inputs" / f"{src.clip}.png"
    inputs.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).resize(size, Image.LANCZOS).save(inputs)
    ref_mask = paint.clean_mask(cv2.resize(figure.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.5)
    urls = [fal.upload(inputs), *refs[:2]]
    entry: dict[str, Any] = {"model": cfg.models.image_paint, "kind": "action", "grip": grip, "quality": "high",
                             "background": paint.BACKGROUND, "size": list(size), "fingerprint": key,
                             "attempts": [], "accepted": False, "iou": None}
    best: tuple[float, Path, np.ndarray] | None = None
    for attempt in (1, 2):
        strict = attempt > 1
        n = paint.RETRY_CANDIDATES if strict else 1
        try:
            res, seconds, tries = subscribe(cfg.models.image_paint, {
                "image_urls": urls, "prompt": paint.prompt("action", strict, grip), "quality": "high",
                "background": paint.BACKGROUND, "image_size": {"width": size[0], "height": size[1]},
                "output_format": "png", "num_images": n})
        except Exception as e:  # noqa: BLE001 - moderated every time: the strict attempt may still pass
            if not _moderated(e):
                raise
            entry["attempts"].append({"strict": strict, "error": "content_policy_violation"})
            continue
        for i, image in enumerate(res["images"]):
            suffix = f"{attempt}{'abcdefghij'[i]}" if n > 1 else f"{attempt}"
            dest = fal.download(image["url"], first_dir / "attempts" / f"{src.clip}_{suffix}.png")
            img = Image.open(dest)
            img = img.resize(size, Image.LANCZOS) if img.size != size else img
            mask = paint.painted_mask(img)
            score = paint.aligned_iou(mask, ref_mask)
            entry["attempts"].append({"file": str(dest.relative_to(first_dir)), "strict": strict,
                                      "iou": round(score, 4), "seconds": seconds, "tries": tries})
            log(f"{src.clip}: frame 0 repaint {suffix} IoU {score:.3f}")
            if best is None or score > best[0]:
                best = (score, dest, mask)
        if best and best[0] >= ANCHOR_IOU:
            break
    if best is None:
        raise RuntimeError("every frame 0 repaint was moderated by the provider")
    score, dest, mask = best
    entry.update(iou=round(score, 4), accepted=False, repaint=str(dest.relative_to(first_dir)))
    for stale in (first_dir / f"{src.clip}.png", first_dir / f"{src.clip}_mask.png"):
        stale.unlink(missing_ok=True)
    if score >= ANCHOR_IOU:
        img = Image.open(dest)
        return _accept_anchor(entry, img.resize(size, Image.LANCZOS) if img.size != size else img, mask, frame,
                              figure, first_dir, src, rec)
    log(f"[yellow]{src.clip}: frame 0 repaint best IoU {score:.3f} < {ANCHOR_IOU}: the clip keeps its render[/]")
    write_json(rec, entry)
    return entry


def _accept_anchor(entry: dict, img: Image.Image, mask: np.ndarray, frame: np.ndarray, figure: np.ndarray,
                   first_dir: Path, src: Source, rec: Path) -> dict:
    """The accepted frame 0 repaint set on the render's feet line over the frame's paper: the clip's anchor."""
    rgb, placed = place(img, mask, frame, figure)
    Image.fromarray(rgb).save(first_dir / f"{src.clip}.png")
    Image.fromarray(placed.astype(np.uint8) * 255).save(first_dir / f"{src.clip}_mask.png")
    entry = {**entry, "accepted": True, "anchor": f"{src.clip}.png",
             "anchor_iou": round(paint.aligned_iou(placed, figure), 4)}
    write_json(rec, entry)
    return entry


def _control(src: Source, first_dir: Path, figures: list[np.ndarray]) -> dict:
    """The matting model on the render itself against the matte (cached with the anchor's record)."""
    rec_path = first_dir / f"{src.clip}.json"
    rec = read_json(rec_path) if rec_path.exists() else {}
    sha = _sha(src.video, src.matte)
    if (rec.get("control") or {}).get("render") == sha:
        return rec["control"]
    masks, info = matte(src.video, len(figures), src.size)
    control = {"render": sha, **info, "iou": iou_stats(masks, figures)}
    rec["control"] = control
    write_json(rec_path, rec)
    return control


def judge(entry: dict) -> tuple[bool, str]:
    """Accept a painted clip on its first-frame match, per-frame IoU and racket recall (this module's thresholds)."""
    why = []
    first = (entry.get("first_match") or {}).get("iou")
    st = entry.get("iou") or {}
    if first is None or first < FIRST_MATCH_IOU:
        why.append(f"frame 0 vs anchor IoU {first} < {FIRST_MATCH_IOU}")
    if st.get("median") is None or st["median"] < ACCEPT_MEDIAN:
        why.append(f"median IoU {st.get('median')} < {ACCEPT_MEDIAN}")
    if st.get("p10") is None or st["p10"] < ACCEPT_P10:
        why.append(f"p10 IoU {st.get('p10')} < {ACCEPT_P10}")
    rk = entry.get("racket") or {}
    if rk.get("frames"):
        if rk["median"] < RACKET_MEDIAN:
            why.append(f"racket recall median {rk['median']} < {RACKET_MEDIAN}")
        if rk["p10"] < RACKET_P10:
            why.append(f"racket recall p10 {rk['p10']} < {RACKET_P10}")
    return not why, "; ".join(why) or "ok"


def check_model(cfg: Config, model: str) -> None:
    """The stage drives models.video_paint only, and only the model its request is written for (VIDEO_MODEL)."""
    if cfg.models.video_paint != VIDEO_MODEL:
        raise ValueError(f"models.video_paint is {cfg.models.video_paint!r}; motion-paint drives {VIDEO_MODEL} only")
    if model != cfg.models.video_paint:
        raise ValueError(f"motion-paint drives models.video_paint ({VIDEO_MODEL}) only, not {model!r}")


def paint_clip(cfg: Config, clip_id: str, model: str | None = None, out_dir: Path | None = None,
               first_dir: Path | None = None, force: bool = False, source: Source | None = None) -> dict:
    """Paint one clip into `out_dir` (default artwork/motion/painted) and return its record. `first_dir` holds the
    anchors (default out_dir/first), `source` overrides where the render is read from (tests, smoke inputs)."""
    model = model or cfg.models.video_paint
    check_model(cfg, model)
    out = Path(out_dir or cfg.run.motion_painted)
    first_dir = Path(first_dir or out / "first")
    src = source or load_source(cfg, clip_id)
    t0 = time.perf_counter()
    frames, figures = _render(src)
    n = len(frames)
    sent_frames, head = plan(n)                       # before the anchor: a clip the model cannot take costs nothing
    entry: dict[str, Any] = {"clip": clip_id, "model": model, "frames": n, "size": list(src.size),
                             "fingerprint": fingerprint(cfg, src), "accepted": False}
    refs = _refs(cfg, first_dir / "inputs")
    first = _first_frame(cfg, src, frames[0], figures[0], first_dir, refs, force)
    entry["first"] = {k: first.get(k) for k in ("model", "iou", "accepted", "anchor", "anchor_iou", "grip")}
    entry["first"]["attempts"] = len(first.get("attempts", []))
    if not first.get("accepted"):
        (out / f"{clip_id}.mp4").unlink(missing_ok=True)
        entry["reason"] = f"frame 0 repaint IoU {first.get('iou')} < {ANCHOR_IOU}"
        write_json(out / f"{clip_id}.json", entry)
        return entry

    # send: frame 0 held at the head up to the model's next frame count
    sent_video = write_video([frames[0]] * head + frames, out / "sent" / f"{clip_id}.mp4", crf=SEND_CRF)
    sent_first = out / "sent" / f"{clip_id}.first.png"
    shutil.copy2(first_dir / f"{clip_id}.png", sent_first)
    args = video_args(fal.upload(sent_video), [fal.upload(sent_first), *refs], src.stroke, sent_frames)
    entry["request"] = {k: v for k, v in args.items() if "url" not in k}
    entry["sent"] = {"frames": sent_frames, "head": head, "seconds": round(sent_frames / FPS, 3)}
    res, seconds, tries = subscribe(model, args)
    fal.download(_video_url(res), out / "attempts" / f"{clip_id}.raw.mp4")
    entry.update(latency_s=seconds, tries=tries, cost_usd=round(VIDEO_PRICE_USD_S * sent_frames / FPS, 3))
    entry = assess(src, entry, out, first_dir, frames, figures)
    entry["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(out / f"{clip_id}.json", entry)
    return entry


def assess(src: Source, entry: dict, out: Path, first_dir: Path, frames: list[np.ndarray] | None = None,
           figures: list[np.ndarray] | None = None) -> dict:
    """Map the model's raw output (attempts/<clip>.raw.mp4) back onto the render's frames, run the pose check and the
    measures, judge, and put the accepted clip in place. The output's masks are kept next to it and reused, so
    judging an output again costs no call."""
    if frames is None or figures is None:
        frames, figures = _render(src)
    n, cid = len(frames), src.clip
    raw = out / "attempts" / f"{cid}.raw.mp4"
    n_out, fps_out, out_size = count_frames(raw)
    stretch = timing(entry["sent"]["frames"], n_out, fps_out)
    fmap = frame_map(n, entry["sent"]["head"], entry["sent"]["frames"], n_out, fps_out, stretch)
    entry["output"] = {"frames": n_out, "fps": round(fps_out, 3), "size": list(out_size),
                       "seconds": round(n_out / fps_out, 3), "timing": "stretched" if stretch else "real time"}
    entry["frame_map"] = fmap
    want = set(fmap)
    decoded = dict(zip(sorted(want), read_video(raw, keep=want.__contains__, fn=lambda f: fit(f, src.size)),
                       strict=True))
    painted = [decoded[j] for j in fmap]
    video = write_video(painted, out / "attempts" / f"{cid}.mp4")

    # pose check: every output frame matted, the mapped ones compared
    mask_file, alpha_file = out / "attempts" / f"{cid}.masks.npz", out / "attempts" / f"{cid}.alpha.mp4"
    masks_out = None
    if mask_file.exists() and alpha_file.exists() and mask_file.stat().st_mtime >= raw.stat().st_mtime:
        masks_out = list(load_masks(mask_file))
        masks_out = masks_out if len(masks_out) == n_out and masks_out[0].shape == figures[0].shape else None
    if masks_out is None:
        masks_out, entry["matting"] = matte(raw, n_out, src.size, keep=alpha_file)
        save_masks(mask_file, masks_out)
    masks = [masks_out[j] for j in fmap]
    entry["iou"] = iou_stats(masks, figures)
    entry["control"] = {k: v for k, v in _control(src, first_dir, figures)["iou"].items() if "per_frame" not in k}
    anchor = np.asarray(Image.open(first_dir / f"{cid}.png").convert("RGB"))
    anchor_mask = np.asarray(Image.open(first_dir / f"{cid}_mask.png").convert("L")) > 127
    entry["first_match"] = {"iou": round(iou(masks[0], anchor_mask), 4),
                            **look_carried(painted, frames, figures, anchor, anchor_mask)}
    entry["racket"] = racket_recall(painted, frames)
    entry["flicker"] = {"painted": flicker(painted, masks), "render": flicker(frames, figures)}
    entry["drift"] = {"painted": colour_drift(painted, masks), "render": colour_drift(frames, figures)}
    entry["accepted"], entry["reason"] = judge(entry)
    entry["attempt"] = str(video.relative_to(out))
    finish(src, entry, out, painted, masks, frames, figures)
    write_json(out / f"{cid}.json", entry)
    st, fm = entry["iou"], entry["first_match"]
    log(f"{cid}: IoU median {st['median']:.3f} p10 {st['p10']:.3f} min {st['min']:.3f}, frame 0 IoU {fm['iou']:.3f} "
        f"look {fm['carry']}, racket {entry['racket']['median']} → "
        f"{'accepted' if entry['accepted'] else 'render kept: ' + entry['reason']}")
    return entry


def reassess(cfg: Config, clip_id: str, out_dir: Path | None = None, first_dir: Path | None = None,
             source: Source | None = None) -> dict:
    """`assess` again on a clip's saved output (after a change to the measures or thresholds), no model call."""
    out = Path(out_dir or cfg.run.motion_painted)
    entry = read_json(out / f"{clip_id}.json")
    if "sent" not in entry or not (out / "attempts" / f"{clip_id}.raw.mp4").exists():
        return entry
    return assess(source or load_source(cfg, clip_id), entry, out, Path(first_dir or out / "first"))


def finish(src: Source, entry: dict, out: Path, painted: list[np.ndarray] | None = None,
           masks: list[np.ndarray] | None = None, frames: list[np.ndarray] | None = None,
           figures: list[np.ndarray] | None = None) -> dict:
    """Put an accepted clip in place as <clip>.mp4 (per BACKGROUND: the painted figure over the render's paper and
    shadow, or the model's frames as they are), or withdraw it; frames not given are read from the saved files."""
    final = out / f"{src.clip}.mp4"
    final.unlink(missing_ok=True)
    entry.pop("video", None)
    if not entry.get("accepted"):
        return entry
    attempt = out / entry["attempt"]
    if BACKGROUND != "render":
        shutil.copy2(attempt, final)
    else:
        fmap = entry["frame_map"]
        if frames is None or figures is None:
            frames, figures = _render(src)
        painted = painted or read_video(attempt)
        if masks is None:
            stack = load_masks(out / "attempts" / f"{src.clip}.masks.npz")
            masks = [stack[j] for j in fmap]
        alphas = soft_alphas(out / "attempts" / f"{src.clip}.alpha.mp4", entry["output"]["frames"], fmap, src.size)
        write_video([composite(p, a, m, f, g) for p, a, m, f, g in zip(painted, alphas, masks, frames, figures,
                                                                        strict=True)], final)
    entry["video"], entry["background"] = final.name, BACKGROUND
    return entry


# --- the stage -------------------------------------------------------------------------------------------
def clip_entries(cfg: Config, out: Path | None = None) -> dict[str, dict]:
    """Per-clip records (<clip>.json; paint.json is their summary)."""
    out = Path(out or cfg.run.motion_painted)
    return {p.stem: read_json(p) for p in sorted(out.glob("*.json")) if p.name != "paint.json"}


def _fresh(e: dict, key: str, out: Path) -> bool:
    if "first_match" not in e and not e.get("accepted") and ((e.get("first") or {}).get("iou") or 0) >= ANCHOR_IOU:
        return False  # its anchor was turned away under a stricter bar: it now passes, so the video is still to make
    return (e.get("fingerprint") == key and "error" not in e
            and (not e.get("accepted") or (out / f"{e['clip']}.mp4").exists()))


def _rejudge(e: dict, out: Path, source: Callable[[str], Source]) -> dict:
    """A cached record judged again with the current thresholds (its clip put in place or withdrawn)."""
    if "iou" not in e or "first_match" not in e:
        return e
    ok, why = judge(e)
    if ok != e.get("accepted"):
        e = finish(source(e["clip"]), {**e, "accepted": ok, "reason": why}, out)
        e["accepted"] = ok and (out / f"{e['clip']}.mp4").exists()
        write_json(out / f"{e['clip']}.json", e)
    return e


def run(cfg: Config, force: bool = False) -> dict:
    run_ = cfg.run
    out = run_.motion_painted
    run_.ensure(out / "first", out / "attempts", out / "sent")
    model = cfg.models.video_paint
    check_model(cfg, model)
    clips, missing = [], []
    for cid in run_.kept_clip_ids():
        try:
            clips.append(load_source(cfg, cid))
        except FileNotFoundError:
            missing.append(cid)
    if missing:
        log(f"[yellow]{len(missing)} clips without a render ({', '.join(missing[:5])}{'…' if len(missing) > 5 else ''})"
            ", skipped[/]")
    if not clips:
        raise RuntimeError(f"no motion renders in {run_.motion_renders}: run `r3d remote artwork motion-render`")
    by_id = {c.clip: c for c in clips}
    report_path = out / "paint.json"
    report: dict[str, Any] = {"model": model, "image_model": cfg.models.image_paint, "matter": MATTER,
                              "accept": {"median": ACCEPT_MEDIAN, "p10": ACCEPT_P10, "first": FIRST_MATCH_IOU,
                                         "racket_median": RACKET_MEDIAN, "racket_p10": RACKET_P10,
                                         "first_repaint": ANCHOR_IOU},
                              "clips": {} if force else clip_entries(cfg)}
    todo = []
    for src in clips:
        old = report["clips"].get(src.clip) or {}
        if force or not _fresh(old, fingerprint(cfg, src), out):
            todo.append(src)
        else:
            report["clips"][src.clip] = _rejudge(old, out, by_id.__getitem__)
    log(f"{len(todo)} clips to paint with {model} ({len(clips)} rendered, the rest cached)")
    if todo:
        with step(f"motion-paint {len(todo)} clips ({model})"), ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(paint_clip, cfg, s.clip, model, out, None, force, s): s.clip for s in todo}
            for fut in as_completed(futures):
                cid = futures[fut]
                if fut.cancelled():
                    continue
                try:
                    entry = fut.result()
                except Exception as e:  # noqa: BLE001 - one failed clip must not lose the others
                    log(f"[red]{cid}: {e}[/]")
                    entry = {"clip": cid, "model": model, "error": str(e), "accepted": False}
                    (out / f"{cid}.mp4").unlink(missing_ok=True)
                    write_json(out / f"{cid}.json", entry)
                    if CLOSED in str(e).lower():  # the endpoint refuses everyone: the rest would only paint anchors
                        n = sum(f.cancel() for f in futures)
                        log(f"[red]{model} is unavailable ({e}); {n} clips not started, run again later[/]")
                report["clips"][cid] = entry
                write_json(report_path, _summary(report))
    report = _summary(report)
    write_json(report_path, report)
    log(f"painted {report['accepted']}/{report['n']} clips with {model}; the others keep their render")
    return report


def _summary(report: dict) -> dict:
    """paint.json: the per-clip records without their per-frame lists, plus totals."""
    rows = {}
    for cid, e in report["clips"].items():
        row = {k: v for k, v in e.items() if k not in ("frame_map", "request")}
        if isinstance(row.get("iou"), dict):
            row["iou"] = {k: v for k, v in row["iou"].items() if "per_frame" not in k}
        rows[cid] = row
    costs = [e.get("cost_usd") for e in rows.values() if e.get("cost_usd") is not None]
    return {**{k: v for k, v in report.items() if k != "clips"}, "clips": rows,
            "accepted": sum(bool(e.get("accepted")) for e in rows.values()), "n": len(rows),
            "cost_usd": round(sum(costs), 2) if costs else None}
