"""Stage `artwork motion-compose`: the moveset reel, every captured clip playing at once on the sheet's paper.

The reel is the moveset atlas set in motion: every clip's video of the game-ready character (`motion-render`,
painted by `motion-paint` where its pose check passed) in a grid of layout.MOTION.reel_columns columns, the stroke
groups in the atlas's order and rows (compose.atlas_grid), on the sheet's paper with the atlas's script group
labels, clip-id captions and curate-flag marks. Every tile is shifted in time so that all the contacts fall on
one reel frame: a clip holds its first frame until its swing has to start and its last frame once it is done,
and over the last half second every figure fades from its finish into the paper and back in at its first frame
(never two ghosted poses at once), so the reel loops without a cut. A thin timeline in the title marks the
contact instant. Real time (the clips' 25 fps), H.264 yuv420p with faststart, 1920 px wide.

The paper and its captions are drawn once with PIL; each tile video is decoded once at tile size (ffmpeg → numpy)
and laid on the paper multiplicatively against its own background colour, frame by frame, so that its paper
becomes the reel's paper (no tile edges, and a painted tile's drifting paper tone is evened out) while its
figure and contact shadow stay as rendered; the frames are piped to libx264.

  → artwork/motion/rybakina_moveset_reel.mp4 (+ .jpg poster at the contact instant): the painted tile where
    motion_painted/<clip>.json says accepted, the render elsewhere (a copy of the render reel while none is)
  → artwork/motion/rybakina_moveset_reel_render.mp4 (+ .jpg): every tile the render
  → artwork/motion/rybakina_moveset_tiles/<clip>.mp4, <clip>_painted.mp4 (+ .jpg at contact): light per-clip
    copies for the review site's grid (dozens of 1024 px videos playing at once would choke a browser), the
    painted one put on the render's frames
  → artwork/motion/motion.json: which tiles are painted, layout, timing, sizes
"""

from __future__ import annotations

import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from ..media import _bin
from ..util import log, read_json, step, write_json
from . import compose
from .compose import CARD, FLAG, INK, INK_SOFT, RULE, SCRIPT, SERIF, atlas_grid, paper
from .layout import MOTION

REEL_W = 1920                  # px; the reel's height follows from the grid
TILE_WEB = 360                 # px, the site's per-clip copies (sharp at ~180 css px on a 2x screen)
HOLD_S = 0.24                  # every tile rests on its last frame before the loop fades back
TAIL_S = 0.48                  # the fade from every tile's finish through the paper to its first frame (loop point)
CRF = 20
FEATHER = 0.035                # of the tile edge: tiles fade into the paper over this margin
KEY_COLUMNS = 4                # free columns the last row needs to take the key
RING = 0.04                    # of the tile edge: the border band whose median is a tile frame's paper colour
WORKERS = 6
RENDER_MARK = (150, 142, 134)  # the ring after a caption whose tile stays the render in the painted reel


# --- inputs --------------------------------------------------------------------------------------------
@dataclass
class ClipMotion:
    """One clip's videos: the render (motion-render) and, if accepted, the painted version (motion-paint)."""

    id: str
    stroke: str
    flagged: bool
    render: Path | None = None
    poster: Path | None = None
    matte: Path | None = None
    n_frames: int | None = None        # renders.json's count; the decoded video is the truth for the reel
    contact: int | None = None         # contact frame within the render video
    fps: float | None = None
    painted: Path | None = None        # the accepted painted video, else None
    paint: dict | None = None          # motion_painted/<clip>.json, summarised (also when not accepted)
    row: dict = field(default_factory=dict)


def _num(*vals) -> float | None:
    return next((float(v) for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)), None)


def resolve_path(run_root: Path, folder: Path, value, default: Path) -> Path:
    """A path from a report: run-root relative when it has a folder, else a name inside the stage's folder."""
    if not isinstance(value, str) or not value:
        return default
    return run_root / value if "/" in value else folder / value


def paint_summary(entry: dict) -> dict:
    """motion_painted/<clip>.json → {accepted, iou_median, iou_p10, iou_min, model, reason}: the per-frame pose
    check's statistics (nested `iou: {median, p10, min}` as motion-paint writes them, or flat iou_median, …)."""
    iou = entry.get("iou") if isinstance(entry.get("iou"), dict) else {}
    accepted = bool(entry.get("accepted"))
    return {
        "accepted": accepted,
        "iou_median": _num(entry.get("iou_median"), iou.get("median")),
        "iou_p10": _num(entry.get("iou_p10"), iou.get("p10")),
        "iou_min": _num(entry.get("iou_min"), iou.get("min")),
        "model": entry.get("video_model") or entry.get("model"),
        "reason": None if accepted else (entry.get("error") or entry.get("reason")),
    }


def motion_clips(cfg: Config) -> list[tuple[str, list[ClipMotion]]]:
    """Every moveset clip with its motion videos, [(stroke, clips)] in the atlas's order (strokes in
    ATLAS.stroke_order, clips in curation order); a clip without a render yet has render None."""
    run = cfg.run
    rdir, pdir = run.motion_renders, run.motion_painted
    renders = read_json(rdir / "renders.json") if (rdir / "renders.json").exists() else {}
    rows = renders.get("clips") or {}
    if isinstance(rows, list):
        rows = {r["id"]: r for r in rows if isinstance(r, dict) and "id" in r}
    cur = {r["id"]: r for r in (read_json(run.curation)["clips"] if run.curation.exists() else [])}
    ids = list(dict.fromkeys(run.kept_clip_ids() + list(rows)))
    groups = []
    for stroke, cells in compose._atlas_groups(cfg, dict.fromkeys(ids)):
        clips = []
        for c in cells:
            cid, row = c["id"], rows.get(c["id"]) or {}
            meta = read_json(run.clip(cid).meta) if run.clip(cid).meta.exists() else {}
            video = resolve_path(run.root, rdir, row.get("mp4") or row.get("video"), rdir / f"{cid}.mp4")
            poster = resolve_path(run.root, rdir, row.get("poster") or row.get("jpg"), rdir / f"{cid}.jpg")
            matte = resolve_path(run.root, rdir, row.get("matte"), rdir / f"{cid}_matte.mp4")
            contact = _num(row.get("contact_frame"), row.get("contact_index"),
                           (cur.get(cid) or {}).get("contact_index"), meta.get("contact_index"))
            m = ClipMotion(cid, stroke, c["flagged"], video if video.exists() else None,
                           poster if poster.exists() else None, matte if matte.exists() else None,
                           int(n) if (n := _num(row.get("n_frames"), row.get("frames"))) else None,
                           int(contact) if contact is not None else None,
                           _num(row.get("fps"), renders.get("fps"), meta.get("play_fps"), meta.get("src_fps")), row=row)
            pj = pdir / f"{cid}.json"
            if pj.exists():
                entry = read_json(pj)
                m.paint = paint_summary(entry)
                pv = resolve_path(run.root, pdir, entry.get("mp4") or entry.get("video"), pdir / f"{cid}.mp4")
                if m.paint["accepted"] and pv.exists():
                    m.painted = pv
            clips.append(m)
        groups.append((stroke, clips))
    return groups


# --- layout + timing (pure) ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ReelLayout:
    """Where everything sits on a reel frame (px): the grid cells of compose.atlas_grid at a fixed width."""

    size: tuple[int, int]
    tile: int                                     # square tile edge
    cells: list[tuple[int, int]]                  # (row, column) of every clip, group after group
    boxes: list[tuple[int, int, int, int]]        # tile boxes, x1/y1 exclusive
    margin: int
    col_w: float
    title_h: int
    head_h: int
    cap_h: int
    row_h: int
    key_y: int                                    # baseline of the key's first line
    key_x: tuple[float, float]                    # the key's left and right edge
    key_in_grid: bool                             # in the last row's free columns (else one line under the grid)


def reel_layout(group_sizes: list[int], columns: int = MOTION.reel_columns, width: int = REEL_W) -> ReelLayout:
    margin = round(width * 0.025)
    col_w = (width - 2 * margin) / columns
    tile = int(col_w * 0.95) // 2 * 2
    title_h = round(width * 0.07)
    head_h, cap_h = round(tile * 0.18), round(tile * 0.12)
    row_h = head_h + tile + cap_h
    cells = atlas_grid(group_sizes, columns)
    n_rows = max((r for r, _ in cells), default=0) + 1
    grid_bottom = title_h + n_rows * row_h
    # like the atlas: the key takes the last row's free columns when its groups leave room there
    key_col = max((k for r, k in cells if r == n_rows - 1), default=-2) + 2
    key_in_grid = columns - key_col >= KEY_COLUMNS
    if key_in_grid:
        key_x = (margin + key_col * col_w + col_w * 0.04, width - margin - col_w * 0.04)
        key_y = title_h + (n_rows - 1) * row_h + head_h + round(cap_h * 0.9)
        height = (grid_bottom + round(width * 0.006) + 1) // 2 * 2       # even, for yuv420p
    else:
        key_x, key_y = (float(margin), float(width - margin)), grid_bottom + round(width * 0.02)
        height = (key_y + round(width * 0.014) + 1) // 2 * 2
    boxes = []
    for r, k in cells:
        x = round(margin + k * col_w + (col_w - tile) / 2)
        y = title_h + r * row_h + head_h
        boxes.append((x, y, x + tile, y + tile))
    return ReelLayout((width, height), tile, cells, boxes, margin, col_w, title_h, head_h, cap_h, row_h, key_y,
                      key_x, key_in_grid)


@dataclass(frozen=True)
class ReelTiming:
    """The loop: every clip's contact on reel frame `contact`; motion, a hold, then the fade back."""

    contact: int        # reel frame of every contact
    motion: int         # reel frames until the last clip's last frame
    hold: int
    tail: int

    @property
    def frames(self) -> int:
        return self.motion + self.hold + self.tail

    def offset(self, contact: int) -> int:
        """Reel frame of a clip's first frame."""
        return self.contact - contact

    def source(self, t: int, n: int, contact: int) -> tuple[int, float]:
        """(frame, opacity over the paper) of a clip of n frames at reel frame t: first/last frames held, and in
        the tail the last frame fading out, then the first fading in, which reel frame 0 shows in full."""
        if t < self.motion + self.hold:
            return min(max(t - self.offset(contact), 0), n - 1), 1.0
        u = 2 * (t - self.motion - self.hold + 1) / (self.tail + 1) - 1      # -1 .. 1 across the tail
        a = min(abs(u), 1.0)
        return (n - 1 if u < 0 else 0), a * a * (3 - 2 * a)


def reel_timing(clips: list[tuple[int, int]], fps: float) -> ReelTiming:
    """clips = [(n_frames, contact frame)]: contacts aligned on the latest one, so no clip starts before frame 0."""
    pre = max((c for _, c in clips), default=0)
    post = max((n - 1 - c for n, c in clips), default=0)
    return ReelTiming(pre, pre + post + 1, round(HOLD_S * fps), max(1, round(TAIL_S * fps)))


def clamp_contact(n: int, contact: int | None) -> int:
    """A clip's contact inside its frames (the middle when unknown)."""
    return min(max(int(contact), 0), n - 1) if contact is not None else n // 2


# --- video io ------------------------------------------------------------------------------------------
def decode(video: Path, size: int) -> np.ndarray:
    """Every frame of a video as uint8 RGB (n, size, size, 3): scaled to cover the square, centre-cropped."""
    vf = f"scale={size}:{size}:force_original_aspect_ratio=increase:flags=area,crop={size}:{size}"
    out = subprocess.run([_bin("ffmpeg"), "-v", "error", "-i", str(video), "-vf", vf, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    frames = np.frombuffer(out, np.uint8)
    if not frames.size or frames.size % (size * size * 3):
        raise RuntimeError(f"{video.name}: no frames decoded")
    return frames.reshape(-1, size, size, 3)


H264 = ["-c:v", "libx264", "-preset", "slow", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart"]
TO_YUV = "scale=out_color_matrix=bt709:out_range=tv:flags=bicubic+accurate_rnd+full_chroma_int,format=yuv420p"


class Encoder:
    """RGB frames piped into libx264 (written to a temp name, renamed when ffmpeg succeeds)."""

    def __init__(self, out: Path, size: tuple[int, int], fps: float, frames: int, crf: int = CRF):
        self.out = out
        self.tmp = out.with_name(f".{out.stem}.part{out.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        # one keyframe per loop: the grain of the static paper does not pulse at every GOP
        self.proc = subprocess.Popen(
            [_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-", "-an", "-vf", TO_YUV, *H264, "-crf", str(crf),
             "-g", str(max(frames, 1)), str(self.tmp)], stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait():
            self.tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed writing {self.out.name}")
        self.tmp.replace(self.out)


# --- drawing -------------------------------------------------------------------------------------------
@dataclass
class Tile:
    """A clip's frames at tile size, their paper colour, and where they go on the reel."""

    clip: ClipMotion
    frames: np.ndarray            # (n, T, T, 3) uint8
    paper: np.ndarray             # (n, 3) float32: each frame's own background colour
    contact: int
    box: tuple[int, int, int, int]
    source: str                   # painted | render
    resampled: tuple[int, int] | None = None


def frame_paper(frames: np.ndarray) -> np.ndarray:
    """Each frame's paper colour: the median of its top, left and right border bands (the floor and contact
    shadow run along the bottom), smoothed over 5 frames against compression flicker."""
    n, t = frames.shape[:2]
    b = max(2, round(t * RING))
    ring = np.concatenate([frames[:, :b].reshape(n, -1, 3), frames[:, b:, :b].reshape(n, -1, 3),
                           frames[:, b:, -b:].reshape(n, -1, 3)], axis=1)
    med = np.median(ring, axis=1).astype(np.float32)
    k = min(5, n)
    pad = np.pad(med, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    return np.stack([pad[i:i + n] for i in range(k)]).mean(0)


def resample(frames: np.ndarray, n: int) -> np.ndarray:
    """A video of another length onto n frames, evenly (a video model's own frame count)."""
    if len(frames) == n:
        return frames
    return frames[np.round(np.linspace(0, len(frames) - 1, n)).astype(int)]


def feather(t: int) -> np.ndarray:
    """(t, t, 1) weight: 1 inside, easing to 0 at the tile's edges."""
    n = max(2, round(t * FEATHER))
    r = np.clip((np.arange(t, dtype=np.float32) + 0.5) / n, 0, 1)
    r = np.minimum(r, r[::-1])
    r = r * r * (3 - 2 * r)
    return (r[:, None] * r[None, :])[..., None]


def base_image(layout: ReelLayout, groups: list[tuple[str, list[ClipMotion]]], tiles: dict[str, Tile],
               timing: ReelTiming, fps: float, painted: bool) -> tuple[Image.Image, dict]:
    """The static reel: paper, title + caption, group labels with rules, clip captions with flag marks, the key,
    the timeline's track and contact tick (returned for the moving dot)."""
    W, H = layout.size
    img = paper(layout.size, seed=17)
    d = ImageDraw.Draw(img)
    m, gx1 = layout.margin, W - layout.margin
    th = layout.title_h
    name = compose._font(SERIF, round(th * 0.38))
    base_y = round(th * 0.56)
    compose._text(d, (m, base_y), CARD["name"], name, INK)
    x = m + compose._text_w(name, CARD["name"]) + th * 0.28
    script = compose._font(SCRIPT, round(th * 0.4))
    d.text((x, base_y), "The Moveset in Motion", font=script, fill=INK_SOFT, anchor="ls")
    sub = compose._font(SERIF, round(th * 0.11))
    lines = [f"{len(tiles)} captured strokes · aligned on contact · real time, {fps:g} fps".upper(),
             ("painted over 3D renders of the game-ready character" if painted
              else "unpainted 3D renders of the game-ready character").upper()]
    tr = th * 0.012
    for i, line in enumerate(lines):
        x = gx1 - compose._text_w(sub, line, tr)
        compose._text(d, (x, th * (0.26 + 0.15 * i)), line, sub, INK_SOFT, tracking=tr)
    # timeline: a hairline across the right of the title, a tick at the contact instant
    tl_x0, tl_x1, tl_y = gx1 - round(W * 0.2), gx1, round(th * 0.56)
    lw = max(1, round(W / 1400))
    d.line([(tl_x0, tl_y), (tl_x1, tl_y)], fill=RULE, width=lw)
    cx = tl_x0 + (tl_x1 - tl_x0) * timing.contact / max(timing.frames - 1, 1)
    d.line([(cx, tl_y - th * 0.05), (cx, tl_y + th * 0.05)], fill=INK_SOFT, width=lw + 1)
    tiny = compose._font(SERIF, round(th * 0.09))
    compose._text(d, (cx - compose._text_w(tiny, "CONTACT", tr) / 2, tl_y + th * 0.17), "CONTACT", tiny, INK_SOFT,
                  tracking=tr)
    d.line([(m, th * 0.86), (gx1, th * 0.86)], fill=RULE, width=lw + 1)

    # group headers: the label where a group starts, its rule across its cells in every row
    head = compose._font(SCRIPT, round(layout.head_h * 0.78))
    count = compose._font(SERIF, round(layout.head_h * 0.28))
    cap = compose._font(SERIF, round(layout.cap_h * 0.5))
    cw = layout.col_w
    placed = [(s, c) for s, cs in groups for c in cs if c.id in tiles]
    cells = dict(zip((c.id for _, c in placed), layout.cells, strict=True))
    for stroke, cs in groups:
        span = [cells[c.id] for c in cs if c.id in cells]
        for r in sorted({r for r, _ in span}):
            ks = [k for rr, k in span if rr == r]
            y = th + r * layout.row_h + layout.head_h * 0.78
            xa, xb = m + min(ks) * cw + cw * 0.04, m + (max(ks) + 1) * cw - cw * 0.04
            if r == span[0][0]:
                d.text((xa, y - layout.head_h * 0.1), stroke.capitalize(), font=head, fill=INK_SOFT, anchor="ls")
                tx = xa + head.getlength(stroke.capitalize()) + layout.head_h * 0.25
                label = f"{len(span)} clip{'s' * (len(span) != 1)}".upper()
                xa = compose._text(d, (tx, y - layout.head_h * 0.1), label, count, INK_SOFT,
                           tracking=layout.head_h * 0.03) + layout.head_h * 0.25
            if xb > xa:
                d.line([(xa, y), (xb, y)], fill=RULE, width=lw)
    mixed = painted and any(t.source == "painted" for t in tiles.values())
    for (_, c), box in zip(placed, layout.boxes, strict=True):
        cx, y = (box[0] + box[2]) / 2, box[3] + layout.cap_h * 0.62
        tw = cap.getlength(c.id)
        d.text((cx - tw / 2, y), c.id, font=cap, fill=INK_SOFT, anchor="ls")
        rad = layout.cap_h * 0.09
        my = y - layout.cap_h * 0.15
        if c.flagged:
            mx = cx - tw / 2 - rad * 3
            d.ellipse((mx - rad, my - rad, mx + rad, my + rad), fill=FLAG)
        if mixed and tiles[c.id].source == "render":
            mx = cx + tw / 2 + rad * 3
            d.ellipse((mx - rad, my - rad, mx + rad, my + rad), outline=RENDER_MARK, width=max(1, round(rad / 2)))

    # key
    small = compose._font(SERIF, round(layout.cap_h * (0.6 if layout.key_in_grid else 0.5)))
    flagged = sum(c.flagged for _, c in placed)
    key = [("flag", f"flagged by curate ({flagged}): the captured motion does not match the clip's stroke label; "
                    "animated all the same")] if flagged else []
    if mixed:
        n_r = sum(t.source == "render" for t in tiles.values())
        key.append(("ring", f"3D render ({n_r}): no painted version kept her pose frame by frame"))
    key.append(("", "every clip shifted in time to hit the ball on the same frame; its first and last frames held"))
    (kx, kx1), ky = layout.key_x, layout.key_y
    rad = layout.cap_h * 0.09
    if layout.key_in_grid:      # a "Key" header with its rule like the groups', then one line per entry
        y = ky - layout.head_h - layout.cap_h * 0.9 + layout.head_h * 0.78
        d.text((kx, y - layout.head_h * 0.1), "Key", font=head, fill=INK_SOFT, anchor="ls")
        d.line([(kx + head.getlength("Key") + layout.head_h * 0.3, y), (kx1, y)], fill=RULE, width=lw)
    for mark, text in key:
        my = ky - layout.cap_h * 0.15
        if mark == "flag":
            d.ellipse((kx, my - rad, kx + 2 * rad, my + rad), fill=FLAG)
        elif mark == "ring":
            d.ellipse((kx, my - rad, kx + 2 * rad, my + rad), outline=RENDER_MARK, width=max(1, round(rad / 2)))
        x_end = compose._text(d, (kx + (4 * rad if mark or layout.key_in_grid else 0), ky), text, small, INK_SOFT)
        if layout.key_in_grid:
            ky += round(layout.cap_h * 1.4)
        else:
            kx = x_end + layout.cap_h * 1.4
    return img, {"timeline": [tl_x0, tl_x1, tl_y]}


def _dot(radius: float) -> tuple[np.ndarray, int]:
    """An anti-aliased disc's alpha (float32) and its half size."""
    h = int(np.ceil(radius)) + 1
    y, x = np.mgrid[-h:h + 1, -h:h + 1].astype(np.float32)
    return np.clip(radius + 0.5 - np.sqrt(x * x + y * y), 0, 1)[..., None], h


def render_reel(out: Path, base: Image.Image, tiles: list[Tile], timing: ReelTiming, fps: float,
                timeline: list[float]) -> np.ndarray:
    """Compose and encode every reel frame; returns the frame at the contact instant (the poster)."""
    canvas = np.asarray(base, np.uint8)
    W, H = base.size
    t_edge = tiles[0].frames.shape[1] if tiles else 0
    alpha = feather(t_edge) if tiles else None
    patches = [canvas[y0:y1, x0:x1].astype(np.float32) for x0, y0, x1, y1 in (t.box for t in tiles)]
    dot, dh = _dot(max(2.5, H / 520))
    ink = np.asarray(INK, np.float32)
    tl_x0, tl_x1, tl_y = timeline
    poster = None
    enc = Encoder(out, (W, H), fps, timing.frames)
    try:
        for t in range(timing.frames):
            frame = canvas.copy()
            for tile, patch in zip(tiles, patches, strict=True):
                i, opacity = timing.source(t, len(tile.frames), tile.contact)
                ratio = tile.frames[i].astype(np.float32) / np.maximum(tile.paper[i], 1.0)
                x0, y0, x1, y1 = tile.box
                frame[y0:y1, x0:x1] = np.clip(patch * (1 + alpha * opacity * (ratio - 1)), 0, 255).astype(np.uint8)
            x = round(tl_x0 + (tl_x1 - tl_x0) * t / max(timing.frames - 1, 1))
            y = round(tl_y)
            region = frame[y - dh:y + dh + 1, x - dh:x + dh + 1].astype(np.float32)
            frame[y - dh:y + dh + 1, x - dh:x + dh + 1] = (region * (1 - dot) + ink * dot).astype(np.uint8)
            enc.write(frame)
            if t == timing.contact:
                poster = frame
    except BaseException:
        enc.proc.kill()
        enc.tmp.unlink(missing_ok=True)
        raise
    enc.close()
    return poster


# --- stage ---------------------------------------------------------------------------------------------
def poster_path(video: Path) -> Path:
    return video.with_suffix(".jpg")


def tiles_dir(cfg: Config) -> Path:
    return cfg.run.motion_dir / "rybakina_moveset_tiles"


def _load_tile(clip: ClipMotion, size: int, fps: float, web: Path, painted: bool
               ) -> tuple[np.ndarray, np.ndarray, tuple | None]:
    """A clip video decoded once at the site copy's size: a painted one put on the render's frames, written out as
    the light site copy (+ its frame at contact), then scaled down to the reel's tile."""
    frames = decode(clip.painted if painted else clip.render, max(size, TILE_WEB))
    resampled = None
    if painted and clip.n_frames and len(frames) != clip.n_frames:
        resampled = (len(frames), clip.n_frames)
        frames = resample(frames, clip.n_frames)
    out = web / f"{clip.id}{'_painted' if painted else ''}.mp4"
    enc = Encoder(out, (frames.shape[2], frames.shape[1]), fps, len(frames), crf=23)
    try:
        for f in frames:
            enc.write(f)
    except BaseException:
        enc.proc.kill()
        raise
    enc.close()
    Image.fromarray(frames[clamp_contact(len(frames), clip.contact)]).save(poster_path(out), quality=86,
                                                                          optimize=True)
    if frames.shape[1] != size:
        frames = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    return frames, frame_paper(frames), resampled


def _video_info(path: Path, size: tuple[int, int], timing: ReelTiming, fps: float) -> dict:
    return {"path": path.name, "poster": poster_path(path).name, "size": list(size), "frames": timing.frames,
            "duration_s": round(timing.frames / fps, 3), "mb": round(path.stat().st_size / 1e6, 2)}


def run(cfg: Config) -> dict:
    run_ = cfg.run
    groups = motion_clips(cfg)
    clips = [c for _, cs in groups for c in cs]
    have = [c for c in clips if c.render]
    if not have:
        raise RuntimeError(f"no clip renders in {run_.motion_renders}: run `r3d remote artwork motion-render` first")
    missing = [c.id for c in clips if not c.render]
    if missing:
        log(f"[yellow]{len(missing)} clips without a motion render, left out of the reel: {', '.join(missing)}[/]")
    groups = [(s, [c for c in cs if c.render]) for s, cs in groups]
    groups = [(s, cs) for s, cs in groups if cs]
    layout = reel_layout([len(cs) for _, cs in groups])
    T = layout.tile

    fps_vals = [c.fps for c in have if c.fps]
    fps = float(np.median(fps_vals)) if fps_vals else 25.0
    web = tiles_dir(cfg)
    web.mkdir(parents=True, exist_ok=True)
    with step(f"decode {len(have)} clip renders + {sum(bool(c.painted) for c in have)} painted "
              f"(site copies at {TILE_WEB} px, reel tiles at {T} px)"):
        with ThreadPoolExecutor(WORKERS) as pool:
            rendered = dict(zip((c.id for c in have), pool.map(lambda c: _load_tile(c, T, fps, web, False), have),
                                strict=True))
            for c in have:   # the reel is driven by the frames actually decoded
                c.n_frames = len(rendered[c.id][0])
            todo = [c for c in have if c.painted]
            painted_frames = dict(zip((c.id for c in todo),
                                      pool.map(lambda c: _load_tile(c, T, fps, web, True), todo), strict=True))
        keep = {f"{c.id}{s}" for c in have for s in (".mp4", ".jpg")}
        keep |= {f"{c.id}_painted{s}" for c in todo for s in (".mp4", ".jpg")}
        turntable = run_.motion_renders / "turntable.mp4"
        if turntable.exists() and not (run_.motion_renders / "turntable.jpg").exists():
            # the site shows a poster until the turntable scrolls into view
            subprocess.run([_bin("ffmpeg"), "-v", "error", "-y", "-i", str(turntable), "-frames:v", "1", "-q:v", "3",
                            str(web / "turntable.jpg")], check=True)
            keep.add("turntable.jpg")
        for p in web.iterdir():
            if p.name not in keep:
                p.unlink()
    contacts = {c.id: clamp_contact(c.n_frames, c.contact) for c in have}
    timing = reel_timing([(c.n_frames, contacts[c.id]) for c in have], fps)
    boxes = dict(zip((c.id for c in have), layout.boxes, strict=True))

    def tiles_for(painted: bool) -> dict[str, Tile]:
        out = {}
        for c in have:
            use = painted and c.id in painted_frames
            frames, bg, rs = painted_frames[c.id] if use else rendered[c.id]
            out[c.id] = Tile(c, frames, bg, contacts[c.id], boxes[c.id], "painted" if use else "render", rs)
        return out

    report: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"renders": str((run_.motion_renders / "renders.json").relative_to(run_.root)),
                   "paint": str((run_.motion_painted / "paint.json").relative_to(run_.root))
                   if (run_.motion_painted / "paint.json").exists() else None,
                   "video_paint_model": cfg.models.video_paint, "image_paint_model": cfg.models.image_paint},
        "fps": fps,
        "timing": {"contact_frame": timing.contact, "contact_s": round(timing.contact / fps, 3),
                   "motion_frames": timing.motion, "hold_frames": timing.hold, "tail_frames": timing.tail,
                   "frames": timing.frames, "duration_s": round(timing.frames / fps, 3)},
        "layout": {"size": list(layout.size), "columns": MOTION.reel_columns, "tile": T,
                   "groups": [{"stroke": s, "n": len(cs)} for s, cs in groups]},
    }
    render_tiles = tiles_for(False)
    with step(f"moveset reel, 3D render ({layout.size[0]}x{layout.size[1]}, {timing.frames} frames)"):
        base, extra = base_image(layout, groups, render_tiles, timing, fps, painted=False)
        poster = render_reel(run_.moveset_reel_render, base, list(render_tiles.values()), timing, fps,
                             extra["timeline"])
        Image.fromarray(poster).save(poster_path(run_.moveset_reel_render), quality=88, optimize=True,
                                     progressive=True)
        report["reel_render"] = _video_info(run_.moveset_reel_render, layout.size, timing, fps)
    paint_tiles = tiles_for(True)
    n_painted = sum(t.source == "painted" for t in paint_tiles.values())
    if n_painted:
        with step(f"moveset reel, painted ({n_painted} of {len(paint_tiles)} tiles painted)"):
            base, extra = base_image(layout, groups, paint_tiles, timing, fps, painted=True)
            poster = render_reel(run_.moveset_reel, base, list(paint_tiles.values()), timing, fps, extra["timeline"])
            Image.fromarray(poster).save(poster_path(run_.moveset_reel), quality=88, optimize=True, progressive=True)
    else:
        log("[yellow]no accepted painted clips: the painted reel is the render reel[/]")
        shutil.copyfile(run_.moveset_reel_render, run_.moveset_reel)
        shutil.copyfile(poster_path(run_.moveset_reel_render), poster_path(run_.moveset_reel))
    report["reel"] = _video_info(run_.moveset_reel, layout.size, timing, fps)


    report["clips"] = {}
    for (r, k), c in zip(layout.cells, have, strict=True):
        t = paint_tiles[c.id]
        report["clips"][c.id] = {
            "stroke": c.stroke, "flagged": c.flagged, "cell": [r, k], "box_px": list(t.box),
            "n_frames": c.n_frames, "contact_frame": contacts[c.id], "start_frame": timing.offset(contacts[c.id]),
            "source": t.source, "paint": c.paint,
            "painted_frames_resampled": list(t.resampled) if t.resampled else None,
            "render": str(c.render.relative_to(run_.root)),
            "painted": str(c.painted.relative_to(run_.root)) if c.painted else None,
            "tile": f"{tiles_dir(cfg).name}/{c.id}.mp4",
            "tile_painted": f"{tiles_dir(cfg).name}/{c.id}_painted.mp4" if c.painted else None,
        }
    report["painted"] = sorted(c for c, t in paint_tiles.items() if t.source == "painted")
    report["rendered"] = sorted(c for c, t in paint_tiles.items() if t.source == "render")
    report["missing"] = missing
    write_json(run_.motion_report, report)
    log(f"reel: {len(have)} clips ({n_painted} painted, {len(have) - n_painted} render"
        + (f", {len(missing)} missing" if missing else "") + f"), {timing.frames} frames = "
        f"{timing.frames / fps:.2f} s, contact at {timing.contact / fps:.2f} s, {layout.size[0]}x{layout.size[1]}, "
        f"{report['reel']['mb']} + {report['reel_render']['mb']} MB → {run_.motion_dir}")
    return report
