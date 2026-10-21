"""Stage `artwork compose`: lay the figures out like the client's reference sheet, plus the moveset atlas.

Three images at layout.SHEET_SIZE on the reference's warm paper (its colour sampled from naomi_style.png, lit a
little brighter in the middle, a soft warm vignette, fine grain):
  * the character sheet: painted figures where `paint` accepted them, the render elsewhere, at the layout's boxes:
    whole figures fit and stand on their feet line with their soft shadows (the turnaround pair and each
    silhouette row share one scale, so they read as one character), the 3/4 bust fades into the paper, heads sit
    on light rounded tiles, silhouettes are flat charcoal from the mattes, the white title card and the script
    labels with thin rules;
  * the same sheet from the raw renders only, to show what is 3D;
  * the moveset atlas: every clip at contact at one physical scale, each stroke group in its own row(s) (small
    groups share the last one, see `atlas_grid`), captioned with its clip id; clips whose motion `curate` flagged
    against their label carry a small mark, explained in a key with a one-metre bar.
Each gets a 1600 px wide JPEG preview next to it (for the review site) and artwork.json records what went where.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import Config
from ..util import log, read_json, step, write_json
from .layout import ATLAS, CARD, CARD_BOX, LABELS, SHEET_SIZE, SLOTS, STYLE_REFERENCE
from .paint import ACCEPT_IOU, Figure, fingerprint, load_renders, mask_of, paper_colour, render_rgba, slot_entries

FONTS = Path("/System/Library/Fonts/Supplemental")
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")
SERIF = [(FONTS / "Baskerville.ttc", 0), (FONTS / "Didot.ttc", 0), (DEJAVU / "DejaVuSerif.ttf", 0)]
SERIF_SEMIBOLD = [(FONTS / "Baskerville.ttc", 4), (DEJAVU / "DejaVuSerif-Bold.ttf", 0)]
SCRIPT = [(FONTS / "SnellRoundhand.ttc", 0), (FONTS / "Savoye LET.ttc", 0), (FONTS / "Apple Chancery.ttf", 0),
          (DEJAVU / "DejaVuSerif-Italic.ttf", 0)]

# colours sampled from the reference sheet
INK = (52, 47, 43)              # card text
INK_SOFT = (128, 118, 108)      # captions
INK_SCRIPT = (146, 136, 126)    # the hand-lettered labels, lighter like the reference's pencil
RULE = (186, 178, 169)          # thin rules
CHARCOAL = (50, 46, 41)         # silhouettes
TILE = (238, 229, 220)          # head close-up tiles
CARD_WHITE = (254, 252, 249)
VIGNETTE = (228, 217, 205)      # the reference's corners
FLAG = (178, 96, 66)            # curate flag mark in the atlas
PREVIEW_W = 1600
SLOT_BY_NAME = {s.name: s for s in SLOTS}
GROUND_MARGIN = 0.035           # of a box's height, left under the feet line for the soft shadow
KEY_COLUMNS = 6                 # atlas: free columns the last row needs to take the key


# --- drawing primitives --------------------------------------------------------------------------------
def _font(cands: list[tuple[Path, int]], size: int) -> ImageFont.FreeTypeFont:
    for path, index in cands:
        if path.exists():
            return ImageFont.truetype(str(path), size, index=index)
    return ImageFont.load_default(size)


def paper(size: tuple[int, int], seed: int = 7) -> Image.Image:
    """The reference's paper: its colour, lit a little brighter in the middle, a soft warm vignette towards the
    corners, fibre grain and a faint, smooth mottling (cloudy, never blotchy: it must not read as stains)."""
    W, H = size
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.sqrt(((x - W / 2) / (W / 2)) ** 2 + ((y - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    v = np.clip((r - 0.45) / 0.55, 0, 1) ** 1.6
    base = np.asarray(paper_colour(), np.float32)
    img = base * (1 - v[..., None]) + np.asarray(VIGNETTE, np.float32) * v[..., None]
    img += (3.0 * np.clip(1 - r / 0.6, 0, 1) ** 2)[..., None]
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (H // 16, W // 16)).astype(np.float32), (0, 0), 4)
    mottle = cv2.resize(coarse / max(float(coarse.std()), 1e-6), (W, H), interpolation=cv2.INTER_LINEAR)
    fibre = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7)
    img += (mottle * 0.9 + fibre * 2.2)[..., None]
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")


def _grain(img: Image.Image, amount: float = 1.6, seed: int = 11) -> Image.Image:
    """A final whisper of grain over everything, so painted figures and paper read as one print."""
    a = np.asarray(img, np.float32)
    n = np.random.default_rng(seed).normal(0, amount, a.shape[:2]).astype(np.float32)
    return Image.fromarray((a + n[..., None]).clip(0, 255).astype(np.uint8), "RGB")


def _scaled(img: Image.Image, f: float) -> Image.Image:
    """RGBA resize in premultiplied alpha (no dark or light fringes on cut-out edges)."""
    w, h = max(1, round(img.width * f)), max(1, round(img.height * f))
    return img.convert("RGBa").resize((w, h), Image.LANCZOS).convert("RGBA")


def _over(sheet: Image.Image, layer: Image.Image, xy: tuple[float, float]) -> None:
    """Alpha-composite `layer` at a (possibly negative or overflowing) position, clipped to the sheet."""
    x, y = round(xy[0]), round(xy[1])
    cx0, cy0 = max(0, -x), max(0, -y)
    cx1, cy1 = min(layer.width, sheet.width - x), min(layer.height, sheet.height - y)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    sheet.alpha_composite(layer.crop((cx0, cy0, cx1, cy1)), (x + cx0, y + cy0))


def _text_w(font: ImageFont.FreeTypeFont, text: str, tracking: float = 0.0) -> float:
    return font.getlength(text) + tracking * max(len(text) - 1, 0)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont,
          fill: tuple, tracking: float = 0.0, anchor: str = "ls") -> float:
    """Draw text on a baseline (anchor 'ls'), letter-spaced by `tracking` px; returns the x after it."""
    x, y = xy
    if not tracking:
        draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
        return x + font.getlength(text)
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill, anchor=anchor)
        x += font.getlength(ch) + tracking
    return x - tracking


def _px(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    W, H = size
    return round(box[0] * W), round(box[1] * H), round(box[2] * W), round(box[3] * H)


def script_label(sheet: Image.Image, text: str, centre: tuple[float, float], size: int, rule: float,
                 arrow: bool = True) -> None:
    """The reference's hand-lettered section label: thin rules either side, a small arrow tick on the left."""
    d = ImageDraw.Draw(sheet)
    font = _font(SCRIPT, size)
    cx, cy = centre
    tw = _text_w(font, text)
    d.text((cx, cy), text, font=font, fill=INK_SCRIPT, anchor="mm")
    gap, lw = size * 0.45, max(2, round(size / 36))
    y = cy + size * 0.12
    d.line([(cx - tw / 2 - gap - rule, y), (cx - tw / 2 - gap, y)], fill=RULE, width=lw)
    d.line([(cx + tw / 2 + gap, y), (cx + tw / 2 + gap + rule, y)], fill=RULE, width=lw)
    if arrow:
        tip, a = cx - tw / 2 - gap, size * 0.16
        d.line([(tip - a, y - a * 0.7), (tip, y), (tip - a, y + a * 0.7)], fill=RULE, width=lw)


def _soft_shadow(size: tuple[int, int], rect: tuple[int, int, int, int], radius: int, blur: float,
                 opacity: float) -> Image.Image:
    a = Image.new("L", size, 0)
    ImageDraw.Draw(a).rounded_rectangle(rect, radius=radius, fill=round(255 * opacity))
    a = a.filter(ImageFilter.GaussianBlur(blur))
    layer = Image.new("RGBA", size, (60, 45, 35, 0))
    layer.putalpha(a)
    return layer


def title_card(sheet: Image.Image, box: tuple[int, int, int, int]) -> None:
    """White card with the name in a refined serif, a hairline, then ROLE / CORE MOOD / VISUAL SIGNATURE rows."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pad = round(h * 0.06)
    shadow = _soft_shadow((w + 4 * pad, h + 4 * pad), (2 * pad, 2 * pad + h * 0.012, 2 * pad + w, 2 * pad + h),
                          4, h * 0.022, 0.22)
    _over(sheet, shadow, (x0 - 2 * pad, y0 - 2 * pad))
    d = ImageDraw.Draw(sheet)
    d.rectangle(box, fill=CARD_WHITE, outline=(232, 225, 216), width=max(1, round(h / 400)))
    left, right = x0 + w * 0.075, x1 - w * 0.075
    name = _font(SERIF, round(h * 0.165))
    _text(d, (left, y0 + h * 0.255), CARD["name"], name, INK)
    lw = max(1, round(h / 300))
    d.line([(left, y0 + h * 0.355), (right, y0 + h * 0.355)], fill=RULE, width=lw)
    label = _font(SERIF_SEMIBOLD, round(h * 0.062))
    value = _font(SERIF, round(h * 0.068))
    for i, (k, v) in enumerate(CARD["rows"]):
        base = y0 + h * (0.52 + 0.155 * i)
        x = _text(d, (left, base), f"{k}:", label, INK, tracking=h * 0.004)
        _text(d, (x + h * 0.035, base), v, value, INK)
    d.line([(left, y0 + h * 0.93), (right, y0 + h * 0.93)], fill=RULE, width=lw)


# --- figures -------------------------------------------------------------------------------------------
@dataclass
class Placed:
    """Where a figure lands: sheet px per render px, and the sheet position of render pixel (0, 0)."""

    scale: float
    x: float
    y: float


def _ground(fig: Figure) -> float:
    return max(fig.ground_y, fig.bbox[3])


def fit_standing(fig: Figure, box: tuple[int, int, int, int]) -> float:
    """Largest scale at which the figure (top to feet line, its width) fits the box above the shadow margin."""
    x0, y0, x1, y1 = box
    bw, bh = fig.bbox[2] - fig.bbox[0], _ground(fig) - fig.bbox[1]
    return min((x1 - x0) / max(bw, 1), (y1 - y0) * (1 - GROUND_MARGIN) / max(bh, 1))


def shared_scales(figs: list[Figure], fits: list[float], pick) -> list[float]:
    """Scales (sheet px per render px) that give figures one physical size: `pick` chooses the common sheet px per
    metre from what each figure's box allows, and a figure that would overflow at it shrinks on its own. Renders
    are framed one by one, so the metric comes from each camera (lens, distance); without it, pixels."""
    ppm = [f.px_per_m for f in figs]
    if any(p is None for p in ppm):
        ppm = [1.0] * len(figs)
    phys = [fit * p for fit, p in zip(fits, ppm, strict=True)]
    common = float(pick(phys)) if phys else 1.0
    return [min(common, ph) / p for ph, p in zip(phys, ppm, strict=True)]


def stand(fig: Figure, box: tuple[int, int, int, int], scale: float) -> Placed:
    """Feet on the box's ground line, figure centred across the box."""
    x0, y0, x1, y1 = box
    ground_y = y1 - (y1 - y0) * GROUND_MARGIN
    cx = (fig.bbox[0] + fig.bbox[2]) / 2
    return Placed(scale, (x0 + x1) / 2 - scale * cx, ground_y - scale * _ground(fig))


def _layer(img: Image.Image, fig: Figure, placed: Placed) -> tuple[Image.Image, tuple[float, float]]:
    """The figure's image (render or painted cut-out at any resolution) scaled to the sheet."""
    k = img.width / fig.size[0]  # image px per render px
    return _scaled(img, placed.scale / k), (placed.x, placed.y)


def _feather(img: Image.Image, frac: float = 0.025) -> Image.Image:
    """Fade the alpha to zero at the canvas edges: a soft shadow running off its render must not end in a line."""
    w, h = img.size
    n = max(2, round(min(w, h) * frac))
    ramp = np.minimum(1.0, (np.arange(max(w, h), dtype=np.float32) + 0.5) / n)
    fx, fy = np.minimum(ramp[:w], ramp[:w][::-1]), np.minimum(ramp[:h], ramp[:h][::-1])
    a = np.asarray(img.getchannel("A"), np.float32) * fy[:, None] * fx[None, :]
    out = img.copy()
    out.putalpha(Image.fromarray(a.round().astype(np.uint8), "L"))
    return out


def silhouette(fig: Figure) -> Image.Image:
    """Flat charcoal figure from the matte over the render's own soft ground shadow (lightened)."""
    rgba = np.asarray(render_rgba(fig), np.float32) / 255
    matte = mask_of(fig.matte) if fig.matte else (rgba[..., 3] > 0.5).astype(np.float32)
    if matte.shape != rgba.shape[:2]:
        matte = cv2.resize(matte, rgba.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    shadow = rgba[..., 3] * (matte < 0.02) * 0.75
    a = matte + shadow * (1 - matte)
    shade = np.asarray((150, 142, 134), np.float32) / 255
    col = (np.asarray(CHARCOAL, np.float32) / 255 * matte[..., None] + shade * (shadow * (1 - matte))[..., None])
    col = col / np.maximum(a, 1e-6)[..., None]
    return Image.fromarray((np.dstack([col, a]) * 255).round().clip(0, 255).astype(np.uint8), "RGBA")


def _bust(sheet: Image.Image, img: Image.Image, fig: Figure, box: tuple[int, int, int, int], seed: int) -> None:
    """3/4 bust scaled into the box from the top, dissolving into the paper over its lower part."""
    x0, y0, x1, y1 = box
    bw, bh = fig.bbox[2] - fig.bbox[0], fig.bbox[3] - fig.bbox[1]
    s = min((x1 - x0) / max(bw, 1), (y1 - y0) / max(bh, 1))
    placed = Placed(s, (x0 + x1) / 2 - s * (fig.bbox[0] + fig.bbox[2]) / 2, y0 - s * fig.bbox[1])
    layer, xy = _layer(_feather(img, 0.06), fig, placed)  # an arm cut by the render's side must not end in a line
    a = np.asarray(layer.getchannel("A"), np.float32) / 255
    ys = np.arange(layer.height, dtype=np.float32)[:, None] + xy[1]
    fade_top, fade_bot = y0 + (y1 - y0) * 0.58, y0 + (y1 - y0) * 0.98
    t = np.clip((ys - fade_top) / (fade_bot - fade_top), 0, 1)
    # the dissolve line wanders across the figure like the edge of a wide brush stroke
    wobble = cv2.resize(np.random.default_rng(seed).normal(0, 1, (1, 9)).astype(np.float32), (a.shape[1], 1),
                        interpolation=cv2.INTER_CUBIC)
    t = np.clip(t + wobble * 0.14, 0, 1)
    keep = (1 - t * t * (3 - 2 * t)) * np.clip((fade_bot - ys) / ((y1 - y0) * 0.1), 0, 1)  # gone by fade_bot
    layer.putalpha(Image.fromarray((a * keep * 255).clip(0, 255).astype(np.uint8), "L"))
    _over(sheet, layer, xy)


def _head_tile(sheet: Image.Image, img: Image.Image, fig: Figure, box: tuple[int, int, int, int]) -> None:
    """Head close-up on a light rounded tile: the image covers the tile, the head's top just inside its top edge."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    radius = round(w * 0.035)
    _over(sheet, _soft_shadow((w + 80, h + 80), (40, 44, 40 + w, 40 + h), radius, 10, 0.16), (x0 - 40, y0 - 40))
    tile = np.empty((h, w, 4), np.float32)
    tile[..., :3] = np.asarray(TILE, np.float32) * np.linspace(1.012, 0.985, h, dtype=np.float32)[:, None, None]
    tile[..., 3] = 255
    tile = Image.fromarray(tile.clip(0, 255).astype(np.uint8), "RGBA")
    s = max(w / fig.size[0], h / fig.size[1])
    # head top just inside the tile, but never uncover the tile's top or bottom
    y = min(0.0, max(h - s * fig.size[1], h * 0.06 - s * fig.bbox[1]))
    layer, xy = _layer(img, fig, Placed(s, (w - s * fig.size[0]) / 2, y))
    _over(tile, layer, xy)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    tile.putalpha(mask)
    sheet.alpha_composite(tile, (x0, y0))


def _figure_image(name: str, fig: Figure, painted: dict[str, Path]) -> tuple[Image.Image, str]:
    if name in painted:
        return Image.open(painted[name]).convert("RGBA"), "painted"
    return render_rgba(fig), "render"


def character_sheet(figures: dict[str, Figure], painted: dict[str, Path],
                    label: tuple[str, str]) -> tuple[Image.Image, dict]:
    """The reference layout with these figures; `painted` maps slot → painted cut-out (others use the render)."""
    W, H = SHEET_SIZE
    sheet = paper(SHEET_SIZE).convert("RGBA")
    used: dict[str, dict] = {}
    boxes = {s.name: _px(s.box, SHEET_SIZE) for s in SLOTS}

    # one physical scale for the turnaround pair (exactly) and for the silhouettes (their median fit)
    fit = {n: fit_standing(figures[n], boxes[n]) for n in boxes if n in figures}
    scale: dict[str, float] = {}
    for names, pick in ((["front", "back"], min), ([s.name for s in SLOTS if s.silhouette], np.median)):
        names = [n for n in names if n in fit]
        scale.update(zip(names, shared_scales([figures[n] for n in names], [fit[n] for n in names], pick),
                         strict=True))

    for slot in SLOTS:
        fig = figures.get(slot.name)
        if fig is None:
            used[slot.name] = {"source": "missing"}
            continue
        box = boxes[slot.name]
        entry: dict = {"box_px": list(box)}
        if slot.silhouette:
            img, entry["source"] = _feather(silhouette(fig)), "silhouette"
            placed = stand(fig, box, scale[slot.name])
            _over(sheet, *_layer(img, fig, placed))
        elif slot.framing == "bust":
            img, entry["source"] = _figure_image(slot.name, fig, painted)
            _bust(sheet, img, fig, box, seed=3)
        elif slot.framing == "head":
            img, entry["source"] = _figure_image(slot.name, fig, painted)
            _head_tile(sheet, img, fig, box)
        else:
            img, entry["source"] = _figure_image(slot.name, fig, painted)
            img = _feather(img)
            placed = stand(fig, box, scale.get(slot.name, fit[slot.name]))
            _over(sheet, *_layer(img, fig, placed))
            entry["scale"] = round(placed.scale, 4)
        used[slot.name] = entry

    for text, (cx, cy) in LABELS:
        script_label(sheet, text, (cx * W, cy * H), round(H * 0.022), W * 0.036)
    title_card(sheet, _px(CARD_BOX, SHEET_SIZE))
    # provenance line in the free corner left of the card, sized to fit there
    room = CARD_BOX[0] * W - W * 0.04
    small = _font(SERIF, round(H * 0.0095))
    for line, y in zip(label, (H * 0.968, H * 0.984), strict=False):
        _text(ImageDraw.Draw(sheet), (W * 0.022, y), line.upper(), small, INK_SOFT,
              tracking=min(H * 0.0014, max(0.0, (room - _text_w(small, line.upper())) / max(len(line) - 1, 1))))
    return _grain(sheet.convert("RGB")), used


# --- moveset atlas -------------------------------------------------------------------------------------
def _atlas_groups(cfg: Config, figures: dict[str, Figure]) -> list[tuple[str, list[dict]]]:
    """[(stroke, [{id, flagged}])] in ATLAS.stroke_order, clips in curation order (best first)."""
    cur = read_json(cfg.run.curation) if cfg.run.curation.exists() else {"clips": []}
    rows = {r["id"]: r for r in cur.get("clips", [])}
    order = [r["id"] for r in cur.get("clips", [])]
    groups: dict[str, list[dict]] = {}
    for cid in sorted(figures, key=lambda c: (order.index(c) if c in order else 999, c)):
        r = rows.get(cid, {})
        stroke = r.get("stroke") or cid.split("_")[0]
        groups.setdefault(stroke, []).append({"id": cid, "flagged": r.get("ok") is False,
                                              "reason": r.get("reason")})
    rank = {s: i for i, s in enumerate(ATLAS.stroke_order)}
    return sorted(groups.items(), key=lambda g: (rank.get(g[0], 99), g[0]))


def atlas_grid(sizes: list[int], cols: int) -> list[tuple[int, int]]:
    """(row, column) of every clip, group after group: a group starts a new row unless it fits whole into the rest
    of the current one (one empty column apart from the group before); a group longer than a row wraps."""
    cells: list[tuple[int, int]] = []
    r, k = 0, 0
    for n in sizes:
        if k and k + 1 + n <= cols:
            k += 1
        elif k:
            r, k = r + 1, 0
        for _ in range(n):
            if k == cols:
                r, k = r + 1, 0
            cells.append((r, k))
            k += 1
    return cells


def moveset_atlas(cfg: Config, figures: dict[str, Figure]) -> tuple[Image.Image, dict]:
    """Every clip at contact on one sheet, one physical scale, grouped by stroke in rows of a grid."""
    W, H = SHEET_SIZE
    sheet = paper(SHEET_SIZE, seed=9).convert("RGBA")
    d = ImageDraw.Draw(sheet)
    groups = _atlas_groups(cfg, figures)
    n = sum(len(g) for _, g in groups)

    gx0, gx1 = W * 0.035, W * 0.965
    # title
    name = _font(SERIF, round(H * 0.05))
    _text(d, (gx0, H * 0.085), CARD["name"], name, INK)
    x = gx0 + _text_w(name, CARD["name"]) + H * 0.035
    script = _font(SCRIPT, round(H * 0.05))
    d.text((x, H * 0.083), "The Moveset", font=script, fill=INK_SOFT, anchor="ls")
    sub = _font(SERIF, round(H * 0.0135))
    caption = f"{n} captured strokes · each at its contact frame · one scale, same 3/4 camera".upper()
    tw = _text_w(sub, caption, H * 0.0016)
    _text(d, (gx1 - tw, H * 0.083), caption, sub, INK_SOFT, tracking=H * 0.0016)
    d.line([(gx0, H * 0.108), (gx1, H * 0.108)], fill=RULE, width=2)

    # every stroke group starts its own row (small groups share one), so a header only ever spans its own clips
    cols = ATLAS.columns
    cells: list[tuple[str, dict]] = [(stroke, c) for stroke, g in groups for c in g]
    grid = atlas_grid([len(g) for _, g in groups], cols)
    n_rows = max(1, max((r for r, _ in grid), default=0) + 1)
    # the key goes into the last row when its groups leave room there, else under the grid
    key_col = max((k for r, k in grid if r == n_rows - 1), default=-2) + 2
    key_in_grid = cols - key_col >= KEY_COLUMNS
    gy0, gy1 = H * 0.125, H * (0.975 if key_in_grid else 0.952)
    cw, rh = (gx1 - gx0) / cols, (gy1 - gy0) / n_rows
    head_h, cap_h = rh * 0.105, rh * 0.085
    fig_boxes = [(round(gx0 + k * cw + cw * 0.01), round(gy0 + r * rh + head_h), round(gx0 + (k + 1) * cw - cw * 0.01),
                  round(gy0 + (r + 1) * rh - cap_h)) for r, k in grid]
    figs = [figures[c["id"]] for _, c in cells]
    scales = shared_scales(figs, [fit_standing(f, b) for f, b in zip(figs, fig_boxes, strict=True)],
                           lambda v: np.percentile(v, 10))
    metric = all(f.px_per_m for f in figs)
    sheet_ppm = max(s * f.px_per_m for s, f in zip(scales, figs, strict=True)) if metric and figs else None

    placed_info = {}
    for (stroke, c), box, scale in zip(cells, fig_boxes, scales, strict=True):
        fig = figures[c["id"]]
        placed = stand(fig, box, scale)
        _over(sheet, *_layer(_feather(render_rgba(fig)), fig, placed))
        placed_info[c["id"]] = {"stroke": stroke, "flagged": c["flagged"], "box_px": list(box),
                                "file": str(fig.image.relative_to(cfg.run.root))}

    # group headers (label where a group starts, its rule across its cells in every row) + captions
    d = ImageDraw.Draw(sheet)
    head = _font(SCRIPT, round(head_h * 0.9))
    count = _font(SERIF, round(head_h * 0.32))
    cap = _font(SERIF, round(cap_h * 0.6))
    lw = max(2, round(H / 1300))
    i = 0
    for stroke, g in groups:
        span = grid[i:i + len(g)]
        for r in sorted({r for r, _ in span}):
            ks = [k for rr, k in span if rr == r]
            y = gy0 + r * rh + head_h * 0.78
            xa, xb = gx0 + min(ks) * cw + cw * 0.04, gx0 + (max(ks) + 1) * cw - cw * 0.04
            if r == span[0][0]:
                d.text((xa, y - head_h * 0.08), stroke.capitalize(), font=head, fill=INK_SOFT, anchor="ls")
                tx = xa + head.getlength(stroke.capitalize()) + head_h * 0.25
                label = f"{len(g)} clip{'s' * (len(g) != 1)}".upper()
                xa = _text(d, (tx, y - head_h * 0.08), label, count, INK_SOFT, tracking=head_h * 0.03) + head_h * 0.25
            if xb > xa:
                d.line([(xa, y), (xb, y)], fill=RULE, width=lw)
        i += len(g)
    for (_, c), box in zip(cells, fig_boxes, strict=True):
        cx, y = (box[0] + box[2]) / 2, box[3] + cap_h * 0.62
        text = c["id"]
        tw = cap.getlength(text)
        d.text((cx - tw / 2, y), text, font=cap, fill=INK_SOFT, anchor="ls")
        if c["flagged"]:
            rad = cap_h * 0.07
            mx, my = cx - tw / 2 - rad * 3, y - cap_h * 0.12
            d.ellipse((mx - rad, my - rad, mx + rad, my + rad), fill=FLAG)

    # key: the flag mark, a one-metre bar at the figures' common scale, how to read a caption
    small = _font(SERIF, round(H * 0.0125))
    flagged = sum(c["flagged"] for _, c in cells)
    lines = [("flag", f"flagged by curate ({flagged}): the captured motion does not match the clip's stroke label; "
                      "animated and kept in the moveset all the same"),
             ("metre", "one metre at the figures' common scale" if sheet_ppm else ""),
             ("", "groups follow the broadcast stroke labels; a caption is the clip id, its time in the source video")]
    lines = [(m, t) for m, t in lines if t]
    if key_in_grid:
        kx, kx1 = gx0 + key_col * cw + cw * 0.04, gx1 - cw * 0.04
        ky = gy0 + (n_rows - 1) * rh + head_h * 0.78
        d.text((kx, ky - head_h * 0.08), "Key", font=head, fill=INK_SOFT, anchor="ls")
        d.line([(kx + head.getlength("Key") + head_h * 0.35, ky), (kx1, ky)], fill=RULE, width=lw)
        ys = [ky + rh * (0.2 + 0.11 * i) for i in range(len(lines))]
    else:
        kx, ys = gx0, [H * 0.978] * len(lines)
        lines = lines[:1]
    mark_w = max(sheet_ppm or 0, H * 0.01) + H * 0.012
    for (mark, text), y in zip(lines, ys, strict=False):
        my = y - H * 0.0045
        if mark == "flag":
            rad = H * 0.0035
            d.ellipse((kx + mark_w - H * 0.012 - 2 * rad, my - rad, kx + mark_w - H * 0.012, my + rad), fill=FLAG)
        elif mark == "metre":
            tick = H * 0.004
            d.line([(kx, my), (kx + sheet_ppm, my)], fill=INK_SOFT, width=lw)
            for x in (kx, kx + sheet_ppm):
                d.line([(x, my - tick), (x, my + tick)], fill=INK_SOFT, width=lw)
        _text(d, (kx + mark_w, y), text, small, INK_SOFT)
    return _grain(sheet.convert("RGB"), seed=13), placed_info


# --- stage ---------------------------------------------------------------------------------------------
def preview_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_preview.jpg")


def _save(img: Image.Image, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)
    pv = img.resize((PREVIEW_W, round(img.height * PREVIEW_W / img.width)), Image.LANCZOS)
    pv.save(preview_path(path), quality=88, optimize=True, progressive=True)
    return {"path": path.name, "preview": preview_path(path).name, "size": list(img.size),
            "mb": round(path.stat().st_size / 1e6, 2)}


def _painted(cfg: Config, slots: dict[str, Figure]) -> tuple[dict[str, Path], dict[str, dict]]:
    """Accepted repaints' cut-outs by slot (only those made from the current render), and every slot's paint result."""
    entries = slot_entries(cfg)
    out = {}
    for name, e in entries.items():
        p = cfg.run.artwork_painted / str(e.get("cutout") or "-")
        if not (e.get("accepted") and p.exists()) or name not in slots:
            continue
        if e.get("fingerprint") != fingerprint(cfg, SLOT_BY_NAME[name], slots[name]):
            log(f"[yellow]{name}: the repaint is of an older render or prompt (run `r3d artwork paint`); "
                "the sheet uses the render[/]")
            e["stale"] = True
            continue
        out[name] = p
    return out, entries


def run(cfg: Config) -> dict:
    run_ = cfg.run
    slots, atlas = load_renders(cfg)
    if not slots and not atlas:
        raise RuntimeError(f"no renders in {run_.artwork_renders}: run `r3d remote artwork render` first")
    painted, paint_entries = _painted(cfg, slots)
    report: dict = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "inputs": {"reference": STYLE_REFERENCE, "renders": str((run_.artwork_renders / "renders.json")
                                                                 .relative_to(run_.root)),
                   "paint": "painted/paint.json" if paint_entries else None,
                   "paint_model": cfg.models.image_paint, "accept_iou": ACCEPT_IOU, "paper": list(paper_colour())},
        "sheet_size": list(SHEET_SIZE),
    }
    with step("compose character sheets"):
        sheet, used = character_sheet(slots, painted, ("Painted over 3D renders of the game-ready character",
                                                        "every pose from her captured motion"))
        report["character_sheet"] = _save(sheet, run_.character_sheet)
        render_sheet, used_r = character_sheet(slots, {}, ("Unpainted 3D renders of the game-ready character",
                                                            "every pose from her captured motion"))
        report["character_sheet_render"] = _save(render_sheet, run_.character_sheet_render)
    for name, e in used.items():
        if name in paint_entries:
            p = paint_entries[name]
            e["paint"] = {"accepted": bool(p.get("accepted")), "stale": bool(p.get("stale")), "iou": p.get("iou"),
                          "attempts": len(p.get("attempts") or []), "model": p.get("model"), "error": p.get("error")}
    report["figures"] = used
    report["painted"] = sorted(n for n, e in used.items() if e.get("source") == "painted")
    report["rendered"] = sorted(n for n, e in used.items() if e.get("source") == "render")
    report["missing"] = sorted(n for n, e in used_r.items() if e.get("source") == "missing")
    if atlas:
        with step(f"compose moveset atlas ({len(atlas)} clips)"):
            img, placed = moveset_atlas(cfg, atlas)
            report["moveset_atlas"] = _save(img, run_.moveset_atlas)
            report["atlas_figures"] = placed
    else:
        log("[yellow]no atlas renders: moveset atlas skipped[/]")
        report["moveset_atlas"] = None
    write_json(run_.artwork_report, report)
    log(f"sheet: {len(report['painted'])} painted, {len(report['rendered'])} render, "
        f"{sum(e.get('source') == 'silhouette' for e in used.values())} silhouettes"
        + (f", {len(report['missing'])} missing ({', '.join(report['missing'])})" if report["missing"] else "")
        + f"; atlas: {len(atlas)} clips → {run_.artwork_dir}")
    return report
