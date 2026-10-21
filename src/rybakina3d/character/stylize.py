"""Stage `character stylize`: references → a front hero in the client art style → an A-pose turnaround.

1. Hero (Gemini 3 Pro Image, the model with dedicated style-reference slots): the athlete from the identity
   photos, rendered in the Naomi Osaka target style, A-pose, flat light, plain background.
2. Turnaround: one wide sheet (front | left | back | right) conditioned on the approved hero, so every view
   shares proportions, outfit and scale; the figures are then cut out of the sheet for image-to-3D.
Prompts never name the athlete (the image filters key on names); the photos carry the identity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import Config
from ..gemini import client, text_part
from ..util import log, read_json, step, write_json

STYLE = (
    "a stylized, semi-realistic AAA sports-game character render: smooth simplified skin shading, clean readable "
    "shapes, slightly idealized proportions, soft cloth folds, matte materials — exactly the look of the style "
    "reference images"
)
OUTFIT = (
    "white sleeveless tennis dress with a dark-green collar trim and hem trim, white visor, white wristbands, "
    "white socks and white tennis shoes, light-blonde hair tied back in a low ponytail"
)
POSE = (
    "a neutral A-pose for rigging: standing straight, feet a little apart, arms straight and angled about 40 degrees "
    "away from the body, palms facing the thighs, fingers relaxed and slightly apart, neutral calm expression"
)
STAGE = (
    "flat even studio lighting with no cast shadows and no rim light, plain very light grey background, the whole "
    "body from head to toes in frame with a small margin, orthographic-looking camera at chest height, no racket, "
    "no ball, no text, no logos"
)

HERO_PROMPT = f"""Create a full-body character design of the woman shown in the IDENTITY photos.
Keep her real facial structure, face shape, eyes, nose, jaw, build and tall athletic proportions.
Render her as {STYLE}.
Outfit: {OUTFIT}.
Pose: {POSE}, facing the camera (front view).
{STAGE}."""

SHEET_PROMPT = f"""Character turnaround sheet of the SAME character as the HERO image (identical face, hair, outfit,
proportions and art style). Four full-body views side by side, evenly spaced, all the same height and scale,
feet on the same ground line, left to right:
1. FRONT view, 2. LEFT side view (her left side towards the camera, facing image-left), 3. BACK view,
4. RIGHT side view (her right side towards the camera, facing image-right).
Every view in the same pose: {POSE}.
{STAGE}. Nothing else in the image."""


def _image_parts(paths_by_label: list[tuple[str, Path]]):
    from google.genai import types

    parts = []
    for label, path in paths_by_label:
        parts.append(text_part(label))
        mime = "image/png" if path.suffix == ".png" else "image/jpeg"
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    return parts


def _generate_image(model: str, parts: list, aspect: str, size: str, out: Path) -> Path:
    from google.genai import types

    cfg = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
        image_config=types.ImageConfig(aspect_ratio=aspect, image_size=size),
    )
    for attempt in range(3):
        resp = client().models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                                config=cfg)
        for cand in resp.candidates or []:
            for p in cand.content.parts if cand.content else []:
                if p.inline_data and p.inline_data.data:
                    out.write_bytes(p.inline_data.data)
                    return out
        reason = resp.candidates[0].finish_reason if resp.candidates else resp.prompt_feedback
        log(f"[yellow]{model} returned no image (attempt {attempt + 1}): {reason}[/]")
    raise RuntimeError(f"{model} produced no image")


def _split_sheet(sheet: Path, out_dir: Path, names: list[str]) -> list[Path]:
    """Cut the figures out of the turnaround by background separation (largest blobs, left → right)."""
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(sheet).convert("RGB"))
    h, w = img.shape[:2]
    bg = np.median(np.concatenate([img[:8].reshape(-1, 3), img[-8:].reshape(-1, 3)]), axis=0)
    fg = (np.abs(img.astype(int) - bg).sum(-1) > 40).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cols = fg.any(axis=0)
    # column runs of foreground = figures; merge runs separated by tiny gaps (a hand's width)
    runs, start = [], None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append([start, x])
            start = None
    if start is not None:
        runs.append([start, w])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < w * 0.01:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    merged = sorted(sorted(merged, key=lambda r: r[1] - r[0], reverse=True)[: len(names)])
    if len(merged) != len(names):  # fall back to equal slices
        merged = [[i * w // len(names), (i + 1) * w // len(names)] for i in range(len(names))]
    outs = []
    for name, (x0, x1) in zip(names, merged, strict=True):
        rows = fg[:, x0:x1].any(axis=1).nonzero()[0]
        y0, y1 = (rows[0], rows[-1]) if len(rows) else (0, h)
        pad = int(0.06 * (y1 - y0))
        crop = Image.fromarray(img).crop((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))
        # square canvas, figure centred: what the image-to-3D models expect
        side = max(crop.size)
        canvas = Image.new("RGB", (side, side), tuple(int(c) for c in bg))
        canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        path = out_dir / f"{name}.png"
        canvas.save(path)
        outs.append(path)
    return outs


def run(cfg: Config, force: bool = False) -> dict:
    run_ = cfg.run
    refs = {r["name"]: run_.refs_dir / r["path"] for r in read_json(run_.refs)}
    out = run_.sheet_dir
    run_.ensure(run_.views_dir)

    hero = out / "hero.png"
    if force or not hero.exists():
        with step(f"hero ({cfg.models.image_hero})"):
            parts = _image_parts([
                ("STYLE reference 1 (target art style, a different person):", refs["osaka_render"]),
                ("STYLE reference 2 (target art style character sheet, a different person):", refs["osaka_character_sheet"]),
                ("IDENTITY + style reference (this same woman, already in the target style):", refs["client_avatar_card"]),
                ("IDENTITY photo:", refs["photo_racket_ready"]),
                ("IDENTITY photo:", refs["commons_face_front"]),
                ("IDENTITY photo:", refs["commons_upper_body"]),
                ("IDENTITY photo (full-body proportions):", refs["commons_full_body"]),
            ]) + [text_part(HERO_PROMPT)]
            _generate_image(cfg.models.image_hero, parts, "3:4", "2K", hero)

    sheet = out / "turnaround.png"
    if force or not sheet.exists():
        with step(f"turnaround sheet ({cfg.models.image_views})"):
            parts = _image_parts([
                ("HERO (the character to turn around):", hero),
                ("STYLE reference (target art style, a different person):", refs["osaka_character_sheet"]),
                ("IDENTITY photo (hair from behind):", refs["commons_back_hair"]),
            ]) + [text_part(SHEET_PROMPT)]
            _generate_image(cfg.models.image_views, parts, "21:9", "4K", sheet)

    views = _split_sheet(sheet, run_.views_dir, cfg.character.views)
    # the hero is the sharpest front view: use it for 'front'
    from PIL import Image

    front = Image.open(hero).convert("RGB")
    side = max(front.size)
    canvas = Image.new("RGB", (side, side), front.getpixel((4, 4)))
    canvas.paste(front, ((side - front.width) // 2, (side - front.height) // 2))
    canvas.save(run_.views_dir / "front.png")

    report = {
        "hero": {"model": cfg.models.image_hero, "prompt": HERO_PROMPT, "path": str(hero.relative_to(run_.root))},
        "turnaround": {"model": cfg.models.image_views, "prompt": SHEET_PROMPT, "path": str(sheet.relative_to(run_.root))},
        "views": {p.stem: str(p.relative_to(run_.root)) for p in views},
    }
    write_json(run_.stylize_report, report)
    log(f"views: {', '.join(p.name for p in views)}")
    return report
