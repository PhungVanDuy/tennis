"""Stage `character refs`: gather identity and style references into run/character/refs.

Sources: the client's reference deck (photos, a stylized avatar card, the target-style render), a screenshot of a
character sheet in the target style, and freely licensed photos from Wikimedia Commons.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import PROJECT_ROOT, Config
from ..util import log, write_json

CLIENT_PDF = PROJECT_ROOT / "Intelligent Internet Mail - Task - 3D tennis.pdf"
SHEET_SCREENSHOT = PROJECT_ROOT / "Screenshot 2026-10-21 at 14.05.59.png"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "rybakina3d/0.1 (+https://ii.inc; research use) python-requests"}  # Wikimedia UA policy


@dataclass
class Ref:
    name: str
    role: str          # identity | style | sheet
    source: str
    path: str = ""
    note: str = ""


# (page, image index on the page, optional crop box in image pixels)
PDF_REFS = [
    (Ref("client_avatar_card", "style", "client deck: stylized avatar card",
         note="the client's own stylized Rybakina: identity + target style"), 2, 0, (17, 12, 372, 488)),
    (Ref("osaka_render", "style", "client deck: target art style render"), 5, 1, None),
    (Ref("photo_racket_ready", "identity", "client deck reference photo"), 4, 0, None),
    (Ref("photo_smile", "identity", "client deck reference photo"), 4, 1, None),
    (Ref("photo_serve_prep", "identity", "client deck reference photo"), 5, 0, None),
]
SHEET_CROP = (120, 1292, 822, 1478)  # the visible part of the target-style character sheet in the screenshot

COMMONS = [
    ("commons_full_body", "Elena Rybakina - 2025 Roland Garros - still.jpg", "full-body proportions"),
    ("commons_face_front", "Elena Rybakina 2023 US Open.jpg", "face, frontal"),
    ("commons_upper_body", "Elena Rybakina US Open 2024 (cropped).jpg", "face + build, white kit"),
    ("commons_back_hair", "Elena Rybakina (51282505614).jpg", "hair from behind, ponytail"),
]


def _pdf_images(pdf: Path) -> dict[tuple[int, int], bytes]:
    import pymupdf

    out = {}
    doc = pymupdf.open(pdf)
    for pno, page in enumerate(doc, 1):
        for i, img in enumerate(page.get_images(full=True)):
            out[(pno, i)] = doc.extract_image(img[0])["image"]
    return out


def _commons(title: str, width: int = 1600) -> tuple[bytes, dict]:
    import requests

    r = requests.get(COMMONS_API, params=dict(action="query", format="json", titles="File:" + title,
                                              prop="imageinfo", iiprop="url|size|extmetadata", iiurlwidth=width),
                     headers=UA, timeout=30).json()
    info = next(iter(r["query"]["pages"].values()))["imageinfo"][0]
    meta = info.get("extmetadata", {})
    attribution = {k: meta.get(k, {}).get("value", "") for k in ("Artist", "LicenseShortName")}
    # Commons only renders thumbnails smaller than the original.
    url = (info["thumburl"] if info["width"] > width else info["url"]).split("?")[0]  # drop utm tracking
    for attempt in range(5):
        img = requests.get(url, headers=UA, timeout=60)
        if img.status_code != 429:
            break
        time.sleep(5 * 2**attempt)  # rate limited
    img.raise_for_status()
    return img.content, {"url": info["descriptionurl"], **attribution}


def run(cfg: Config) -> list[Ref]:
    import io

    from PIL import Image

    if cfg.character.refs_dir:  # a variant the art team drew: their renders are the references
        from .art import collect

        return collect(cfg)
    out_dir = cfg.run.refs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: list[Ref] = []

    images = _pdf_images(CLIENT_PDF)
    for ref, page, idx, crop in PDF_REFS:
        im = Image.open(io.BytesIO(images[(page, idx)])).convert("RGB")
        if crop:
            im = im.crop(crop)
        ref.path = f"{ref.name}.png"
        im.save(out_dir / ref.path)
        refs.append(ref)

    sheet = Image.open(SHEET_SCREENSHOT).convert("RGB").crop(SHEET_CROP)
    sheet.save(out_dir / "osaka_character_sheet.png")
    refs.append(Ref("osaka_character_sheet", "sheet", "screenshot: target-style character sheet (partly visible)",
                    "osaka_character_sheet.png", "layout of the turnaround: front, back, action, face"))

    for name, title, note in COMMONS:
        path = f"{name}.jpg"
        data, attr = _commons(title)
        (out_dir / path).write_bytes(data)
        refs.append(Ref(name, "identity", f"Wikimedia Commons: {attr['url']}", path,
                        f"{note} — {attr.get('LicenseShortName', '')}"))
        time.sleep(1.0)

    write_json(cfg.run.refs, [asdict(r) for r in refs])
    log(f"{len(refs)} references in {out_dir}")
    return refs
