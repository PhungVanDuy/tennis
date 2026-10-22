"""A character variant drawn by the client's art team: their renders → references → the four A-pose views.

`character refs` and `character stylize` take this path when the config names `character.refs_dir` (configs/
naomi.yaml): the art team already made the character, so nothing is invented from photos. Their A-pose front
render IS the front view; the views it lacks (left side, back, right side) are drawn by GPT Image 2.5 Sunburst
(fal, models.image_paint) from that front, their action art (outfit, ponytail, racket) and their head renders
(the face in profile), on the front's canvas size so every view comes back at the same scale. The back is drawn
first and joins the references of both sides, which keeps hair and outfit consistent around the figure.

Outputs (under the variant's character/):
  refs/<name>.png + refs.json            the art team's files as references (named in <refs_dir>/refs.yaml, which also
                                         holds the outfit text the prompts quote; it stays with their files)
  sheet/attempts/<view>_<n><x>.png       every Sunburst candidate (transparent background)
  sheet/<view>.png                       the chosen candidate per view (the first unless `PICK` names another)
  sheet/views/{front,left,back,right}.png  each figure cut out by alpha, scaled to the front's figure height,
                                         feet on one line, centred on a square light-grey canvas (+ facing.json,
                                         so `character mesh` takes the views as named)
  sheet/stylize.json                     model, prompts, candidates, picks
Prompts never name the athlete (the image filters key on names); the references carry the identity.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import fal
from ..config import Config
from ..util import log, read_json, step, write_json


@dataclass
class ArtRef:
    name: str
    role: str        # front | style | head
    file: str        # in character.refs_dir
    note: str


MANIFEST = "refs.yaml"   # in character.refs_dir, beside the art team's images (out of the repository)
# what the code needs from the manifest: these reference names, and `outfit` for the view prompts
REQUIRED = ("art_front", "art_action", "art_head_front", "art_head_left", "art_head_right", "art_head_back")


def manifest(cfg: Config) -> tuple[list[ArtRef], str]:
    """The art team's references and outfit text from <refs_dir>/refs.yaml:
    {outfit: str, refs: [{name, role (front | style | head), file, note}]}."""
    import yaml

    path = cfg.character.refs_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"no {path}: list the art team's files there (see docs/character.md)")
    data = yaml.safe_load(path.read_text()) or {}
    refs = [ArtRef(r["name"], r["role"], r["file"], r.get("note", "")) for r in data.get("refs", [])]
    missing = set(REQUIRED) - {r.name for r in refs}
    if missing or not data.get("outfit"):
        raise ValueError(f"{path}: needs `outfit` and refs named {sorted(missing) or list(REQUIRED)}")
    return refs, " ".join(str(data["outfit"]).split())


STYLE = (
    "the exact look of the reference renders: a stylized, semi-realistic AAA sports-game character, smooth skin "
    "shading, clean readable shapes, soft cloth folds"
)
VIEW_TEXT = {
    "back": "BACK view: seen exactly from behind, so the back of the head and hair and the back of the outfit face "
            "the camera",
    "left": "LEFT SIDE view: turned 90 degrees so that her LEFT side faces the camera and her nose points to "
            "image-left, a pure profile",
    "right": "RIGHT SIDE view: turned 90 degrees so that her RIGHT side faces the camera and her nose points to "
             "image-right, a pure profile",
}
ORDER = ("back", "left", "right")       # the back joins the references of both sides
CANDIDATES = 2
QUALITY = "high"
CANVAS_GREY = (232, 232, 232)
MARGIN = 0.06                           # of the figure height, above the head and below the feet
PICK: dict[str, str] = {}               # view → attempt file name, when the first candidate is not the best


def collect(cfg: Config) -> list[ArtRef]:
    """`character refs` for an art-team variant: copy their files in and say what each one is for."""
    src, out = cfg.character.refs_dir, cfg.run.refs_dir
    out.mkdir(parents=True, exist_ok=True)
    refs = []
    for ref in manifest(cfg)[0]:
        if not (src / ref.file).exists():
            log(f"[yellow]missing {src / ref.file}[/]")
            continue
        shutil.copy(src / ref.file, out / f"{ref.name}.png")
        refs.append(ref)
    write_json(cfg.run.refs, [{"name": r.name, "role": r.role, "source": f"art team: {r.file}",
                               "path": f"{r.name}.png", "note": r.note} for r in refs])
    log(f"{len(refs)} art-team references in {out}")
    return refs


def prompt(view: str, has_back: bool, outfit: str) -> str:
    refs = ("Image 1 is the FRONT view of the character (A-pose). Image 2 is the same character in action (outfit "
            "and hair from another angle). The next images show her face and head from several angles, but with a "
            "DIFFERENT hairstyle and top: take only the face and skull shape from them.")
    if has_back:
        refs += " The last image is the approved BACK view of this same character."
    return f"""{refs}
Draw the SAME character as image 1, {VIEW_TEXT[view]}.
Same person, same proportions, same outfit ({outfit}), same A-pose as image 1: standing straight, feet a little
apart, arms straight and angled about 40 degrees away from the body, fingers relaxed. Same scale and framing as
image 1: the whole body from the top of the hair to the soles in frame, the same height in the image, feet at the
same height. Render it as {STYLE}. Flat even studio light, no cast shadow, no ground, no racket, no ball, no text.
Output: only this one figure on a fully transparent background."""


def _request_size(w: int, h: int, max_pixels: int = 4_000_000) -> tuple[int, int]:
    s = min(1.0, (max_pixels / (w * h)) ** 0.5)
    return int(w * s) // 16 * 16, int(h * s) // 16 * 16


def _cutout(path: Path):
    """The figure as RGBA cropped to its alpha bbox (alpha > 8)."""
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    a = np.asarray(im)[..., 3]
    ys, xs = np.nonzero(a > 8)
    if not len(ys):
        raise RuntimeError(f"{path}: no figure in the alpha channel")
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def _square_view(fig, height: int, out: Path) -> Path:
    """A cut-out scaled to `height` px, feet MARGIN above the bottom, centred on a square grey canvas."""
    from PIL import Image

    fig = fig.resize((max(1, round(fig.width * height / fig.height)), height), Image.LANCZOS)
    side = max(round(height * (1 + 2 * MARGIN)), fig.width + 2 * round(height * MARGIN))
    canvas = Image.new("RGBA", (side, side), (*CANVAS_GREY, 255))
    canvas.alpha_composite(fig, ((side - fig.width) // 2, side - round(height * MARGIN) - height))
    canvas.convert("RGB").save(out)
    return out


def draw_views(cfg: Config, force: bool = False) -> dict:
    """`character stylize` for an art-team variant: the missing views with Sunburst, then the four mesh views."""
    from PIL import Image

    run_ = cfg.run
    refs = {r["name"]: run_.refs_dir / r["path"] for r in read_json(run_.refs)}
    outfit = manifest(cfg)[1]
    sheet, attempts = run_.sheet_dir, run_.sheet_dir / "attempts"
    run_.ensure(attempts, run_.views_dir)
    report = read_json(run_.stylize_report) if run_.stylize_report.exists() and not force else {}
    report |= {"model": cfg.models.image_paint, "quality": QUALITY, "views": report.get("views", {})}
    front = refs["art_front"]
    size = _request_size(*Image.open(front).size)
    heads = {"back": ["art_head_back", "art_head_left"], "left": ["art_head_left", "art_head_front"],
             "right": ["art_head_right", "art_head_front"]}

    chosen: dict[str, Path] = {"front": front}
    for view in ORDER:
        out = sheet / f"{view}.png"
        if force or not out.exists() or view not in report["views"]:
            extra = [chosen["back"]] if view != "back" and "back" in chosen else []
            images = [front, refs["art_action"], *(refs[h] for h in heads[view]), *extra]
            with step(f"{view} view ({cfg.models.image_paint}, {CANDIDATES} candidates)"):
                res = fal.subscribe(cfg.models.image_paint, {
                    "image_urls": [fal.upload(p) for p in images], "prompt": prompt(view, bool(extra), outfit),
                    "quality": QUALITY, "background": "transparent", "output_format": "png",
                    "image_size": {"width": size[0], "height": size[1]}, "num_images": CANDIDATES})
            files = [fal.download(im["url"], attempts / f"{view}_{'abcdefgh'[i]}.png").name
                     for i, im in enumerate(res["images"])]
            report["views"][view] = {"candidates": files, "prompt": prompt(view, bool(extra), outfit),
                                     "references": [p.name for p in images]}
        pick = PICK.get(view) or report["views"][view]["candidates"][0]
        shutil.copy(attempts / pick, out)
        report["views"][view]["picked"] = pick
        chosen[view] = out

    figs = {v: _cutout(p) for v, p in chosen.items()}
    height = figs["front"].height
    for v, fig in figs.items():
        _square_view(fig, height, run_.views_dir / f"{v}.png")
    write_json(run_.views_dir / "facing.json", {v: f"{v}.png" for v in ("front", "left", "back", "right")})
    report["mesh_views"] = {v: str((run_.views_dir / f"{v}.png").relative_to(run_.root)) for v in figs}
    report["figure_height_px"] = height
    write_json(run_.stylize_report, report)
    log("views: " + ", ".join(f"{v} {figs[v].width}x{figs[v].height}" for v in figs) + f" → {run_.views_dir}")
    return report
