"""Unit tests for the artwork paint check and the sheet composition (rybakina3d.artwork.paint / .compose)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from rybakina3d.artwork import compose, paint
from rybakina3d.artwork.layout import SHEET_SIZE, SLOTS
from rybakina3d.config import Config


def _figure_mask(h: int = 200, w: int = 120, top: int = 20, bottom: int = 180) -> np.ndarray:
    """A stick figure: head, torso, two legs, one arm out (thin parts make the IoU honest)."""
    m = np.zeros((h, w), bool)
    cx = w // 2
    m[top:top + 24, cx - 10:cx + 10] = True                    # head
    m[top + 24:top + 90, cx - 18:cx + 18] = True               # torso
    m[top + 90:bottom, cx - 16:cx - 4] = True                  # legs
    m[top + 90:bottom, cx + 4:cx + 16] = True
    m[top + 34:top + 42, cx + 18:min(w, cx + 55)] = True       # arm
    return m


def _write_figure(path, mask: np.ndarray, shadow: bool = True) -> None:
    """RGBA render like `artwork render` writes: grey figure + a soft dark ellipse under the feet."""
    h, w = mask.shape
    a = mask.astype(np.float32)
    if shadow:
        ys, xs = np.nonzero(mask)
        yy, xx = np.mgrid[0:h, 0:w]
        ell = np.exp(-(((xx - xs.mean()) / (0.4 * (np.ptp(xs) + 1))) ** 2 + ((yy - ys.max()) / 4.0) ** 2))
        a = np.maximum(a, 0.4 * ell * (~mask))
    rgb = np.where(mask[..., None], 180, 40).astype(np.uint8).repeat(3, axis=2)
    rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])
    Image.fromarray(rgba, "RGBA").save(path)
    matte = np.zeros((h, w, 4), np.uint8)
    matte[..., 3] = mask * 255
    Image.fromarray(matte, "RGBA").save(str(path).replace(".png", "_matte.png"))


def test_aligned_iou_ignores_shift_and_scale_but_not_pose():
    ref = _figure_mask()
    shifted = np.roll(np.roll(ref, 9, axis=0), -7, axis=1)
    assert paint.aligned_iou(shifted, ref) == pytest.approx(1.0)
    big = np.asarray(Image.fromarray(ref.astype(np.uint8) * 255).resize((180, 300), Image.NEAREST)) > 127
    assert paint.aligned_iou(big, ref) > 0.9
    moved = ref.copy()
    moved[54:62, 78:115] = False                    # the arm swings down instead of out
    moved[62:140, 78:86] = True
    assert paint.aligned_iou(moved, ref) < 0.85
    assert paint.aligned_iou(np.zeros_like(ref), ref) == 0.0


def test_clean_mask_fills_holes_and_drops_specks():
    m = np.zeros((300, 300), bool)
    m[50:250, 50:250] = True
    m[120:180, 120:180] = False      # racket head / enclosed gap
    m[10:13, 10:13] = True           # speck
    out = paint.clean_mask(m)
    assert out[150, 150] and not out[11, 11]


def test_similarity_sets_whole_figures_on_the_feet_line_and_leaves_crops_alone():
    s, tx, ty = paint.similarity((40, 20, 80, 120), (30, 10, 90, 160), (200, 200))
    assert s == pytest.approx(1.5)
    assert 120 * s + ty == pytest.approx(160)                  # feet on the render's feet line
    assert (40 + 80) / 2 * s + tx == pytest.approx(60)         # centred on the render's figure
    assert paint.similarity((0, 30, 200, 200), (0, 50, 200, 200), (200, 200)) == (1.0, 0.0, 0.0)  # a bust


def test_request_size_respects_the_pixel_budget():
    for w, h in ((326, 397), (1574, 1741), (4000, 3000)):
        rw, rh = paint.request_size(w, h)
        assert rw % 16 == 0 and rh % 16 == 0
        assert 655_360 <= rw * rh <= 8_294_400 and max(rw, rh) <= 3840
        assert abs(rw / rh - w / h) < 0.05


def test_paper_colour_is_the_reference_paper():
    r, g, b = paint.paper_colour()
    assert 240 <= r <= 255 and r >= g >= b and r - b < 20   # warm off-white


def test_shared_scales_give_one_physical_size():
    a = paint.Figure("a", None, size=(100, 200), bbox=(10, 10, 90, 190), ground_y=190, px_per_m=100.0)
    b = paint.Figure("b", None, size=(100, 400), bbox=(10, 10, 90, 390), ground_y=390, px_per_m=200.0)
    box = (0, 0, 1000, 1000)
    sa, sb = compose.shared_scales([a, b], [compose.fit_standing(a, box), compose.fit_standing(b, box)], min)
    assert sa * a.px_per_m == pytest.approx(sb * b.px_per_m)     # same sheet px per metre
    placed = compose.stand(a, box, sa)
    assert placed.y + sa * a.ground_y == pytest.approx(1000 * (1 - compose.GROUND_MARGIN))
    assert placed.x + sa * 50 == pytest.approx(500)


def test_compose_lays_out_sheets_and_atlas(tmp_path):
    cfg = Config.load()
    cfg.run_dir = tmp_path
    run = cfg.run
    renders = run.artwork_renders
    (renders / "atlas").mkdir(parents=True)
    rows = {"slots": {}, "atlas": {}}
    for s in SLOTS:
        if s.name == "face_4":
            continue                               # a slot without a render stays empty
        _write_figure(renders / f"{s.name}.png", _figure_mask(), shadow=s.framing == "full")
        rows["slots"][s.name] = {"png": f"artwork/renders/{s.name}.png",
                                 "matte": f"artwork/renders/{s.name}_matte.png", "ground_y": 182.0}
    clips = ["serve_00m01s", "serve_00m02s", "forehand_00m03s", "backhand_00m04s"]
    for c in clips:
        _write_figure(renders / "atlas" / f"{c}.png", _figure_mask())
        rows["atlas"][c] = {"png": f"artwork/renders/atlas/{c}.png", "matte": f"artwork/renders/atlas/{c}_matte.png"}
    (renders / "renders.json").write_text(json.dumps(rows))
    run.curation.write_text(json.dumps({"kept": clips, "clips": [
        {"id": c, "stroke": c.split("_")[0], "ok": c != "serve_00m02s"} for c in clips]}))
    # one accepted repaint of the current render: compose must take its cut-out, the render elsewhere; a repaint
    # of an older render (back) is stale and must not be laid out
    run.artwork_painted.mkdir(parents=True)
    for name in ("hero", "back"):
        Image.open(renders / f"{name}.png").save(run.artwork_painted / f"{name}_cut.png")
    figs, _ = paint.load_renders(cfg)
    hero = next(s for s in SLOTS if s.name == "hero")
    (run.artwork_painted / "paint.json").write_text(json.dumps({"model": "m", "slots": {
        "hero": {"accepted": True, "iou": 0.9, "cutout": "hero_cut.png",
                 "fingerprint": paint.fingerprint(cfg, hero, figs["hero"])},
        "back": {"accepted": True, "iou": 0.9, "cutout": "back_cut.png", "fingerprint": "older-render"},
        "front": {"accepted": False, "iou": 0.5}}}))

    report = compose.run(cfg)
    for path in (run.character_sheet, run.character_sheet_render, run.moveset_atlas):
        with Image.open(path) as im:
            assert im.size == SHEET_SIZE
        with Image.open(compose.preview_path(path)) as pv:
            assert pv.width == compose.PREVIEW_W
    assert report["painted"] == ["hero"]
    assert "front" in report["rendered"] and "back" in report["rendered"] and report["missing"] == ["face_4"]
    assert report["figures"]["back"]["paint"]["stale"] is True
    assert report["atlas_figures"]["serve_00m02s"]["flagged"] is True
    assert json.loads(run.artwork_report.read_text())["figures"]["sil_1"]["source"] == "silhouette"
    # the figures really landed in their boxes: the hero's box is darker than the empty paper above the card
    sheet = np.asarray(Image.open(run.character_sheet).convert("L"), np.float32)
    hero = next(s for s in SLOTS if s.name == "hero")
    x0, y0, x1, y1 = compose._px(hero.box, SHEET_SIZE)
    assert sheet[y0:y1, x0:x1].min() < 200 < np.median(sheet[y0:y1, x0:x1])


def test_atlas_grid_gives_each_stroke_group_its_own_row():
    # 12 serves fill a row, forehands start the next; slice (1) and overhead (2) share the last row, one column apart
    cells = compose.atlas_grid([12, 12, 12, 1, 2], 12)
    assert cells[11] == (0, 11) and cells[12] == (1, 0) and cells[24] == (2, 0)
    assert cells[36:] == [(3, 0), (3, 2), (3, 3)]
    # a group longer than a row wraps onto the next one
    assert compose.atlas_grid([3, 5], 4) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0)]


def test_paint_prompt_closes_the_hands_on_the_grip():
    by = {s.name: s for s in SLOTS}
    assert paint.grip_of(by["hero"]) == "two_handed"          # the hero is a two-handed backhand
    assert paint.grip_of(by["action_b"]) == "racket"
    assert paint.grip_of(by["front"]) == "" and paint.grip_of(by["hero"], "one_handed") == "racket"
    assert "both hands hold the racket handle" in paint.prompt("hero", grip="two_handed")
    assert "racket grip" not in paint.prompt("standing")
