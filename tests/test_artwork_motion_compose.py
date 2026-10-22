"""Unit tests for the moveset reel (rybakina3d.artwork.motion_compose) and the site's motion block."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from rybakina3d import site
from rybakina3d.artwork import compose
from rybakina3d.artwork import motion_compose as mc
from rybakina3d.config import PROJECT_ROOT, Config
from rybakina3d.media import probe

PAPER = (240, 233, 224)
RED = (200, 30, 30)


def test_reel_timing_puts_every_contact_on_one_frame_and_loops():
    clips = [(58, 37), (81, 75), (58, 21)]
    t = mc.reel_timing(clips, 25)
    assert t.contact == 75 and t.motion == 75 + 36 + 1          # longest run-up + longest follow-through
    assert t.hold == 6 and t.tail == 12 and t.frames == t.motion + 18
    for n, c in clips:
        assert t.source(t.contact, n, c) == (c, 1.0)             # every clip hits the ball on the same reel frame
        assert t.source(0, n, c) == (0, 1.0)                     # ... starting from its first frame, held
        assert t.source(t.motion + t.hold - 1, n, c) == (n - 1, 1.0)   # and ending on its last, held
        assert t.source(t.contact + 1, n, c)[0] == min(c + 1, n - 1)
    # the tail fades the last frame out into the paper, then the first frame in, which frame 0 shows in full:
    # no cut at the loop point and never two poses at once
    tail = [t.source(f, 58, 21) for f in range(t.motion + t.hold, t.frames)]
    out, back = tail[:t.tail // 2], tail[t.tail // 2:]
    assert all(f == 57 for f, _ in out) and all(f == 0 for f, _ in back)
    ops = [o for _, o in out] + [-o for _, o in back]
    assert ops == sorted(ops, reverse=True) and out[0][1] > 0.9 and back[-1][1] > 0.9 and min(o for _, o in tail) < 0.1


def test_contact_and_resample_helpers():
    assert mc.clamp_contact(58, None) == 29 and mc.clamp_contact(58, 70) == 57 and mc.clamp_contact(58, -3) == 0
    frames = np.arange(50)[:, None, None, None] * np.ones((1, 2, 2, 3), np.uint8)
    out = mc.resample(frames, 81)
    assert len(out) == 81 and out[0, 0, 0, 0] == 0 and out[-1, 0, 0, 0] == 49
    assert mc.resample(frames, 50) is frames


def test_reel_layout_follows_the_atlas_rows_inside_the_frame():
    sizes = [12, 12, 12, 1, 2]
    lay = mc.reel_layout(sizes, columns=8, width=1920)
    assert lay.cells == compose.atlas_grid(sizes, 8)
    assert lay.size[0] == 1920 and lay.size[1] % 2 == 0 and lay.tile % 2 == 0
    assert max(r for r, _ in lay.cells) == 6                     # 8 + 4 per 12-clip group, slice beside backhands
    # the overhead pair leaves the last row's other columns to the key
    assert lay.key_in_grid and lay.key_x[0] > lay.boxes[-1][2] and lay.key_y < lay.size[1]
    for x0, y0, x1, y1 in lay.boxes:
        assert 0 <= x0 and x1 <= lay.size[0] and lay.title_h <= y0 and y1 < lay.size[1]
        assert x1 - x0 == y1 - y0 == lay.tile
    # a full last row: the key goes under the grid
    full = mc.reel_layout([8, 8], columns=8, width=1920)
    assert not full.key_in_grid and full.key_y > max(b[3] for b in full.boxes) and full.key_y < full.size[1]
    # no two tiles overlap, and each sits under its row's group header
    for i, a in enumerate(lay.boxes):
        for b in lay.boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_frame_paper_is_the_background_not_the_figure():
    f = np.empty((4, 100, 100, 3), np.uint8)
    f[:] = PAPER
    f[:, 20:95, 35:65] = (60, 50, 40)            # the figure
    f[:, 90:, 10:90] = (180, 175, 170)           # contact shadow along the bottom
    bg = mc.frame_paper(f)
    assert bg.shape == (4, 3) and np.allclose(bg, PAPER)
    assert mc.feather(50)[25, 25, 0] == 1.0 and mc.feather(50)[0, 0, 0] < 0.05


def test_paint_summary_reads_the_motion_paint_record():
    s = mc.paint_summary({"accepted": False, "model": "kling/v4", "iou": {"median": 0.8, "p10": 0.7, "min": 0.5},
                          "reason": "median IoU 0.8 < 0.85"})
    assert s == {"accepted": False, "iou_median": 0.8, "iou_p10": 0.7, "iou_min": 0.5, "model": "kling/v4",
                 "reason": "median IoU 0.8 < 0.85"}
    assert mc.paint_summary({"accepted": True, "iou_median": 0.9, "reason": "ok"})["reason"] is None


# --- end to end on a tiny run --------------------------------------------------------------------------
CLIPS = {"serve_00m01s": (20, 15), "serve_00m02s": (12, 4), "forehand_00m03s": (16, 8), "backhand_00m04s": (14, 7)}


def _clip_frames(n: int, contact: int, size: int = 64, tint: tuple = (0, 0, 0)) -> np.ndarray:
    """Paper with a dark block that walks across the frame; red on the contact frame only."""
    f = np.empty((n, size, size, 3), np.uint8)
    f[:] = np.clip(np.asarray(PAPER) + tint, 0, 255)
    for i in range(n):
        x = 8 + round(30 * i / max(n - 1, 1))
        f[i, 20:52, x:x + 18] = RED if i == contact else (70, 60, 50)
    return f


def _write(frames: np.ndarray, path, fps: float = 25) -> None:
    enc = mc.Encoder(path, (frames.shape[2], frames.shape[1]), fps, len(frames))
    for fr in frames:
        enc.write(fr)
    enc.close()


@pytest.fixture
def motion_run(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    cfg = Config.load()
    cfg.run_dir = tmp_path
    run = cfg.run
    ids = list(CLIPS)
    run.curation.write_text(json.dumps({"kept": ids, "clips": [
        {"id": c, "stroke": c.split("_")[0], "ok": c != "serve_00m02s", "contact_index": CLIPS[c][1]} for c in ids]}))
    for c, (n, contact) in CLIPS.items():
        (run.clips_dir / c).mkdir(parents=True)
        run.clip(c).meta.write_text(json.dumps({"id": c, "n_frames": n, "contact_index": contact, "src_fps": 25}))
    run.ensure(run.motion_renders, run.motion_painted)
    rows = {}
    for c, (n, contact) in CLIPS.items():
        if c == "backhand_00m04s":
            continue                                 # not rendered yet: left out of the reel, pending on the site
        frames = _clip_frames(n, contact)
        _write(frames, run.motion_renders / f"{c}.mp4")
        Image.fromarray(frames[contact]).save(run.motion_renders / f"{c}.jpg")
        rows[c] = {"n_frames": n, "contact_frame": contact, "fps": 25}
    (run.motion_renders / "renders.json").write_text(json.dumps({"blender": "5.2.2", "fps": 25, "clips": rows}))
    # an accepted painted clip from a model that made fewer frames, and a rejected one (its video must not be used)
    n, contact = CLIPS["serve_00m01s"]
    painted = _clip_frames(n, contact, tint=(8, -4, -12))[np.round(np.linspace(0, n - 1, 10)).astype(int)]
    _write(painted, run.motion_painted / "serve_00m01s.mp4")
    (run.motion_painted / "serve_00m01s.json").write_text(json.dumps({
        "clip": "serve_00m01s", "model": "kling/v4/video-to-video/edit", "accepted": True, "video": "serve_00m01s.mp4",
        "iou": {"median": 0.92, "p10": 0.88, "min": 0.8}, "reason": "ok"}))
    _write(_clip_frames(*CLIPS["forehand_00m03s"]), run.motion_painted / "forehand_00m03s.mp4")
    (run.motion_painted / "forehand_00m03s.json").write_text(json.dumps({
        "clip": "forehand_00m03s", "model": "kling/v4/video-to-video/edit", "accepted": False,
        "iou": {"median": 0.7, "p10": 0.5, "min": 0.3}, "reason": "median IoU 0.7 < 0.85"}))
    return cfg


def test_motion_compose_aligns_every_contact_and_reports(motion_run):
    cfg = motion_run
    run = cfg.run
    report = mc.run(cfg)
    assert report["painted"] == ["serve_00m01s"]
    assert report["rendered"] == ["forehand_00m03s", "serve_00m02s"] and report["missing"] == ["backhand_00m04s"]
    t = report["timing"]
    assert t["contact_frame"] == 15 and t["frames"] == 15 + 7 + 1 + round(mc.HOLD_S * 25) + round(mc.TAIL_S * 25)
    c = report["clips"]
    assert c["serve_00m01s"]["painted_frames_resampled"] == [10, 20]
    assert c["serve_00m02s"]["flagged"] is True and c["serve_00m02s"]["start_frame"] == 11
    for key, path in (("reel", run.moveset_reel), ("reel_render", run.moveset_reel_render)):
        assert report[key]["path"] == path.name and path.exists() and mc.poster_path(path).exists()
        frames = mc.decode(path, 64)
        assert len(frames) == t["frames"]
    # the poster is the contact instant: every tile shows its red contact frame at once
    poster = np.asarray(Image.open(mc.poster_path(run.moveset_reel_render)).convert("RGB"), np.int16)
    assert list(poster.shape[1::-1]) == report["layout"]["size"]
    for cid, e in c.items():
        x0, y0, x1, y1 = e["box_px"]
        tile = poster[y0:y1, x0:x1]
        red = (tile[..., 0] - tile[..., 1] > 100).mean()
        assert red > 0.05, cid
    # light per-clip copies for the site, in step with the render (the painted one too)
    tiles = mc.tiles_dir(cfg)
    assert sorted(p.name for p in tiles.iterdir()) == sorted(
        [f"{x}.{s}" for x in ("serve_00m01s", "serve_00m02s", "forehand_00m03s") for s in ("mp4", "jpg")]
        + ["serve_00m01s_painted.mp4", "serve_00m01s_painted.jpg"])
    assert len(mc.decode(tiles / "serve_00m01s_painted.mp4", 32)) == 20
    assert json.loads(run.motion_report.read_text())["reel"]["frames"] == t["frames"]


def test_site_motion_block(motion_run, tmp_path):
    cfg = motion_run
    link = site._Linker(tmp_path / "site")
    # before motion-compose: the clips from the renders, played from the full videos
    m = site._artwork_motion(cfg, link)
    assert m["reel"] is None and m["reel_render"] is None and m["turntable"] is None
    assert [c["id"] for c in m["clips"]] == ["serve_00m01s", "serve_00m02s", "forehand_00m03s", "backhand_00m04s"]
    first = m["clips"][0]
    assert first["render"]["tile"] == first["render"]["url"] == "media/artwork/motion/render/serve_00m01s.mp4"
    assert m["n_clips"] == 4 and m["n_rendered"] == 3 and m["n_painted"] == 1 and m["n_flagged"] == 1
    mc.run(cfg)
    m = site._artwork_motion(cfg, link)
    assert m["reel"]["url"] == "media/artwork/motion/rybakina_moveset_reel.mp4"
    assert m["reel"]["poster"] == "media/artwork/motion/rybakina_moveset_reel.jpg" and m["reel"]["frames"] > 0
    assert m["reel_stale"] is False
    by = {c["id"]: c for c in m["clips"]}
    s1 = by["serve_00m01s"]
    assert s1["stroke"] == "serve" and s1["flagged"] is False
    assert s1["render"] == {"url": "media/artwork/motion/render/serve_00m01s.mp4",
                            "tile": "media/artwork/motion/tiles/serve_00m01s.mp4",
                            "poster": "media/artwork/motion/tiles/serve_00m01s.jpg",
                            "poster_full": "media/artwork/motion/render/serve_00m01s.jpg",
                            "n_frames": 20, "contact_frame": 15}
    assert s1["painted"]["url"] == "media/artwork/motion/painted/serve_00m01s.mp4"
    assert s1["painted"]["tile"] == "media/artwork/motion/tiles/serve_00m01s_painted.mp4"
    assert (s1["painted"]["iou_median"], s1["painted"]["iou_p10"]) == (0.92, 0.88) and s1["paint"] is None
    fh = by["forehand_00m03s"]
    assert fh["painted"] is None and fh["paint"]["reason"] == "median IoU 0.7 < 0.85"
    assert by["serve_00m02s"]["flagged"] is True and by["backhand_00m04s"]["render"] is None
    for c in m["clips"]:                         # every linked file is really in the site
        for v in (c["render"] or {}, c["painted"] or {}):
            for key in ("url", "tile", "poster"):
                if v.get(key):
                    assert (tmp_path / "site" / v[key]).exists(), v[key]
    # a clip rendered again after the reel: its site copy is stale, the full video plays and the reel is flagged
    render = cfg.run.motion_renders / "serve_00m02s.mp4"
    tile = mc.tiles_dir(cfg) / "serve_00m02s.mp4"
    os.utime(render, (tile.stat().st_mtime + 10, tile.stat().st_mtime + 10))
    m = site._artwork_motion(cfg, link)
    s2 = next(c for c in m["clips"] if c["id"] == "serve_00m02s")
    assert s2["render"]["tile"] == s2["render"]["url"] and m["reel_stale"] is True
    # no artwork sheets yet: the artwork block still carries the motion
    assert set(site._artwork(cfg, link)) == {"motion"}


def test_site_motion_block_is_none_without_motion(tmp_path):
    cfg = Config.load()
    cfg.run_dir = tmp_path
    assert site._artwork_motion(cfg, site._Linker(tmp_path / "site")) is None
    assert site._artwork(cfg, site._Linker(tmp_path / "site")) is None


# --- the Quest 3 reel (non-web render profiles) ---------------------------------------------------------
MOVESET = [12, 12, 12, 1, 2]       # this run's groups: serve, forehand, backhand, slice, overhead


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def test_fixed_canvas_layout_fits_the_quest3_eye_without_cropping():
    W, H = mc.REEL_EYE
    assert (W, H) == (2064, 2208)
    natural = mc.reel_layout(MOVESET, columns=8, width=W)
    assert natural.size[1] > H                                   # ~2270 px: the grid has to give way
    lay = mc.reel_layout(MOVESET, columns=8, width=W, height=H)
    assert lay.size == (W, H) and lay.cells == natural.cells and len(lay.boxes) == 39
    # compacted, not cropped: the largest even tile whose grid and key fit
    assert lay.tile % 2 == 0 and lay.tile < natural.tile
    assert mc._grid_layout(MOVESET, 8, W, lay.tile + 2).size[1] > H
    assert (lay.head_h, lay.cap_h) == (round(lay.tile * 0.18), round(lay.tile * 0.12))
    for i, (x0, y0, x1, y1) in enumerate(lay.boxes):
        assert lay.margin <= x0 and x1 <= W - lay.margin and lay.title_h + lay.head_h <= y0
        assert y1 + lay.cap_h <= H                               # its caption line too
        assert x1 - x0 == y1 - y0 == lay.tile
        assert not any(_overlap(lay.boxes[i], b) for b in lay.boxes[i + 1:])
    # the key (flag line + timing line) in the last row's free columns, inside the canvas, clear of every tile
    assert lay.key_in_grid
    key = lay.key_box(2)
    assert lay.margin <= key[0] and key[2] <= W - lay.margin and lay.title_h <= key[1] and key[3] <= H
    assert not any(_overlap(key, b) for b in lay.boxes)
    # a full last row puts the key under the grid; still inside
    full = mc.reel_layout([8, 8, 8, 8, 8, 8, 8], columns=8, width=W, height=H)
    assert full.size == (W, H) and not full.key_in_grid and full.key_box(1)[3] <= H
    assert not any(_overlap(full.key_box(1), b) for b in full.boxes)
    # the web reel is untouched by any of this
    assert mc.reel_layout(MOVESET) == mc.reel_layout(MOVESET, columns=8, width=1920)
    assert mc.reel_layout(MOVESET).size == (1920, 2114) and mc.reel_layout(MOVESET).tile == 216
    with pytest.raises(ValueError):
        mc.reel_layout(MOVESET, columns=8, width=W, height=301)
    with pytest.raises(ValueError):
        mc.reel_layout(MOVESET, columns=8, width=W, height=H + 1)


def test_fixed_canvas_reel_draws_every_label_inside_the_canvas():
    """Title, captions, group headers, clip captions, flag marks and key: drawn inside the margins, never over a
    tile, the key inside its box, the caption clear of the title."""
    W, H = mc.REEL_EYE
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    strokes = ["serve", "forehand", "backhand", "slice", "overhead"]
    groups = [(s, [mc.ClipMotion(f"{s}_{i:02d}m{(7 * i) % 60:02d}s", s, i % 3 == 0) for i in range(n)])
              for s, n in zip(strokes, MOVESET, strict=True)]
    lay = mc.reel_layout(MOVESET, width=W, height=H)
    clips = [c for _, cs in groups for c in cs]
    tiles = {c.id: mc.Tile(c, np.zeros((1, 2, 2, 3), np.uint8), np.zeros((1, 3), np.float32), 0, box, "render")
             for c, box in zip(clips, lay.boxes, strict=True)}
    timing = mc.reel_timing([(137, 70), (195, 180)], 60)
    lines = mc.caption_lines(cfg, len(tiles), 60)
    img, extra = mc.base_image(lay, groups, tiles, timing, 60, painted=False, title=cfg.display_name, lines=lines)
    assert img.size == (W, H)
    drawn = np.abs(np.asarray(img, np.int16) - np.asarray(compose.paper((W, H), seed=17), np.int16)).max(-1) > 10
    ys, xs = np.nonzero(drawn)
    assert len(xs) and xs.min() >= lay.margin // 2 and xs.max() < W - lay.margin // 2
    assert ys.min() > 0 and ys.max() < H - 4
    for x0, y0, x1, y1 in lay.boxes:
        assert not drawn[y0:y1, x0:x1].any()
    kx0, ky0, kx1, ky1 = lay.key_box(2)
    assert drawn[ky0:ky1, kx0:kx1].any() and not drawn[ky0:ky1, kx1:].any() and not drawn[ky1:, kx0:].any()
    assert extra["caption_x0"] > extra["title_x1"]


def test_caption_names_whose_motion_a_variant_plays():
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    assert cfg.display_name == "Naomi Osaka"
    assert mc.caption_lines(cfg, 39, 60) == [
        "39 STROKES CAPTURED FROM ELENA RYBAKINA · ALIGNED ON CONTACT · REAL TIME, 60 FPS",
        "3D RENDERS OF THE GAME-READY CHARACTER"]
    base = Config.load()
    assert mc.caption_lines(base, 39, 60)[0] == "39 CAPTURED STROKES · ALIGNED ON CONTACT · REAL TIME, 60 FPS"


def test_contain_fits_a_portrait_frame_whole():
    assert mc.fit_size((2064, 2208), 224) == (209, 224)
    assert mc.fit_size((2064, 2208), mc.TILE_WEB, even=True) == (336, 360)
    assert mc.fit_size((516, 552), mc.TILE_WEB, even=True) == (336, 360)
    assert mc.fit_size((1024, 1024), 216) == (216, 216)
    src = np.empty((3, 92, 86, 3), np.uint8)                 # 2064 x 2208 / 24
    src[:] = PAPER
    src[:, 30:60, :6] = RED                                  # touching the left edge
    src[:, 30:60, -6:] = (60, 50, 40)                        # and the right
    src[:, :5, 40:46] = (60, 50, 40)                         # a racket at the very top
    w, h = mc.fit_size((86, 92), 64)
    small = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in src])
    bg = mc.frame_paper(small)
    assert np.allclose(bg, PAPER, atol=1)
    tile = mc.contain(small, 64, bg)
    assert tile.shape == (3, 64, 64, 3) and (w, h) == (60, 64)
    x0 = (64 - w) // 2
    assert np.array_equal(tile[:, :, x0:x0 + w], small)     # the whole picture, nothing cut
    assert (tile[:, 34:40, x0, 0] > 150).all() and (tile[:, 34:40, x0, 1] < 100).all()
    assert (tile[:, 0, x0 + 28:x0 + 31] < 150).all()        # the racket's top row is still there
    pad = np.concatenate([tile[:, :, :x0], tile[:, :, x0 + w:]], axis=2)
    assert (pad == np.asarray(PAPER, np.uint8)).all()       # padded with the frame's own paper ...
    ratio = pad.astype(np.float32) / bg[:, None, None, :]
    assert np.abs(ratio - 1).max() < 0.005                   # ... which the paper-ratio compositing makes invisible
    # the feather follows the picture's edges (zero over the padding); the square case is the web reel's
    f = mc.feather(64, (h, w))
    assert f[:, :x0].max() == 0 and f[:, x0 + w:].max() == 0 and f[32, 32, 0] == 1.0 and f[32, x0, 0] < 0.2
    assert np.array_equal(mc.feather(64, (64, 64)), mc.feather(64))


# a portrait run at 60 fps for the `quest3` profile, on the naomi variant (frames are 2064 x 2208 / 24)
QCLIPS = {"serve_00m01s": (40, 30), "serve_00m02s": (24, 8), "forehand_00m03s": (32, 16), "backhand_00m04s": (28, 14)}


def _portrait_frames(n: int, contact: int, w: int = 86, h: int = 92) -> np.ndarray:
    """Paper with a dark block that walks across the frame; red on the contact frame, with a red bar along the top
    edge (the rows a square cover crop of a portrait frame would cut)."""
    f = np.empty((n, h, w, 3), np.uint8)
    f[:] = PAPER
    for i in range(n):
        x = round(40 * i / max(n - 1, 1))
        f[i, 20:80, x:x + 26] = RED if i == contact else (70, 60, 50)
    f[contact, :3, 10:76] = RED
    return f


@pytest.fixture
def quest3_run(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    cfg.run_dir = tmp_path
    run = cfg.run
    assert run.out == tmp_path / "variants" / "naomi"
    ids = list(QCLIPS)
    run.curation.write_text(json.dumps({"kept": ids, "clips": [
        {"id": c, "stroke": c.split("_")[0], "ok": c != "serve_00m02s", "contact_index": 0} for c in ids]}))
    for c, (n, _) in QCLIPS.items():
        (run.clips_dir / c).mkdir(parents=True)
        run.clip(c).meta.write_text(json.dumps({"id": c, "n_frames": n, "contact_index": 0, "src_fps": 25}))
    rdir = run.motion_renders_for("quest3")
    run.ensure(rdir, run.motion_painted)
    rows = {}
    for c, (n, contact) in QCLIPS.items():
        if c == "backhand_00m04s":
            continue                                 # not rendered: left out
        frames = _portrait_frames(n, contact)
        _write(frames, rdir / f"{c}.mp4", fps=60)
        Image.fromarray(frames[contact]).save(rdir / f"{c}.jpg")
        rel = str((rdir / c).relative_to(tmp_path))
        rows[c] = {"id": c, "video": f"{rel}.mp4", "poster": f"{rel}.jpg", "n_frames": n, "contact_frame": contact,
                   "fps": 60.0, "size": [86, 92]}
    (rdir / "renders.json").write_text(json.dumps({"profile": "quest3", "fps": 60.0, "clips": rows}))
    # an accepted painted clip: a render-only reel never uses it
    _write(_portrait_frames(*QCLIPS["serve_00m01s"])[:, :, :, ::-1], run.motion_painted / "serve_00m01s.mp4", 60)
    (run.motion_painted / "serve_00m01s.json").write_text(json.dumps({"accepted": True, "video": "serve_00m01s.mp4"}))
    return cfg


def test_quest3_reel_on_the_fixed_canvas_and_its_report(quest3_run):
    cfg = quest3_run
    run = cfg.run
    with pytest.raises(ValueError):
        mc.run(cfg, profile="vr")
    report = mc.run(cfg, profile="quest3")
    assert set(report) == {"profile", "name", "motion_from", "size", "fps", "reel", "groups", "clips", "made"}
    assert (report["profile"], report["name"], report["motion_from"]) == ("quest3", "Naomi Osaka", "Elena Rybakina")
    assert report["size"] == [2064, 2208] and report["fps"] == 60
    assert report["groups"] == [{"stroke": "serve", "clips": ["serve_00m01s", "serve_00m02s"]},
                                {"stroke": "forehand", "clips": ["forehand_00m03s"]}]
    # timing in seconds at 60 fps: contacts on frame 30, the longest follow-through 15 frames, then hold + tail
    frames = 30 + 15 + 1 + round(mc.HOLD_S * 60) + round(mc.TAIL_S * 60)
    reel = report["reel"]
    assert set(reel) == {"mp4", "poster", "frames", "loop_s", "mb"} and reel["frames"] == frames
    assert reel["mp4"] == "variants/naomi/artwork/motion/naomi_moveset_reel_quest3.mp4"
    assert reel["poster"] == "variants/naomi/artwork/motion/naomi_moveset_reel_quest3.jpg"
    assert reel["loop_s"] == round(frames / 60, 3)
    p = probe(run.root / reel["mp4"])
    assert (p.width, p.height, p.fps, p.frames) == (2064, 2208, 60, frames)
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of",
                                      "json", str(run.root / reel["mp4"])], capture_output=True, text=True,
                                     check=True).stdout)["streams"][0]
    assert (info["profile"], info["level"], info["pix_fmt"]) == ("High", 52, "yuv420p")
    assert (info["color_space"], info["color_primaries"], info["color_transfer"]) == ("bt709",) * 3
    clips = {c["id"]: c for c in report["clips"]}
    assert list(clips) == ["serve_00m01s", "serve_00m02s", "forehand_00m03s"]
    keys = {"id", "stroke", "flagged", "mp4", "poster", "tile", "tile_poster", "size", "fps", "frames",
            "contact_frame", "mb"}
    for cid, c in clips.items():
        assert set(c) == keys
        assert c["mp4"] == f"variants/naomi/artwork/motion/render_quest3/{cid}.mp4"
        assert c["tile"] == f"variants/naomi/artwork/motion/naomi_moveset_tiles_quest3/{cid}.mp4"
        for k in ("mp4", "poster", "tile", "tile_poster"):
            assert (run.root / c[k]).exists(), c[k]
        assert (c["size"], c["fps"], c["frames"], c["contact_frame"]) == ([86, 92], 60, *QCLIPS[cid])
        t = probe(run.root / c["tile"])                     # the light copy keeps the clip's aspect
        assert (t.width, t.height, t.frames) == (336, 360, QCLIPS[cid][0])
    assert clips["serve_00m02s"]["flagged"] is True and clips["serve_00m01s"]["flagged"] is False
    assert sorted(p.name for p in mc.tiles_dir(cfg, "quest3").iterdir()) == sorted(
        f"{c}.{s}" for c in clips for s in ("mp4", "jpg"))
    # the poster is the contact instant: every tile shows its red contact frame, the top edge included (contained)
    poster = np.asarray(Image.open(run.root / reel["poster"]).convert("RGB"), np.int16)
    assert poster.shape == (2208, 2064, 3)
    lay = mc.reel_layout([2, 1], width=2064, height=2208)
    for (x0, y0, x1, y1), cid in zip(lay.boxes, clips, strict=True):
        tile = poster[y0:y1, x0:x1]
        red = tile[..., 0] - tile[..., 1] > 100
        assert red.mean() > 0.05, cid
        top = (tile[..., 0] - tile[..., 1] > 40).any(1).nonzero()[0]
        assert top.min() < lay.tile * 0.05, cid                          # the bar along the frame's top edge
    report_file = json.loads(mc.profile_report(cfg, "quest3").read_text())
    assert report_file["reel"] == reel and report_file["clips"] == report["clips"]
    # render-only: the web reels and report are not made, and no painted copy appears
    assert not run.moveset_reel.exists() and not run.motion_report.exists()


def test_quest3_reel_composes_while_the_render_runs(quest3_run):
    """motion-render writes renders.json when its run ends: clips already encoded compose at the profile's rate
    (not the clips' 25 fps), their contacts taken from curation and put on the output frame motion-render's
    frame_times puts them on."""
    from rybakina3d.blender.artwork_motion import frame_times

    cfg = quest3_run
    run = cfg.run
    rdir = run.motion_renders_for("quest3")
    (rdir / "renders.json").unlink()
    src = {"serve_00m01s": (13, 5), "serve_00m02s": (15, 10), "forehand_00m03s": (8, 3)}   # (frames, contact) at 25
    cur = json.loads(run.curation.read_text())
    for r in cur["clips"]:
        r["contact_index"] = src.get(r["id"], (0, 0))[1]
    run.curation.write_text(json.dumps(cur))
    want = {}
    for cid, (n, c) in src.items():
        times, k = frame_times(n, c, 25, 60)
        want[cid] = (len(times), k)
        _write(_portrait_frames(len(times), k), rdir / f"{cid}.mp4", fps=60)
    assert want == {"serve_00m01s": (31, 12), "serve_00m02s": (36, 24), "forehand_00m03s": (19, 7)}
    report = mc.run(cfg, profile="quest3")
    assert report["fps"] == 60
    clips = {c["id"]: c for c in report["clips"]}
    assert {cid: (c["frames"], c["contact_frame"]) for cid, c in clips.items()} == want
    assert all(c["fps"] == 60 for c in clips.values())
    # the latest contact (24), the longest follow-through (31 - 1 - 12), then hold + tail
    assert report["reel"]["frames"] == 24 + 18 + 1 + round(mc.HOLD_S * 60) + round(mc.TAIL_S * 60)
    poster = np.asarray(Image.open(run.root / report["reel"]["poster"]).convert("RGB"), np.int16)
    lay = mc.reel_layout([2, 1], width=2064, height=2208)
    for (x0, y0, x1, y1), cid in zip(lay.boxes, clips, strict=True):
        tile = poster[y0:y1, x0:x1]
        assert (tile[..., 0] - tile[..., 1] > 100).mean() > 0.05, cid      # every contact on the poster
