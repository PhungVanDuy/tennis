"""Unit tests for the moveset reel (rybakina3d.artwork.motion_compose) and the site's motion block."""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
from PIL import Image

from rybakina3d import site
from rybakina3d.artwork import compose
from rybakina3d.artwork import motion_compose as mc
from rybakina3d.config import Config

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


def _write(frames: np.ndarray, path) -> None:
    enc = mc.Encoder(path, (frames.shape[2], frames.shape[1]), 25, len(frames))
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
