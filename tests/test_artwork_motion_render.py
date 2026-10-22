"""The moving artwork (rybakina3d.artwork.motion_render and the numpy half of blender/artwork_motion.py): framing a
whole motion in one fixed camera, the floor under a jump, the turntable orbit, the paper composite, fingerprints
and the renders.json rows; the render profiles (web as before, the Quest 3 portrait frame at 60 fps, its draft):
their timing, framing, scaled pixel constants, encoding and separate folders. Nothing here needs Blender."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rybakina3d.artwork import motion_render as mr
from rybakina3d.artwork.layout import MOTION, PROFILES
from rybakina3d.blender import artwork as still
from rybakina3d.blender import artwork_motion as bm
from rybakina3d.config import Config

PAPER = (251, 246, 240)


def _moving_figure(n: int = 40) -> list[np.ndarray]:
    """A 1.7 m box-ish figure that lunges 1.2 m to its side and swings a racket up over the clip."""
    rng = np.random.default_rng(1)
    body = rng.uniform([-0.25, -0.15, 0.0], [0.25, 0.15, 1.7], size=(400, 3))
    frames = []
    for i in range(n):
        t = i / (n - 1)
        racket = np.array([[0.3 + 0.4 * t, -0.3, 1.0 + 1.5 * t], [0.5, -0.4, 1.2 + 1.2 * t]])
        frames.append(np.concatenate([body, racket]) + [1.2 * t, 0.0, 0.0])
    return frames


# ------------------------------------------------------------------------------------- framing (numpy)
def test_one_square_frame_holds_every_frame_of_the_motion_inside_the_margins():
    frames = _moving_figure()
    pts = np.concatenate(frames)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    target, distance = (lo + hi) / 2, still.DISTANCE_PER_SIZE * float(np.max(hi - lo))
    cam = still.camera_matrix(target, -math.pi / 2, MOTION.azimuth, MOTION.elevation, distance)
    frame = bm.fit_square(still.project(pts, cam), MOTION.size, still.MARGINS["full"])
    W, H = MOTION.size
    assert (frame["width"], frame["height"]) == (W, H)
    assert math.isclose(frame["lens_mm"], 24.0 * frame["f_px"] / H)          # vertical sensor fit
    for f in frames:                                                          # nobody leaves the frame
        px = still.to_pixels(still.project(f, cam), frame)
        assert px.min() > 0 and px[:, 0].max() < W and px[:, 1].max() < H
    px = still.to_pixels(still.project(pts, cam), frame)
    span = px[:, 1].max() - px[:, 1].min()
    top, side, bottom = (m * span for m in still.MARGINS["full"])
    left, right, up, down = px[:, 0].min(), W - px[:, 0].max(), px[:, 1].min(), H - px[:, 1].max()
    if math.isclose(up, top, rel_tol=1e-6):        # height-bound: the stills' margins top and bottom, centred across
        assert math.isclose(down, bottom, rel_tol=1e-6) and math.isclose(left, right) and left >= side - 1e-6
    else:                                          # width-bound: side margins, the slack shared top and bottom
        assert math.isclose(left, side, rel_tol=1e-6) and math.isclose(right, side, rel_tol=1e-6)
        assert math.isclose(up - top, down - bottom, abs_tol=1e-6) and up > top


def test_fit_square_keeps_the_stills_lens_and_shift_conventions():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.4, -0.2, 0.0], [0.5, 0.3, 1.8], size=(500, 3))
    cam = still.camera_matrix(np.array([0.05, 0.05, 0.9]), 0.3, 25, 5, 7.0)
    uv = still.project(pts, cam)
    sq = bm.fit_square(uv, (1000, 1000), (0.04, 0.05, 0.03))
    tall = still.fit_frame(uv, 1000, (0.04, 0.05, 0.03))
    # a figure taller than wide: the square frame is the still's frame with slack at the sides
    assert tall["width"] < 1000
    assert math.isclose(sq["f_px"], tall["f_px"]) and math.isclose(sq["lens_mm"], tall["lens_mm"])
    assert math.isclose(sq["shift_y"], tall["shift_y"])
    assert np.allclose(still.to_pixels(uv, sq)[:, 1], still.to_pixels(uv, tall)[:, 1])


def test_floor_follows_her_lowest_vertex_on_the_ground_and_bridges_a_jump():
    lowest = np.array([-0.02, -0.03, -0.01, 0.20, 0.45, 0.30, 0.05, -0.04, -0.04])
    floor = bm.floor_track(lowest)
    down = lowest < still.GROUNDED_M
    assert np.allclose(floor[down], lowest[down])                       # feet exactly on the floor
    assert np.allclose(floor[3:6], np.interp([3, 4, 5], [2, 6], [-0.01, 0.05]))   # across the jump
    assert np.all(floor <= lowest + 1e-12)                              # never cuts a foot
    assert np.allclose(bm.floor_track(np.array([0.3, 0.25])), 0.0)      # never comes down: court level
    held = bm.floor_track(np.array([0.4, 0.3, -0.02, 0.3]))
    assert np.allclose(held, [-0.02, -0.02, -0.02, -0.02])              # held before / after


def test_turntable_closes_the_loop_and_the_ground_box_holds_the_shadow():
    az = bm.orbit_azimuths(150)
    assert len(az) == 150 and az[0] == 0 and math.isclose(az[1], 2.4) and math.isclose(az[-1] + 2.4, 360)
    assert math.isclose(bm.orbit_azimuths(4, 30)[1], 120)
    feet = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    target = np.array([0.0, 0.0, 0.9])
    cams = [still.camera_matrix(target, 0.0, a, 8, 7.0) for a in (0, 90, 180)]
    pts = np.concatenate([np.array([[0, 0, 0], [0, 0, 1.8]]), bm.ground_patch(feet)])
    frame = bm.fit_square(np.concatenate([still.project(pts, c) for c in cams]), (1024, 1024), (0.035, 0.05, 0.03))
    box = bm.ground_box(feet, cams, frame, reach=0.5)
    reach_pts = np.concatenate([feet + [dx, dy, 0] for dx, dy in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5))])
    for c in cams:
        px = still.to_pixels(still.project(reach_pts, c), frame)
        inside = (px[:, 0] >= box[0] - 0.1) & (px[:, 0] <= box[2] + 0.1)        # the box is rounded to 0.1 px
        assert inside.all() and (px[:, 1] >= box[1] - 0.1).all()
    assert 0 <= box[0] < box[2] <= 1024 and 0 <= box[1] < box[3] <= 1024
    assert len(bm.ground_patch(np.zeros((3, 2, 3)))) == 4 * 6


# ------------------------------------------------------------------------------------ composite + checks
def test_shadow_region_is_one_in_the_box_and_fades_outside_and_at_the_border():
    r = mr.shadow_region([40, 60, 80, 90], (128, 100), fade=10)
    assert r.shape == (100, 128) and r.dtype == np.float32
    assert r[75, 60] == 1.0 and r[10, 10] == 0.0 and r[75, 20] == 0.0
    assert 0 < r[75, 35] < 1                                    # inside the fade band left of the box
    full = mr.shadow_region([0, 0, 128, 100], (128, 100), fade=10)
    assert full[0, 50] == 0 and full[50, 0] == 0 and full[50, 64] == 1.0
    assert np.all(mr.shadow_region(None, (8, 6)) == 1)


def test_composite_keeps_the_figure_prints_the_shadow_and_drops_catcher_noise():
    H, W = 60, 80
    rgba = np.zeros((H, W, 4), np.uint8)
    matte = np.zeros((H, W), np.uint8)
    rgba[10:40, 30:50] = (200, 30, 30, 255)                      # the figure
    matte[10:40, 30:50] = 255
    rgba[42:48, 20:60, 3] = 128                                  # her shadow, black at half alpha
    rgba[2:6, 2:6, 3] = 9                                        # catcher speckle up in the empty floor
    rgba[44:46, 70:74, 3] = 200                                  # a blotch outside the shadow region
    region = np.zeros((H, W), np.float32)
    region[40:55, 10:66] = 1
    out = mr.composite(rgba, matte, PAPER, region)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert tuple(out[20, 40]) == (200, 30, 30)
    assert tuple(out[3, 3]) == PAPER                             # noise below SHADOW_FLOOR never prints
    assert tuple(out[45, 72]) == PAPER                           # nor alpha outside the region
    a = (128 / 255 - mr.SHADOW_FLOOR) / (1 - mr.SHADOW_FLOOR)
    assert np.allclose(out[45, 25], np.round(np.array(PAPER) * (1 - a)), atol=1)
    assert tuple(out[0, 79]) == PAPER


def test_bbox_and_border_checks():
    m = np.zeros((50, 40), np.uint8)
    assert mr.figure_bbox(m) is None and not mr.touches_border(None, (40, 50))
    m[5:20, 10:30] = 255
    m[30, 2] = 4                                                 # anti-aliasing dust below ALPHA_MIN
    assert mr.figure_bbox(m) == [10, 5, 29, 19]
    assert not mr.touches_border([10, 5, 29, 19], (40, 50)) and mr.touches_border([0, 5, 29, 19], (40, 50))
    assert mr.touches_border([3, 5, 39, 19], (40, 50))
    assert mr.union_bbox([[1, 2, 3, 4], None, [0, 5, 2, 9]]) == [0, 2, 3, 9] and mr.union_bbox([None]) is None


# ------------------------------------------------------------------------------- fingerprint, items, rows
def test_fingerprint_follows_the_moveset_the_clip_and_the_settings(monkeypatch):
    item = {"kind": "clip", "clip": "backhand_a", "contact": 30, "n_frames": 58, "name": "backhand_a",
            "frames_dir": "/x"}
    fp = mr.fingerprint("glb1", item, PAPER)
    assert fp == mr.fingerprint("glb1", {**item, "frames_dir": "/elsewhere"}, PAPER)   # paths do not count
    assert len({fp, mr.fingerprint("glb2", item, PAPER), mr.fingerprint("glb1", {**item, "contact": 31}, PAPER),
                mr.fingerprint("glb1", {**item, "clip": "backhand_b"}, PAPER),
                mr.fingerprint("glb1", item, (250, 246, 240))}) == 5
    samples = mr.SAMPLES
    monkeypatch.setattr(mr, "SAMPLES", samples + 16)
    assert mr.fingerprint("glb1", item, PAPER) != fp
    monkeypatch.setattr(mr, "SAMPLES", samples)
    assert mr.fingerprint("glb1", item, PAPER) == fp
    monkeypatch.setattr(mr, "MOTION_VERSION", mr.MOTION_VERSION + 1)
    assert mr.fingerprint("glb1", item, PAPER) != fp


def test_file_digest_is_the_content_hash(tmp_path):
    a, b = tmp_path / "a.glb", tmp_path / "b.glb"
    a.write_bytes(b"glTF" * 1000)
    b.write_bytes(b"glTF" * 1000)
    assert mr.file_digest(a) == mr.file_digest(b)
    b.write_bytes(b"glTF" * 999 + b"glTf")
    assert mr.file_digest(a) != mr.file_digest(b)


def _run(tmp_path) -> Config:
    cfg = Config.load()
    cfg.run_dir = tmp_path
    clips = [("serve_b", "serve", False, 55, 81), ("serve_a", "serve", True, 50, 81),
             ("forehand_a", "forehand", True, 30, 58), ("slice_a", "slice", False, 32, 58)]
    cur = {"keep_all": True, "kept": [c[0] for c in clips[1:]] + ["serve_b"],
           "clips": [{"id": c, "stroke": s, "label_ok": ok, "ok": ok, "contact_index": k} for c, s, ok, k, _ in clips]}
    (tmp_path / "curation.json").write_text(json.dumps(cur))
    (tmp_path / "anim").mkdir()
    (tmp_path / "anim" / "retarget.json").write_text(json.dumps({
        "mapping": {}, "glb": {"animations": sorted(c[0] for c in clips if c[0] != "slice_a")},
        "clips": {c: {"frames": n, "fps": 25.0, "contact_index": k} for c, _, _, k, n in clips}}))
    return cfg


def test_items_follow_curation_order_keep_flagged_clips_and_end_with_the_turntable(tmp_path):
    cfg = _run(tmp_path)
    its = mr.items(cfg)
    assert [i["name"] for i in its] == ["serve_a", "forehand_a", "serve_b", "turntable"]   # slice_a: not in the GLB
    serve_b = its[2]
    assert serve_b["label_ok"] is False and serve_b["contact"] == 55 and serve_b["n_frames"] == 81
    tt = its[-1]
    assert tt["kind"] == "turntable" and tt["n_frames"] == round(MOTION.turntable_s * 25)
    assert all(i["frames_dir"].startswith(str(cfg.run.motion_frames)) for i in its)
    order = mr._render_order([*its[:1], {"name": "serve_16m13s", "kind": "clip"}, tt,
                              {"name": "backhand_14m19s", "kind": "clip"}])
    assert [o["name"] for o in order] == ["backhand_14m19s", "serve_16m13s", "serve_a", "turntable"]


def test_rows_are_run_relative_and_carry_the_camera_and_the_checks(tmp_path):
    cfg = _run(tmp_path)
    item = {**mr.items(cfg)[0], "fingerprint": "abc"}
    out = cfg.run.motion_renders
    blender = {"n_frames": 81, "contact_frame": 50, "size": [1024, 1024], "px_per_m": 410.5, "ground_y_px": 900.0,
               "ground_xy_px": [500.0, 900.0], "ground_track_px": [[500.0, 900.0]] * 81,
               "ground_box_px": [0, 800, 1024, 1024], "floor_z": -0.02, "camera": {"azimuth": 30.0, "lens_mm": 80.0},
               "racket": True, "seconds": {"setup": 2.0, "cycles": 80.0, "matte": 8.0}}
    encoded = {"video": str(out / "serve_a.mp4"), "matte": str(out / "serve_a_matte.mp4"),
               "poster": str(out / "serve_a.jpg"), "frame_bbox_px": [[100, 50, 300, 950], None, [90, 40, 310, 940]],
               "checks": {"border_frames": [], "empty_frames": [1], "shadow_flicker": {"mean": 0.1, "max": 0.3}},
               "mb": {"video": 1.0, "matte": 0.2, "poster": 0.1}, "seconds": 3.0}
    r = mr.row(item, blender, encoded, cfg.run.root)
    assert r["id"] == "serve_a" and r["stroke"] == "serve" and r["label_ok"] is True
    assert r["video"] == "artwork/motion/render/serve_a.mp4" and r["poster"] == "artwork/motion/render/serve_a.jpg"
    assert r["contact_frame"] == 50 and r["n_frames"] == 81 and r["fps"] == 25.0 and r["duration_s"] == 3.24
    assert r["bbox_px"] == [90, 40, 310, 950] and r["camera"]["azimuth"] == 30.0 and r["px_per_m"] == 410.5
    assert r["fingerprint"] == "abc" and r["seconds"]["encode"] == 3.0
    json.dumps(r)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_encode_writes_real_time_videos_whose_paper_decodes_to_the_paper(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "PRESET", "ultrafast")
    frames = tmp_path / "frames"
    frames.mkdir()
    import cv2

    n, size = 6, (64, 48)
    for i in range(n):
        rgba = np.zeros((size[1], size[0], 4), np.uint8)
        rgba[10:30, 20 + i:30 + i] = (40, 60, 200, 255)          # BGRA on disk
        matte = np.zeros_like(rgba)
        matte[10:30, 20 + i:30 + i, 3] = 255
        cv2.imwrite(str(frames / f"rgba_{i:04d}.png"), rgba)
        cv2.imwrite(str(frames / f"matte_{i:04d}.png"), matte)
    item = {"name": "clip_a", "frames_dir": str(frames)}
    enc = mr.encode(item, {"n_frames": n, "contact_frame": 2, "size": list(size), "ground_box_px": None}, PAPER,
                    tmp_path)
    assert enc["frame_bbox_px"][2] == [22, 10, 31, 29] and enc["checks"]["border_frames"] == []
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                       enc["video"]], capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert int(probe["nb_read_frames"]) == n and probe["avg_frame_rate"] == "25/1" and probe["pix_fmt"] == "yuv420p"
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", enc["video"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    first = np.frombuffer(raw, np.uint8).reshape(n, size[1], size[0], 3)[0]
    assert np.abs(first[2, 2].astype(int) - PAPER).max() <= 2     # BT.709 both ways: the paper stays the paper
    assert (tmp_path / "clip_a.jpg").exists() and (tmp_path / "clip_a_matte.mp4").exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_run_skips_fresh_videos_reencodes_kept_frames_and_force_renders_again(tmp_path, monkeypatch):
    import cv2

    cfg = _run(tmp_path)
    (tmp_path / "anim" / "rybakina_moveset.glb").write_bytes(b"glTF-1")
    monkeypatch.setattr(mr, "PRESET", "ultrafast")
    rendered: list[list[str]] = []

    def fake_blender(binary, job_path, log_path, on_done):   # 3 frames of 32x32 per video
        job = json.loads(job_path.read_text())
        rendered.append([it["name"] for it in job["items"]])
        for it in job["items"]:
            d = Path(it["frames_dir"])
            d.mkdir(parents=True, exist_ok=True)
            for i in range(3):
                img = np.zeros((32, 32, 4), np.uint8)
                img[8:24, 10 + i:20 + i] = 255
                cv2.imwrite(str(d / f"rgba_{i:04d}.png"), img)
                cv2.imwrite(str(d / f"matte_{i:04d}.png"), img)
            (d / "blender.json").write_text(json.dumps({
                "n_frames": 3, "contact_frame": 1, "size": [32, 32], "px_per_m": 10.0, "ground_y_px": 30.0,
                "ground_box_px": [0, 20, 32, 32], "camera": {}, "fingerprint": it["fingerprint"],
                "render_key": it["render_key"],
                "blender": "test", "device": {"type": "CPU"}, "seconds": {"setup": 0, "cycles": 1, "matte": 0}}))
            on_done(it["name"])

    monkeypatch.setattr(mr, "_blender", fake_blender)
    first = mr.run(cfg)
    assert rendered == [["serve_a", "forehand_a", "serve_b", "turntable"]]
    assert list(first["clips"]) == ["serve_a", "forehand_a", "serve_b"] and first["turntable"]["n_frames"] == 3
    assert (cfg.run.motion_renders / "serve_b.mp4").exists() and first["clips"]["serve_b"]["contact_frame"] == 1
    assert mr.run(cfg)["clips"] == first["clips"] and len(rendered) == 1            # all fresh: nothing to do
    (cfg.run.motion_renders / "forehand_a.mp4").unlink()
    again = mr.run(cfg)                                                             # frames kept: re-encoded only
    assert len(rendered) == 1 and (cfg.run.motion_renders / "forehand_a.mp4").exists()
    assert again["clips"]["forehand_a"]["fingerprint"] == first["clips"]["forehand_a"]["fingerprint"]
    forced = mr.run(cfg, force=True, only=["serve_b"])
    assert rendered[-1] == ["serve_b"] and list(forced["clips"]) == ["serve_a", "forehand_a", "serve_b"]
    (tmp_path / "anim" / "rybakina_moveset.glb").write_bytes(b"glTF-2")             # a new moveset
    mr.run(cfg)
    assert rendered[-1] == ["serve_a", "forehand_a", "serve_b", "turntable"]


# ------------------------------------------------------------------------------------------ render profiles
def _old_fingerprint(glb_sha: str, item: dict, paper) -> str:
    """motion_render.fingerprint as it was before profiles existed (verbatim constants of that version)."""
    spec = {"v": 1, "glb": glb_sha, "azimuth": 30.0, "elevation": 8.0, "size": [1024, 1024], "samples": 48,
            "fps": 25.0, "paper": list(paper), "encode": [18, 12, "slow"],
            **{k: item.get(k) for k in ("kind", "clip", "contact", "n_frames")}}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def test_web_fingerprint_is_the_formula_from_before_profiles():
    clip = {"kind": "clip", "clip": "backhand_14m19s", "contact": 30, "n_frames": 58, "name": "backhand_14m19s"}
    turn = {"kind": "turntable", "name": "turntable", "n_frames": 150, "start_azimuth": 0.0}
    for item in (clip, turn):
        old = _old_fingerprint("9f" * 20, item, PAPER)
        assert mr.fingerprint("9f" * 20, item, PAPER) == old            # the default is web
        assert mr.fingerprint("9f" * 20, item, PAPER, "web") == old
        assert mr.fingerprint("9f" * 20, item, PAPER, PROFILES["web"]) == old
    web = PROFILES["web"]
    assert (web.size, web.fps, web.samples, web.crf, web.matte_crf, web.preset, web.x264) == \
        ((1024, 1024), 25.0, 48, 18, 12, "slow", ())
    assert web.turntable and web.matte and web.scale == 1.0


def test_the_render_key_is_the_fingerprint_without_paper_and_encoding():
    item = {"kind": "clip", "clip": "serve_a", "contact": 50, "n_frames": 81}
    q = PROFILES["quest3"]
    key = mr.render_key("g", item, q)
    for other in (dataclasses.replace(q, crf=20), dataclasses.replace(q, preset="fast"),
                  dataclasses.replace(q, x264=()), dataclasses.replace(q, matte=True)):
        assert mr.render_key("g", item, other) == key and mr.fingerprint("g", item, PAPER, other) != \
            mr.fingerprint("g", item, PAPER, q)
    assert mr.fingerprint("g", item, (250, 246, 240), q) != mr.fingerprint("g", item, PAPER, q)
    for other in (dataclasses.replace(q, samples=48), dataclasses.replace(q, size=(2000, 2000)),
                  dataclasses.replace(q, fps=72.0)):
        assert mr.render_key("g", item, other) != key
    assert mr.render_key("g", {**item, "contact": 51}, q) != key and mr.render_key("h", item, q) != key
    assert len({mr.render_key("g", item, name) for name in PROFILES}) == 3


def test_profiles_have_their_own_fingerprints_folders_and_frames(tmp_path):
    item = {"kind": "clip", "clip": "serve_a", "contact": 50, "n_frames": 81}
    fps = {name: mr.fingerprint("g", item, PAPER, name) for name in PROFILES}
    assert len(set(fps.values())) == 3
    q = PROFILES["quest3"]
    assert mr.fingerprint("g", item, PAPER, dataclasses.replace(q, samples=q.samples + 8)) != fps["quest3"]
    assert mr.fingerprint("g", item, PAPER, dataclasses.replace(q, x264=())) != fps["quest3"]
    assert mr.fingerprint("g", {**item, "contact": 51}, PAPER, "quest3") != fps["quest3"]
    with pytest.raises(ValueError, match="unknown render profile"):
        mr.resolve("quest2")
    cfg = _run(tmp_path)
    run = cfg.run
    assert mr.frames_root(run) == run.motion_frames                                  # web: as before profiles
    assert mr.frames_root(run, "quest3") == run.motion_frames / "quest3"
    assert run.motion_renders_for("quest3") == run.motion_dir / "render_quest3"
    assert (q.size, q.fps, q.samples, q.crf, q.preset, q.turntable, q.matte) == \
        ((2064, 2208), 60.0, 16, 16, "slow", False, False)
    d = PROFILES["draft"]
    assert (d.size[0] * 4, d.size[1] * 4) == q.size and d.fps == q.fps and d.samples <= q.samples


def test_frame_times_keep_the_clip_rate_and_land_the_contact_on_an_output_frame():
    t, k = bm.frame_times(58, 30, 25.0, 25.0)                       # web: frame i is clip frame i
    assert np.array_equal(t, np.arange(58)) and k == 30
    for n, contact in ((58, 30), (58, 31), (81, 55), (75, 0), (77, 76)):
        t, k = bm.frame_times(n, contact, 25.0, 60.0)
        assert len(t) == round(n * 60 / 25)                          # real time: the duration is kept
        assert abs(len(t) / 60 - n / 25) <= 0.5 / 60
        assert math.isclose(t[k], contact, abs_tol=1e-9)             # the contact pose itself is a frame
        assert t.min() >= 0 and t.max() == n - 1                     # inside the clip, reaching its end
        steps = np.diff(t)
        assert np.all(steps >= 0) and np.allclose(steps[(t[:-1] > 0) & (t[1:] < n - 1)], 25 / 60)
    t, k = bm.frame_times(58, 30, 25.0, 60.0)
    assert k == 72 and len(t) == 139 and t[1] == pytest.approx(25 / 60)
    assert mr.out_frames(58, PROFILES["quest3"]) == 139 and mr.out_frames(58, PROFILES["web"]) == 58


def test_portrait_frames_hold_the_whole_motion_and_the_draft_is_quest3_at_a_quarter():
    frames = _moving_figure()
    pts = np.concatenate(frames)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    target, distance = (lo + hi) / 2, still.DISTANCE_PER_SIZE * float(np.max(hi - lo))
    cam = still.camera_matrix(target, -math.pi / 2, MOTION.azimuth, MOTION.elevation, distance)
    uv = still.project(pts, cam)
    fit = {name: bm.fit_square(uv, PROFILES[name].size, still.MARGINS["full"]) for name in ("quest3", "draft")}
    for name, frame in fit.items():
        W, H = PROFILES[name].size
        assert (frame["width"], frame["height"]) == (W, H) and math.isclose(frame["lens_mm"], 24 * frame["f_px"] / H)
        assert math.isclose(frame["shift_x"], frame["centre"][0] * frame["f_px"] / H)   # vertical fit: units of H
        px = still.to_pixels(uv, frame)
        span = px[:, 1].max() - px[:, 1].min()
        top, side, bottom = (m * span for m in still.MARGINS["full"])
        assert px[:, 0].min() >= side - 1e-6 and W - px[:, 0].max() >= side - 1e-6
        assert px[:, 1].min() >= top - 1e-6 and H - px[:, 1].max() >= bottom - 1e-6
        assert math.isclose(px[:, 0].min(), side, rel_tol=1e-6) or math.isclose(px[:, 1].min(), top, rel_tol=1e-6)
    assert math.isclose(fit["draft"]["f_px"] * 4, fit["quest3"]["f_px"])
    assert np.allclose(still.to_pixels(uv, fit["draft"]) * 4, still.to_pixels(uv, fit["quest3"]))
    assert fit["draft"]["lens_mm"] == pytest.approx(fit["quest3"]["lens_mm"])      # the same picture


def test_pixel_constants_scale_with_the_profile():
    q, d = PROFILES["quest3"], PROFILES["draft"]
    assert q.scale == pytest.approx(math.sqrt(2064 * 2208) / 1024) and d.scale == pytest.approx(q.scale / 4)
    assert mr.kernel(mr.NEAR_PX).shape == (5, 5) and mr.kernel(mr.FLOOR_CLEAR_PX).shape == (9, 9)   # web: as before
    assert mr.kernel(mr.NEAR_PX, q.scale).shape == (9, 9) and mr.kernel(mr.FLOOR_CLEAR_PX, q.scale).shape == (17, 17)
    assert mr.kernel(mr.NEAR_PX, d.scale).shape == (3, 3) and mr.kernel(0.1, d.scale).shape == (3, 3)
    box = [400.0, 1500.0, 1600.0, 2100.0]
    r = mr.shadow_region(box, q.size, mr.EDGE_FADE_PX * q.scale)
    fade = mr.EDGE_FADE_PX * q.scale
    assert r[1800, 1000] == 1.0 and r[1800, int(400 - fade) - 1] == 0.0 and 0 < r[1800, 380] < 1
    # the same shadow region at a quarter of the size is the quest3 one subsampled
    small = mr.shadow_region([v / 4 for v in box], d.size, mr.EDGE_FADE_PX * d.scale)
    assert np.abs(small - r[1::4, 1::4][: d.size[1], : d.size[0]]).max() < 0.12


def test_items_of_a_headset_profile_skip_the_turntable_and_use_their_own_frames(tmp_path):
    cfg = _run(tmp_path)
    its = mr.items(cfg, "quest3")
    assert [i["name"] for i in its] == ["serve_a", "forehand_a", "serve_b"]
    assert all(Path(i["frames_dir"]).parent == cfg.run.motion_frames / "quest3" for i in its)
    assert its[0]["n_frames"] == 81 and its[0]["contact"] == 50              # the clip's own frames, at 25 fps
    assert [i["name"] for i in mr.items(cfg)][-1] == "turntable"
    # a turntable is one turn in MOTION.turntable_s at the videos' rate: web's 150 frames, 360 at 60 fps
    assert mr.items(cfg)[-1]["n_frames"] == round(MOTION.turntable_s * 25) == 150
    spin = dataclasses.replace(PROFILES["quest3"], turntable=True)
    assert mr.items(cfg, spin)[-1]["n_frames"] == round(MOTION.turntable_s * 60)


def test_encoder_command_is_unchanged_for_web_and_high_at_5_2_for_quest3(tmp_path):
    out = tmp_path / "x.mp4"
    web = mr.encoder_args((1024, 1024), "rgb24", 18, mr.resolve("web"), out)
    assert web[1:] == ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                       "1024x1024", "-framerate", "25", "-i", "-", "-vf",
                       "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset",
                       "slow", "-crf", "18", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc",
                       "bt709", "-color_range", "tv", "-movflags", "+faststart", "-r", "25", str(out)]
    q = mr.encoder_args((2064, 2208), "rgb24", 16, mr.resolve("quest3"), out)
    joined = " ".join(q)
    assert "-s 2064x2208 -framerate 60" in joined and "-preset slow -crf 16 -profile:v high -level:v 5.2" in joined
    assert "-r 60" in joined and "+faststart" in joined
    assert "-x264-params colorprim=bt709:transfer=bt709:colormatrix=bt709" in joined


def test_figure_meshes_are_skinned_or_hang_off_the_armature():
    arm = SimpleNamespace(type="ARMATURE", modifiers=[], parent=None)
    skin = SimpleNamespace(type="ARMATURE")

    def mesh(parent=None, skinned=False):
        return SimpleNamespace(type="MESH", modifiers=[skin] if skinned else [], parent=parent)

    head_empty = SimpleNamespace(type="EMPTY", modifiers=[], parent=arm)
    assert still.moves_with(mesh(skinned=True), arm)                   # the body
    assert still.moves_with(mesh(parent=arm), arm)                      # a visor on the head bone
    assert still.moves_with(mesh(parent=head_empty), arm)               # ... or under an empty under the rig
    assert not still.moves_with(mesh(), arm)                            # the importer's bone-shape icosphere
    assert not still.moves_with(mesh(parent=SimpleNamespace(type="EMPTY", modifiers=[], parent=None)), arm)
    assert not still.moves_with(head_empty, arm)


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
def test_quest3_encode_is_60_fps_high_profile_without_a_matte_video(tmp_path):
    import cv2

    frames = tmp_path / "frames"
    frames.mkdir()
    n, size = 12, (96, 104)
    for i in range(n):
        rgba = np.zeros((size[1], size[0], 4), np.uint8)
        rgba[20:60, 30 + i:44 + i] = (40, 60, 200, 255)
        matte = np.zeros_like(rgba)
        matte[20:60, 30 + i:44 + i, 3] = 255
        cv2.imwrite(str(frames / f"rgba_{i:04d}.png"), rgba)
        cv2.imwrite(str(frames / f"matte_{i:04d}.png"), matte)
    # the real settings (tiny frames, so preset slow is quick): with ultrafast x264 would fall back to Constrained
    # Baseline, its profile option being a ceiling
    enc = mr.encode({"name": "clip_a", "frames_dir": str(frames)},
                    {"n_frames": n, "contact_frame": 7, "size": list(size), "ground_box_px": [0, 50, 96, 104]},
                    PAPER, tmp_path, "quest3")
    assert "matte" not in enc and "matte" not in enc["mb"] and not (tmp_path / "clip_a_matte.mp4").exists()
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json",
                                       enc["video"]], capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert int(probe["nb_read_frames"]) == n and probe["avg_frame_rate"] == "60/1"
    assert probe["profile"] == "High" and probe["level"] == 52 and probe["pix_fmt"] == "yuv420p"
    assert probe["color_space"] == "bt709" and probe["color_range"] == "tv"
    assert probe["color_primaries"] == "bt709" and probe["color_transfer"] == "bt709"     # the full BT.709 VUI
    poster = cv2.imread(str(tmp_path / "clip_a.jpg"))
    assert poster.shape[:2] == (size[1], size[0]) and poster[40, 30 + 7 + 5, 0] < 120   # the contact frame's figure


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_profiles_never_touch_each_others_videos_frames_or_reports(tmp_path, monkeypatch):
    import cv2

    cfg = _run(tmp_path)
    (tmp_path / "anim" / "rybakina_moveset.glb").write_bytes(b"glTF-1")
    monkeypatch.setattr(mr, "PRESET", "ultrafast")
    # frames kept here so the re-encode path can be checked (the real quest3 drops them, see the next test)
    monkeypatch.setitem(mr.PROFILES, "quest3", dataclasses.replace(PROFILES["quest3"], preset="ultrafast",
                                                                   keep_frames=True))
    jobs: list[dict] = []

    def fake_blender(binary, job_path, log_path, on_done):   # out_fps / 25 x 2 frames of the job's size
        job = json.loads(job_path.read_text())
        jobs.append(job)
        n = int(round(2 * job["out_fps"] / job["fps"]))
        W, H = (v // 48 * 2 for v in job["size"])   # even sizes, as H.264 needs
        for it in job["items"]:
            d = Path(it["frames_dir"])
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = np.zeros((H, W, 4), np.uint8)
                img[H // 4: H // 2, W // 4 + i: W // 2 + i] = 255
                cv2.imwrite(str(d / f"rgba_{i:04d}.png"), img)
                cv2.imwrite(str(d / f"matte_{i:04d}.png"), img)
            (d / "blender.json").write_text(json.dumps({
                "n_frames": n, "contact_frame": n // 2, "size": [W, H], "fps": job["out_fps"],
                "source": {"fps": 25.0, "n_frames": 2, "contact_frame": 1}, "camera": {},
                "fingerprint": it["fingerprint"], "render_key": it["render_key"], "blender": "test",
                "device": {"type": "CPU"}, "seconds": {"setup": 0, "cycles": 1, "matte": 0}}))
            on_done(it["name"])

    monkeypatch.setattr(mr, "_blender", fake_blender)
    web = mr.run(cfg)
    web_json = (cfg.run.motion_renders / "renders.json").read_text()
    web_frames = sorted(p.name for p in cfg.run.motion_frames.glob("serve_a/*"))
    q = mr.run(cfg, profile="quest3")
    assert jobs[-1]["out_fps"] == 60.0 and jobs[-1]["size"] == [2064, 2208] and jobs[-1]["fps"] == 25.0
    assert jobs[-1]["settings"]["samples"] == 16 and [it["name"] for it in jobs[-1]["items"]] == \
        ["serve_a", "forehand_a", "serve_b"]                                  # no turntable
    assert q["profile"] == "quest3" and web["profile"] == "web" and q["turntable"] is None and q["fps"] == 60.0
    row = q["clips"]["serve_a"]
    assert row["video"] == "artwork/motion/render_quest3/serve_a.mp4" and "matte" not in row
    assert row["fps"] == 60.0 and row["n_frames"] == 5 and row["contact_frame"] == 2 and row["duration_s"] == 0.083
    assert row["fingerprint"] != web["clips"]["serve_a"]["fingerprint"]
    assert (cfg.run.motion_frames / "quest3" / "job.json").exists()
    # web's report, videos and frames are untouched by quest3 (and quest3 re-runs nothing while fresh)
    assert (cfg.run.motion_renders / "renders.json").read_text() == web_json
    assert sorted(p.name for p in cfg.run.motion_frames.glob("serve_a/*")) == web_frames
    assert mr.run(cfg, profile="quest3")["clips"] == q["clips"] and len(jobs) == 2
    mr.run(cfg, profile="quest3", force=True, only=["serve_a"])
    assert [it["name"] for it in jobs[-1]["items"]] == ["serve_a"] and len(jobs) == 3
    assert sorted(p.name for p in cfg.run.motion_frames.glob("serve_a/*")) == web_frames
    assert mr.run(cfg)["clips"] == web["clips"] and len(jobs) == 3          # web still fresh after all that
    # a new encoding re-encodes quest3's kept frames under the same render key; Blender is not called again
    monkeypatch.setitem(mr.PROFILES, "quest3", dataclasses.replace(PROFILES["quest3"], preset="ultrafast", crf=20,
                                                                   keep_frames=True))
    recoded = mr.run(cfg, profile="quest3")
    assert len(jobs) == 3 and recoded["clips"]["serve_a"]["fingerprint"] != q["clips"]["serve_a"]["fingerprint"]
    assert recoded["encoding"]["crf"] == 20


def test_keep_frames_is_housekeeping_and_quest3_drops_its_pngs(tmp_path, monkeypatch):
    import cv2

    q = PROFILES["quest3"]
    item = {"kind": "clip", "clip": "serve_a", "contact": 1, "n_frames": 2}
    assert not q.keep_frames
    assert mr.fingerprint("sha", item, (250, 245, 240), q) == \
        mr.fingerprint("sha", item, (250, 245, 240), dataclasses.replace(q, keep_frames=True))
    cfg = _run(tmp_path)
    (tmp_path / "anim" / "rybakina_moveset.glb").write_bytes(b"glTF-1")
    monkeypatch.setitem(mr.PROFILES, "quest3", dataclasses.replace(q, preset="ultrafast"))

    def fake_blender(binary, job_path, log_path, on_done):
        job = json.loads(job_path.read_text())
        for it in job["items"]:
            d = Path(it["frames_dir"])
            d.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                img = np.zeros((92, 86, 4), np.uint8)
                img[20:40, 20 + i:40 + i] = 255
                cv2.imwrite(str(d / f"rgba_{i:04d}.png"), img)
                cv2.imwrite(str(d / f"matte_{i:04d}.png"), img)
            (d / "blender.json").write_text(json.dumps({
                "n_frames": 5, "contact_frame": 2, "size": [86, 92], "fps": 60.0,
                "source": {"fps": 25.0, "n_frames": 2, "contact_frame": 1}, "camera": {},
                "fingerprint": it["fingerprint"], "render_key": it["render_key"], "blender": "test",
                "device": {"type": "CPU"}, "seconds": {"setup": 0, "cycles": 1, "matte": 0}}))
            on_done(it["name"])

    monkeypatch.setattr(mr, "_blender", fake_blender)
    report = mr.run(cfg, profile="quest3")
    frames = cfg.run.motion_frames / "quest3" / "serve_a"
    assert report["clips"]["serve_a"]["n_frames"] == 5
    assert not list(frames.glob("*.png")) and (frames / "blender.json").exists()
    assert (cfg.run.root / report["clips"]["serve_a"]["video"]).exists()
