"""Unit tests for the painted-motion stage (rybakina3d.artwork.motion_paint): masks, IoU, padding, trimming, frame
mapping, the anchor's placement and the acceptance rule, on synthetic frames (no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from rybakina3d.artwork import motion_paint as mp
from rybakina3d.artwork.paint import clean_mask
from rybakina3d.config import Config

PAPER = (251, 246, 240)


def _figure(h: int = 240, w: int = 240, cx: int = 120, top: int = 40, bottom: int = 200, arm: int = 0) -> np.ndarray:
    """A stick figure (head, torso, legs, one arm out, raised by `arm` px): thin parts keep the IoU honest."""
    m = np.zeros((h, w), bool)
    m[top:top + 24, cx - 10:cx + 10] = True
    m[top + 24:top + 90, cx - 18:cx + 18] = True
    m[top + 90:bottom, cx - 16:cx - 4] = True
    m[top + 90:bottom, cx + 4:cx + 16] = True
    m[top + 34 - arm:top + 42 - arm, cx + 18:min(w, cx + 60)] = True
    return m


def _frame(mask: np.ndarray, shadow: bool = True) -> np.ndarray:
    """A render frame: off-white figure (a white dress) on the paper, a soft grey shadow under the feet."""
    h, w = mask.shape
    f = np.empty((h, w, 3), np.float32)
    f[:] = PAPER
    if shadow:
        ys, xs = np.nonzero(mask)
        yy, xx = np.mgrid[0:h, 0:w]
        ell = np.exp(-(((xx - xs.mean()) / 30.0) ** 2 + ((yy - ys.max()) / 5.0) ** 2))
        f *= (1 - 0.35 * ell)[..., None]
    f[mask] = (240, 238, 235)
    return f.round().astype(np.uint8)


# --- masks + IoU -----------------------------------------------------------------------------------------
def test_matte_mask_reads_either_polarity():
    m = _figure()
    light_on_dark = np.where(m[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
    dark_on_light = np.where(m[..., None], 26, 255).astype(np.uint8).repeat(3, axis=2)
    on_paper = np.where(m[..., None], np.uint8(30), np.array(PAPER, np.uint8)).astype(np.uint8)
    for frame in (light_on_dark, dark_on_light, on_paper):
        assert mp.iou(mp.matte_mask(frame), clean_mask(m)) > 0.97


def test_alpha_mask_thresholds_and_cleans():
    m = _figure()
    g = np.where(m, 230, 10).astype(np.uint8)
    g[5:7, 5:7] = 255                                  # a speck far from her
    assert mp.iou(mp.alpha_mask(np.dstack([g] * 3)), clean_mask(m)) > 0.97


def test_iou_stats_separate_shape_from_place():
    refs = [_figure(arm=a) for a in range(0, 20, 2)]
    shifted = [np.roll(r, 15, axis=1) for r in refs]
    st = mp.iou_stats(shifted, refs)
    assert st["median"] == pytest.approx(1.0) and st["min"] == pytest.approx(1.0)
    assert st["raw_median"] < 0.6                       # moved: the raw IoU sees it, the aligned one does not
    wrong = [r.copy() for r in refs]
    wrong[3] = _figure(arm=30)                          # one frame in the wrong pose
    st = mp.iou_stats(wrong, refs)
    assert st["worst_frame"] == 3 and st["min"] < 0.95 and st["median"] == pytest.approx(1.0)
    assert len(st["per_frame"]) == len(refs)


# --- sending, padding, trimming --------------------------------------------------------------------------
def test_plan_holds_frame_zero_up_to_the_next_4k_plus_1():
    assert mp.plan(58) == (61, 3)                     # a groundstroke: 58 frames sent as 61
    assert mp.plan(81) == (81, 0)                     # a serve is 4k+1 already
    assert mp.plan(1) == (5, 4) and mp.plan(120) == (121, 1) and mp.plan(121) == (121, 0)
    with pytest.raises(ValueError):
        mp.plan(122)                                  # longer than Bernini-R takes


def test_fit_resizes_the_model_output_back():
    frame = _frame(_figure())
    out = np.asarray(Image.fromarray(frame).resize((1080, 1080), Image.LANCZOS))
    back = mp.fit(out, (240, 240))
    assert back.shape == frame.shape and np.abs(back.astype(int) - frame.astype(int)).mean() < 3
    assert mp.fit(frame, (240, 240)) is frame


def test_frame_map_trims_the_held_head_and_follows_the_output_rate():
    # same rate: render frame i is sent frame head + i
    assert mp.frame_map(58, 17, 75, 75, 25.0) == list(range(17, 75))
    # a 24 fps output of the same 3 s: nearest earlier frame at the same moment, monotone, never past the end
    m = mp.frame_map(58, 17, 75, 72, 24.0)
    assert m[0] == 16 and m[-1] == 71 and all(b >= a for a, b in zip(m, m[1:], strict=False))
    # a model that fitted 3 s into 5 s (slow motion): stretched mapping lands at the same fraction of the clip
    m = mp.frame_map(58, 17, 75, 125, 25.0, stretch=True)
    assert m[0] == int((17.5 / 75) * 125) and m[-1] == 124
    assert not mp.timing(75, 72, 24.0) and mp.timing(75, 125, 25.0)
    # a model that dropped a frame keeps real time; the last render frame maps to the last output frame
    assert mp.frame_map(58, 0, 58, 57, 25.0)[-1] == 56


def test_video_round_trip_keeps_frames_size_and_rate(tmp_path):
    frames = [np.full((64, 96, 3), v, np.uint8) for v in range(0, 250, 10)]
    dest = mp.write_video(frames, tmp_path / "v.mp4")
    n, fps, size = mp.count_frames(dest)
    assert (n, size) == (len(frames), (96, 64)) and fps == pytest.approx(25.0)
    back = mp.read_video(dest, keep=lambda i: i % 5 == 0)
    assert len(back) == 5 and abs(int(back[1].mean()) - 50) <= 2


# --- the anchor ------------------------------------------------------------------------------------------
def test_plate_takes_the_figure_out_but_keeps_the_shadow():
    fig = _figure()
    frame = _frame(fig)
    p = mp.plate(frame, fig)
    inner = np.zeros_like(fig)
    inner[60:150, 108:132] = True                       # the torso, well inside her
    assert np.abs(p[inner].astype(int) - np.array(PAPER)).max() <= 2
    below = (198, 120)                                  # between her feet, on the shadow
    assert p[below].mean() < np.mean(PAPER) - 20


def test_place_sets_the_repaint_on_the_render_feet_line():
    fig = _figure(top=60, bottom=200)
    frame = _frame(fig)
    # the "repaint": the same figure smaller and elsewhere, as a transparent cut-out at another size
    crop = fig[60:200, 80:200].astype(np.uint8)
    small = np.zeros((300, 300), bool)
    small[30:142, 40:136] = np.asarray(Image.fromarray(crop * 255).resize((96, 112), Image.NEAREST)) > 127
    rgba = np.zeros((300, 300, 4), np.uint8)
    rgba[small] = (200, 120, 90, 255)
    out, placed = mp.place(Image.fromarray(rgba, "RGBA"), small, frame, fig)
    assert out.shape == frame.shape
    assert mp.iou(placed, fig) > 0.85                   # scaled + shifted onto her
    ys, xs = np.nonzero(placed)
    assert abs(ys.max() - np.nonzero(fig)[0].max()) <= 2
    assert np.abs(out[100, 120].astype(int) - (200, 120, 90)).max() < 12
    assert (out[5, 5] == frame[5, 5]).all()             # the paper away from her is the render's


# --- measures + acceptance -------------------------------------------------------------------------------
def test_flicker_and_drift_see_noise_and_colour_change():
    fig = _figure()
    still = [_frame(fig)] * 6
    masks = [fig] * 6
    calm = mp.flicker(still, masks)
    assert calm["figure"] == 0 and calm["paper"] == 0
    rng = np.random.default_rng(0)
    noisy = [np.clip(f.astype(int) + rng.integers(-20, 20, f.shape) * fig[..., None], 0, 255).astype(np.uint8)
             for f in still]
    fl = mp.flicker(noisy, masks)
    assert fl["figure"] > 10 and fl["paper"] == 0
    assert mp.colour_drift(still, masks)["body_max"] == 0
    tinted = [f.copy() for f in still]
    tinted[-1][fig] = (240, 200, 150)
    d = mp.colour_drift(tinted, masks)
    assert d["body_max"] > 10 and d["body_median"] == 0


def test_judge_needs_first_frame_and_iou_throughout():
    ok = {"first_match": {"iou": 0.9}, "iou": {"median": 0.9, "p10": 0.8}}
    assert mp.judge(ok) == (True, "ok")
    assert not mp.judge({**ok, "first_match": {"iou": 0.6}})[0]
    assert not mp.judge({**ok, "iou": {"median": 0.9, "p10": 0.6}})[0]
    accepted, why = mp.judge({**ok, "iou": {"median": 0.8, "p10": 0.7}})
    assert not accepted and "median" in why and "p10" in why


def test_rejudge_promotes_or_withdraws_the_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "BACKGROUND", "model")      # the model's frames as they are: no render needed here
    (tmp_path / "attempts").mkdir()
    (tmp_path / "attempts" / "c.m.mp4").write_bytes(b"video")
    src = mp.Source("c", tmp_path / "r.mp4", tmp_path / "m.mp4", "forehand", 3, (8, 8))
    e = {"clip": "c", "model": "m", "accepted": False, "attempt": "attempts/c.m.mp4", "first_match": {"iou": 0.9},
         "iou": {"median": 0.84, "p10": 0.8}}
    assert not mp._rejudge(e, tmp_path, {"c": src}.__getitem__)["accepted"]
    monkeypatch.setattr(mp, "ACCEPT_MEDIAN", 0.8)
    e2 = mp._rejudge(e, tmp_path, {"c": src}.__getitem__)
    assert e2["accepted"] and (tmp_path / "c.mp4").read_bytes() == b"video"
    assert json.loads((tmp_path / "c.json").read_text())["accepted"]
    monkeypatch.setattr(mp, "ACCEPT_MEDIAN", 0.9)
    e3 = mp._rejudge(e2, tmp_path, {"c": src}.__getitem__)
    assert not e3["accepted"] and not (tmp_path / "c.mp4").exists() and "video" not in e3


def test_judge_gates_a_vanished_racket():
    ok = {"first_match": {"iou": 0.9}, "iou": {"median": 0.9, "p10": 0.85},
          "racket": {"median": 0.95, "p10": 0.8, "frames": 50}}
    assert mp.judge(ok)[0]
    accepted, why = mp.judge({**ok, "racket": {"median": 0.9, "p10": 0.2, "frames": 50}})
    assert not accepted and "racket" in why
    assert mp.judge({**ok, "racket": {"median": None, "p10": None, "frames": 0}})[0]   # never in view: not gated


# --- request + cache key ---------------------------------------------------------------------------------
def test_request_names_every_input_in_order():
    args = mp.video_args("VIDEO", ["ANCHOR", "STYLE", "HERO", "SHEET"], "backhand", 61)
    assert args["video_url"] == "VIDEO" and args["reference_image_urls"] == ["ANCHOR", "STYLE", "HERO", "SHEET"]
    assert args["num_frames"] == 61 and args["frames_per_second"] == 25 and not args["enable_prompt_expansion"]
    for ref in ("source video", "first reference image", "second reference image", "third reference image",
                "fourth reference image", "backhand"):
        assert ref in args["prompt"]
    assert "fourth" not in mp.video_args("VIDEO", ["ANCHOR", "STYLE", "HERO"], "serve", 81)["prompt"]
    assert mp.grip_for("backhand", "two_handed") == "two_handed" and mp.grip_for("serve", "two_handed") == "racket"


def test_fingerprint_follows_render_and_stroke(tmp_path):
    cfg = Config.load()
    for name, val in (("r.mp4", b"a"), ("m.mp4", b"b"), ("r2.mp4", b"c")):
        (tmp_path / name).write_bytes(val)
    src = mp.Source("c", tmp_path / "r.mp4", tmp_path / "m.mp4", "forehand", 58, (1024, 1024))
    key = mp.fingerprint(cfg, src)
    assert key == mp.fingerprint(cfg, src) and mp.VIDEO_MODEL in key
    assert key != mp.fingerprint(cfg, mp.Source("c", tmp_path / "r2.mp4", tmp_path / "m.mp4", "forehand", 58,
                                                (1024, 1024)))
    assert key != mp.fingerprint(cfg, mp.Source("c", tmp_path / "r.mp4", tmp_path / "m.mp4", "backhand", 58,
                                                (1024, 1024)))


def test_paint_clip_and_run_drive_models_video_paint_only():
    cfg = Config.load()
    assert cfg.models.video_paint == mp.VIDEO_MODEL
    with pytest.raises(ValueError, match="only"):
        mp.paint_clip(cfg, "c", model="kling/v4/video-to-video/edit")
    cfg.models.video_paint = "fal-ai/some/other-model"
    for call in (lambda: mp.paint_clip(cfg, "c"), lambda: mp.run(cfg)):
        with pytest.raises(ValueError, match="models.video_paint"):
            call()


def test_load_source_reads_the_contract_files(tmp_path):
    cfg = Config.load()
    cfg.run_dir = tmp_path
    cfg.__dict__.pop("run", None)
    root = cfg.run.motion_renders
    root.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        mp.load_source(cfg, "forehand_00m07s")
    frames = [_frame(_figure(arm=a)) for a in range(4)]
    mp.write_video(frames, root / "forehand_00m07s.mp4")
    mp.write_video([np.where(_figure(arm=a)[..., None], 255, 0).astype(np.uint8).repeat(3, 2) for a in range(4)],
                   root / "forehand_00m07s_matte.mp4")
    src = mp.load_source(cfg, "forehand_00m07s")
    assert (src.stroke, src.frames, src.size) == ("forehand", 4, (240, 240))


def test_composite_puts_the_painted_figure_on_the_render_paper():
    fig = _figure()
    frame = _frame(fig)
    painted = np.empty_like(frame)
    painted[:] = (230, 220, 150)                      # the model's yellowish paper ...
    painted[fig] = (200, 120, 90)                     # ... and its painted figure
    alpha = np.where(fig, 255, 0).astype(np.uint8)
    alpha[5:9, 5:9] = 255                             # a matting speck far from her
    out = mp.composite(painted, alpha, fig, frame, fig)
    assert (out[fig] == (200, 120, 90)).all()
    assert (out[3:12, 3:12] == frame[3:12, 3:12]).all()   # paper (and no speck) from the render
    assert out[198, 120].mean() < np.mean(PAPER) - 20     # the render's shadow under her


def test_racket_recall_sees_a_vanished_racket():
    fig = _figure()
    frame = _frame(fig)
    frame[60:90, 170:200] = (220, 30, 30)             # the red racket head
    kept = frame.copy()
    kept[60:90, 170:200] = (190, 50, 40)              # painted a little darker: still the racket
    gone = frame.copy()
    gone[60:90, 170:200] = PAPER                      # dissolved into the swing
    assert mp.racket_recall([kept] * 3, [frame] * 3)["median"] > 0.95
    assert mp.racket_recall([gone] * 3, [frame] * 3)["median"] == 0
    assert mp.racket_recall([frame], [_frame(fig)])["frames"] == 0   # no racket in the render: not counted


def test_look_carried_tells_anchor_from_render():
    fig = _figure()
    render = _frame(fig)
    anchor = render.copy()
    anchor[fig] = (200, 150, 120)
    same_as_anchor = mp.look_carried([anchor] * 4, [render] * 4, [fig] * 4, anchor, fig)
    assert same_as_anchor["carry"] == pytest.approx(1.0) and same_as_anchor["restyle"] == pytest.approx(1.0)
    back_to_render = mp.look_carried([render] * 4, [render] * 4, [fig] * 4, anchor, fig)
    assert back_to_render["carry"] == pytest.approx(0.0) and back_to_render["restyle"] == pytest.approx(0.0)


def test_load_source_reads_renders_json_rows(tmp_path):
    cfg = Config.load()
    cfg.run_dir = tmp_path
    cfg.__dict__.pop("run", None)
    root = cfg.run.motion_renders
    mp.write_video([_frame(_figure())] * 3, root / "a.mp4")
    mp.write_video([np.where(_figure()[..., None], 255, 0).astype(np.uint8).repeat(3, 2)] * 3, root / "a_m.mp4")
    (root / "renders.json").write_text(json.dumps({"clips": {"serve_x": {
        "video": "artwork/motion/render/a.mp4", "matte": "artwork/motion/render/a_m.mp4", "stroke": "serve",
        "n_frames": 3, "contact_frame": 2}}}))
    src = mp.load_source(cfg, "serve_x")
    assert (src.video.name, src.matte.name, src.stroke, src.frames, src.contact) == ("a.mp4", "a_m.mp4", "serve", 3, 2)
