"""Unit tests for the motion QA metrics and gate, on synthetic motion with known answers."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rybakina3d import qa
from rybakina3d.config import Config

FPS = 25.0


def _K(f: float = 1000.0) -> np.ndarray:
    return np.array([[f, 0, 960], [0, f, 540], [0, 0, 1.0]])


# --- skate ---------------------------------------------------------------------------------------------
def test_planted_foot_has_zero_skate():
    pos = np.tile(np.array([[0.3, 1.2], [0.5, 1.0]]), (20, 1, 1))     # (L, K, 2), not moving
    out = qa.skate(pos, np.ones((20, 2), bool))
    assert out["mean_cm"] == 0 and out["max_cm"] == 0
    assert out["n_steps"] == 19 * 2


def test_skate_speed_and_direction_split():
    L = 11
    pos = np.zeros((L, 1, 2))
    pos[:, 0, 0] = np.arange(L) * 0.01                                 # 1 cm/frame along x
    contact = np.ones((L, 1), bool)
    along = qa.skate(pos, contact, np.tile([1.0, 0.0], (L, 1, 1)))      # ray along x
    assert along["mean_cm"] == pytest.approx(1.0)
    assert along["depth_mean_cm"] == pytest.approx(1.0)
    assert along["lateral"]["mean_cm"] == pytest.approx(0.0)
    assert along["depth_share"] == pytest.approx(1.0)
    across = qa.skate(pos, contact, np.tile([0.0, 1.0], (L, 1, 1)))     # ray along z
    assert across["lateral"]["mean_cm"] == pytest.approx(1.0)
    assert across["depth_share"] == pytest.approx(0.0)


def test_skate_only_counts_steps_planted_at_both_ends():
    pos = np.cumsum(np.full((10, 1, 2), 0.05), axis=0)
    contact = (np.arange(10) % 2 == 0)[:, None]                         # never planted twice in a row
    out = qa.skate(pos, contact)
    assert out["n_steps"] == 0 and out["mean_cm"] is None


# --- ground --------------------------------------------------------------------------------------------
def test_ground_planted_at_rest_height_is_flat():
    offsets = np.array([0.07, 0.015])
    out = qa.ground(np.tile(offsets, (30, 1)), np.ones((30, 2), bool), offsets)
    assert out["offset_cm"] == 0 and out["penetration_max_cm"] == 0 and out["float_mean_cm"] == 0


def test_ground_sunk_clip_penetrates_and_regrounds():
    offsets = np.array([0.07, 0.015])
    out = qa.ground(np.tile(offsets - 0.05, (30, 1)), np.ones((30, 2), bool), offsets)
    assert out["penetration_max_cm"] == pytest.approx(5.0)
    assert out["penetration_frac"] == 1.0
    assert out["offset_cm"] == pytest.approx(-5.0)
    assert out["penetration_regrounded_cm"] == pytest.approx(0.0)


# --- jitter --------------------------------------------------------------------------------------------
def test_jitter_of_constant_jerk_motion():
    jerk = 40.0                                                        # m/s^3
    t = np.arange(30) / FPS
    joints = np.zeros((30, 2, 3))
    joints[:, :, 1] = (jerk * t**3 / 6)[:, None]                        # third derivative = jerk exactly
    assert qa.jitter(joints, FPS) == pytest.approx(jerk)
    assert qa.jitter(np.ones((30, 5, 3)), FPS) == 0.0
    assert qa.jitter(joints[:3], FPS) is None


def test_depth_share_of_noise_along_the_ray():
    track = np.zeros((40, 3))
    track[:, 2] = np.random.default_rng(0).normal(size=40)
    rays = np.tile([0.0, 0.0, 1.0], (40, 1))
    assert qa.depth_share(track, rays) == pytest.approx(1.0)
    track[:, 0] = track[:, 2]
    assert qa.depth_share(track, rays) == pytest.approx(0.5)
    assert qa.depth_share(track, np.tile([2**-0.5, 0.0, 2**-0.5], (40, 1))) == pytest.approx(1.0)


# --- reprojection --------------------------------------------------------------------------------------
def _scene(L: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    joints_cam = rng.uniform([-1, -1, 8], [1, 1, 12], size=(L, 24, 3))
    K = np.tile(_K(), (L, 1, 1))
    kp = np.zeros((L, 17, 3))
    uv = qa.project(joints_cam, K)
    for c, s in qa.COCO_TO_SMPL.items():
        kp[:, c, :2] = uv[:, s]
    kp[:, :, 2] = 0.9
    return joints_cam, K, kp


def test_perfect_reprojection_has_zero_error():
    joints_cam, K, kp = _scene()
    kp[:, 0] = [5000, 5000, 0.2]                                       # unconfident nose far away: ignored
    out = qa.reprojection(joints_cam, K, kp, np.full(len(kp), 300.0))
    assert out["median_px"] == 0 and out["pck"] == 1.0
    assert out["n"] == len(kp) * 12
    assert out["parts"]["arms"]["n"] == out["parts"]["legs"]["n"] == len(kp) * 6


def test_reprojection_known_offset():
    joints_cam, K, kp = _scene()
    kp[..., 0] += 5.0
    out = qa.reprojection(joints_cam, K, kp, np.full(len(kp), 100.0))
    assert out["median_px"] == pytest.approx(5.0)
    assert out["median_norm"] == pytest.approx(0.05)
    assert out["pck"] == 1.0
    kp[..., 0] += 10.0                                                  # 15 px > 0.1 x 100 px
    assert qa.reprojection(joints_cam, K, kp, np.full(len(kp), 100.0))["pck"] == 0.0


def test_project_matches_pinhole():
    uv = qa.project(np.array([[[1.0, -0.5, 10.0]]]), _K()[None])
    assert np.allclose(uv[0, 0], [960 + 100, 540 - 50])
    assert np.isnan(qa.project(np.array([[[0.0, 0.0, -1.0]]]), _K()[None])).all()


# --- camera direction, timing, rig names ---------------------------------------------------------------
def test_camera_rotations_and_rays():
    rng = np.random.default_rng(1)
    world = rng.normal(size=(5, 22, 3))
    R = Rotation.from_euler("yx", [35, 12], degrees=True).as_matrix()  # heading + downward tilt
    cam = world @ R.T + [0.3, 0.2, 15.0]
    assert np.allclose(qa.camera_rotations(world, cam), R, atol=1e-9)
    on_axis = qa.ray_dirs(np.tile(R, (5, 1, 1)), np.tile([0.0, 0.0, 10.0], (5, 1, 1)))
    assert np.allclose(on_axis, R[2], atol=1e-12)                      # optical axis = R^T e_z
    p = np.array([3.0, -1.0, 12.0])
    assert np.allclose(qa.ray_dirs(R[None], p[None, None])[0, 0], R.T @ p / np.linalg.norm(p))


def test_skate_split_follows_the_ray_through_the_foot():
    # a static camera looking along +z sees a foot 30 degrees off-axis slide along its own ray (depth noise)
    L = 12
    R = np.tile(np.eye(3), (L, 1, 1))
    ray = np.array([np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)])
    foot_cam = (10 + 0.05 * np.arange(L))[:, None, None] * ray          # 5 cm/frame along the ray
    out = qa.skate(foot_cam[..., [0, 2]], np.ones((L, 1), bool), qa.horizontal(qa.ray_dirs(R, foot_cam)))
    assert out["mean_cm"] == pytest.approx(5.0)
    assert out["lateral"]["mean_cm"] == pytest.approx(0.0, abs=1e-9)  # the optical axis would say 2.5
    assert out["depth_share"] == pytest.approx(1.0)


def test_speed_peaks_at_the_swing():
    track = np.zeros((40, 3))
    track[:, 0] = 0.5 * np.tanh((np.arange(40) - 17) / 2.0)            # fastest at frame 17
    v = qa.speed(track, FPS)
    assert int(np.argmax(v)) == 17 and v.max() == pytest.approx(0.5 / 2.0 * FPS, rel=0.1)


def test_feet_indices_naming_schemes():
    meshy = ["Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot",
             "RightToeBase"]
    assert qa.feet_indices(meshy) == [3, 4, 7, 8]
    assert qa.feet_indices([f"mixamorig:{n}" for n in meshy]) == [3, 4, 7, 8]
    smpl = ["pelvis", "left_ankle", "right_ankle", "left_foot", "right_foot"]
    assert qa.feet_indices(smpl) == [1, 3, 2, 4]
    assert qa.feet_indices(["Hips", "Spine"]) is None


# --- gate ----------------------------------------------------------------------------------------------
def _good_metrics() -> dict:
    return {
        "skate": {"mean_cm": 0.5, "p95_cm": 1.0, "n_steps": 40, "depth_share": 0.5,
                  "lateral": {"mean_cm": 0.3, "p95_cm": 0.8, "max_cm": 1.0}},
        "ground": {"offset_cm": 1.0, "penetration_regrounded_cm": 1.0, "penetration_max_cm": 0.0,
                   "float_mean_cm": 1.0},
        "jitter": {"pose_ms3": 200.0, "root_ms3": 100.0, "root_depth_share": 0.5},
        "reprojection": {"n": 100, "median_norm": 0.03, "pck": 0.98},
        "timing": {"offset_frames": 1, "apex_offset_frames": None},
        "tracking": {"coverage": 1.0},
        "camera": {"static": True, "motion_px": 0.2, "calibrated": False},
        "retarget": None,
    }


def test_gate_pass_warn_fail():
    q = _good_metrics()
    status, reasons, notes = qa.gate(q)
    assert status == "PASS" and reasons == [] and "not retargeted yet" in notes

    q["skate"].update(mean_cm=7.0, depth_share=0.99)                    # depth noise only: repairable
    q["jitter"]["root_ms3"] = 3000.0
    status, reasons, _ = qa.gate(q)
    assert status == "WARN" and len(reasons) == 2

    q["timing"]["offset_frames"] = 20                                   # swing nowhere near the contact
    status, reasons, _ = qa.gate(q)
    assert status == "FAIL" and any("contact" in r for r in reasons)


def test_gate_fails_on_lateral_skate_pose_jitter_and_retarget():
    q = _good_metrics()
    q["skate"]["lateral"]["mean_cm"] = 4.0
    assert qa.gate(q)[0] == "FAIL"
    q = _good_metrics()
    q["jitter"]["pose_ms3"] = 900.0
    assert qa.gate(q)[0] == "FAIL"
    q = _good_metrics()
    q["retarget"] = {"mean_cm": 3.5, "p95_cm": 5.0}
    assert qa.gate(q)[0] == "FAIL"


def test_gate_warns_on_a_locked_but_jittery_retarget():
    q = _good_metrics()
    q["retarget"] = {"mean_cm": 0.1, "p95_cm": 0.3, "root_ms3": 1200.0, "pose_ms3": 150.0}
    status, reasons, _ = qa.gate(q)
    assert status == "WARN" and any("retargeted root jitter" in r for r in reasons)


def test_gate_needs_a_swing_at_contact():
    q = _good_metrics()
    q["timing"]["swing_speed_ms"] = 0.4                                 # she waits while the opponent hits
    status, reasons, _ = qa.gate(q)
    assert status == "FAIL" and any("no swing" in r for r in reasons)
    q["timing"]["swing_speed_ms"] = 2.5
    assert qa.gate(q)[0] == "WARN"
    q["timing"]["swing_speed_ms"] = 6.0
    assert qa.gate(q)[0] == "PASS"


# --- end to end ----------------------------------------------------------------------------------------
def _motion(L: int = 50, k: int = 30) -> dict:
    """A clean synthetic clip: planted feet, a right-wrist swing peaking at k, camera 10 m away."""
    rng = np.random.default_rng(2)
    J = np.tile(rng.uniform(-0.3, 0.3, size=(1, 24, 3)) + [0, 1.0, 0], (L, 1, 1))
    for _, _, j in qa.FOOT_CHANNELS:                                    # feet planted on the ground
        J[:, j, 1] = qa.SOLE_OFFSET_M[j]
    J[:, 21, 0] += 0.5 * np.tanh((np.arange(L) - k) / 2.0)             # the swing peaks at contact
    Jc = J + [0, 0, 10.0]
    K = np.tile(_K(), (L, 1, 1))
    kp = np.zeros((L, 17, 3))
    uv = qa.project(Jc, K)
    for c, s in qa.COCO_TO_SMPL.items():
        kp[:, c] = np.c_[uv[:, s], np.full(L, 0.9)]
    return {"fps": FPS, "joints_world": J, "joints_camera": Jc, "K": K, "kp2d": kp,
            "static_conf": np.ones((L, 6)), "contact_index": k, "stroke": "forehand", "static_camera": True,
            "camera_motion_px": 0.1, "calibrated": False}


def test_timing_judges_the_curated_contact():
    m = _motion(k=30)
    m["contact_index"] = 20                                            # audio contact, as at motion time
    meta = {"contact_index": 30, "contact_index_selected": 20, "contact_source": "audio+motion"}
    t = qa.clip_metrics(m, meta)["timing"]
    assert t["offset_frames"] == 0 and t["audio_offset_frames"] == 10
    t = qa.clip_metrics(m, {})["timing"]                               # no clip.json contact: motion's
    assert t["contact_index"] == 20 and t["offset_frames"] == 10


def test_swing_speed_is_measured_around_the_contact():
    m = _motion(k=30)
    t = qa.clip_metrics(m, {"contact_index": 30})["timing"]
    assert t["swing_speed_ms"] == pytest.approx(0.5 * np.tanh(0.5) * FPS, rel=0.01)   # central difference
    t = qa.clip_metrics(m, {"contact_index": 5})["timing"]             # label far before the swing
    assert t["swing_speed_ms"] < 0.5 and t["offset_frames"] == 25
    assert qa.gate(qa.clip_metrics(m, {"contact_index": 5}))[0] == "FAIL"


def test_a_later_stroke_in_a_long_clip_is_not_this_ones_swing():
    m = _motion(L=130, k=30)
    m["joints_world"][:, 21, 0] += 0.8 * np.tanh((np.arange(130) - 85) / 2.0)   # a faster swing 2.2 s later
    m["joints_camera"] = m["joints_world"] + [0, 0, 10.0]
    t = qa.clip_metrics(m, {"contact_index": 30})["timing"]
    assert t["peak_frame"] == 30 and t["offset_frames"] == 0


def test_pose_jitter_ignores_the_root():
    m = _motion()
    m["joints_world"] = m["joints_world"] + np.random.default_rng(3).normal(0, 0.05, (50, 1, 3))  # root noise
    jit = qa.clip_metrics(m, {"contact_index": 30})["jitter"]
    assert jit["root_ms3"] > 1000
    clean = qa.clip_metrics(_motion(), {"contact_index": 30})["jitter"]["pose_ms3"]
    assert jit["pose_ms3"] == pytest.approx(clean)


def test_retarget_skate_is_per_source_frame():
    L = 20
    contact = np.ones((L, 4), bool)
    P = np.zeros((2 * L, 4, 3))                                        # animation at twice the source rate
    P[:, :, 0] = (np.arange(2 * L) * 0.01)[:, None]                     # 1 cm per animation frame
    anim = {"names": np.array(["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]), "positions": P,
            "fps": 2 * FPS}
    out = qa.retarget_metrics(anim, contact, FPS)
    assert out["mean_cm"] == pytest.approx(2.0) and out["n_frames"] == 2 * L
    assert "error" in qa.retarget_metrics({"positions": P}, contact, FPS)


def _write_clip(root, cid: str = "forehand_01", L: int = 50, k: int = 30) -> None:
    clip = root / "clips" / cid
    clip.mkdir(parents=True)
    m = _motion(L, k)
    np.savez(clip / "motion.npz", **m)
    np.savez(clip / "track.npz", bbx_xyxy=np.tile([800, 300, 950, 600.0], (L, 1)), valid=np.ones(L, bool),
             track_id=1, coverage=1.0)
    (clip / "clip.json").write_text(json.dumps({"id": cid, "stroke": "forehand", "contact_index": k}))
    anim = root / "anim" / "joints"
    anim.mkdir(parents=True, exist_ok=True)
    names = ["Hips", "LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    P = m["joints_world"][:, [0, 7, 10, 8, 11]][..., [0, 2, 1]] * [1, -1, 1]   # Y-up -> Z-up
    np.savez(anim / f"{cid}.npz", names=np.array(names), positions=P, fps=FPS)


def test_run_writes_clip_reports_and_summary(tmp_path):
    _write_clip(tmp_path)
    (tmp_path / "clips" / "serve_09").mkdir()
    (tmp_path / "clips" / "serve_09" / "clip.json").write_text("{}")   # no motion yet
    cfg = Config.load()
    cfg.run_dir = tmp_path
    summary = qa.run(cfg)

    rep = json.loads((tmp_path / "clips" / "forehand_01" / "qa.json").read_text())
    assert rep["status"] == rep["gate"] == "PASS", rep["reasons"]
    assert rep["skate"]["mean_cm"] == 0 and rep["retarget"]["mean_cm"] == 0
    assert rep["retarget"]["joints"] == ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]
    assert rep["retarget"]["root_ms3"] == 0 and rep["retarget"]["pose_ms3"] == 0
    assert rep["reprojection"]["median_px"] == 0
    assert rep["timing"]["offset_frames"] == 0
    assert summary["missing_motion"] == ["serve_09"]
    assert summary["by_stroke"]["forehand"]["PASS"] == 1
    on_disk = json.loads((tmp_path / "qa_summary.json").read_text())
    assert on_disk["clips"][0]["id"] == "forehand_01"


def test_run_survives_an_unreadable_clip(tmp_path):
    _write_clip(tmp_path)
    _write_clip(tmp_path, "forehand_02")
    bad = tmp_path / "clips" / "forehand_02" / "motion.npz"
    m = dict(np.load(bad))
    m["joints_world"][3] = np.nan                                       # SVD cannot converge on NaN
    np.savez(bad, **m)
    cfg = Config.load()
    cfg.run_dir = tmp_path
    summary = qa.run(cfg)
    assert list(summary["errors"]) == ["forehand_02"]
    assert [r["id"] for r in summary["clips"]] == ["forehand_01"]
    rep = json.loads((tmp_path / "clips" / "forehand_02" / "qa.json").read_text())
    assert rep["gate"] == rep["status"] == "FAIL"
