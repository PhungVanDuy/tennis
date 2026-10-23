"""Unit tests for the retarget driver (rybakina3d.assemble): input validation and the re-import gate."""

from __future__ import annotations

import numpy as np

from rybakina3d import assemble


def _motion(path, n: int = 5, **override) -> None:
    m = {"fps": 25.0, "global_orient": np.zeros((n, 3)), "body_pose": np.zeros((n, 63)),
         "joints_world": np.zeros((n, 24, 3)), "rest_joints": np.zeros((24, 3)),
         "joints_camera": np.ones((n, 24, 3)), "K": np.tile(np.eye(3), (n, 1, 1))}
    m.update(override)
    np.savez(path, **{k: v for k, v in m.items() if v is not None})


def test_motion_problem_flags_missing_short_and_nan(tmp_path):
    ok = tmp_path / "ok.npz"
    _motion(ok)
    assert assemble.motion_problem(ok) is None
    bad = tmp_path / "nan.npz"
    _motion(bad, body_pose=np.full((5, 63), np.nan))
    assert "non-finite" in assemble.motion_problem(bad)
    short = tmp_path / "short.npz"
    _motion(short, n=1)
    assert "frames" in assemble.motion_problem(short)
    lacking = tmp_path / "lacking.npz"
    _motion(lacking, K=None)
    assert "lacks K" in assemble.motion_problem(lacking)
    torn = tmp_path / "torn.npz"
    torn.write_bytes(ok.read_bytes()[:100])
    assert "unreadable" in assemble.motion_problem(torn)


def test_check_frames_cover_the_stroke_sequence():
    assert assemble._check_frames({"stroke": "serve", "contact_index": 55}, 81) == [35, 47, 55, 63]
    assert assemble._check_frames({"stroke": "slice", "contact_index": 32}, 58) == [32]
    assert assemble._check_frames({"stroke": "serve", "contact_index": 75}, 81) == [55, 67, 75, 80]


def test_verify_problems_catch_a_misplaced_racket_and_a_bad_replay():
    report = {"mapping": {"right_wrist": "RightHand"}, "racket": {"bone": "RightHand"},
              "clips": {"a": {"frames": 58}, "b": {"frames": 81}}}
    good = {"glb_names_match": True, "glb_racket_parent": "BONE:RightHand",
            "glb_racket_rest": {"butt_mm": 0.1, "axis_deg": 0.01, "normal_deg": 0.02},
            "glb_clips": {"a": {"frames": 58, "joint_error_mm": 0.02, "racket_drift_mm": 0.0},
                          "b": {"frames": 81, "joint_error_mm": 0.02, "racket_drift_mm": 0.0}}}
    assert assemble.verify_problems(good, report, ["a", "b"]) == []
    bad = {**good, "glb_racket_rest": {"butt_mm": 25100.0, "axis_deg": 0.0, "normal_deg": 0.0},
           "glb_clips": {"a": {"frames": 57, "joint_error_mm": 0.02},
                         "b": {"frames": 81, "joint_error_mm": 4.0}}}
    problems = assemble.verify_problems(bad, report, ["a", "b"])
    assert len(problems) == 3 and "racket" in problems[0] and "57 frames" in problems[1]
    # with finger bones built, the export must carry them and their keys in every clip
    hands = {**report, "hands": {"finger_bones": 30}}
    kept = {**good, "glb_finger_bones": 30, "glb_finger_keys": {"a": 30, "b": 30}, "fbx_finger_bones": 30}
    assert assemble.verify_problems(kept, hands, ["a", "b"]) == []
    assert assemble.verify_problems(good, report, ["a", "b"]) == []        # no fingers: nothing asked
    lost = {**kept, "glb_finger_bones": 0, "glb_finger_keys": {"a": 0, "b": 30}, "fbx_finger_bones": 28}
    problems = assemble.verify_problems(lost, hands, ["a", "b"])
    assert len(problems) == 3 and "0 finger bones in the GLB" in problems[0] and problems[1].startswith("a: 0 of 30")
