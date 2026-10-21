"""Character-sheet poses (rybakina3d.artwork.poses) and the numpy half of blender/artwork.py: moments, clip
choice per slot, camera placement and framing, on synthetic joint tracks."""

from __future__ import annotations

import math

import numpy as np

from rybakina3d.artwork import poses
from rybakina3d.artwork.layout import Slot
from rybakina3d.blender import artwork as bl

FPS = 25.0
JOINTS = ["pelvis", "left_hip", "right_hip", "spine3", "neck", "head", "left_wrist", "right_wrist", "left_ankle",
          "right_ankle"]


def _clip(cid: str, stroke: str, n: int = 60, contact: int = 40, label_ok: bool = True, qa: str = "PASS",
          score: float = 2.0, hand_gap: float = 0.1, swing: float = 0.0) -> poses.ClipPose:
    """Standing figure facing -Y; wrists `hand_gap` apart in front of her; `swing` m/frame of hand travel."""
    P = np.zeros((n, len(JOINTS), 3))
    ix = {j: i for i, j in enumerate(JOINTS)}
    P[:, ix["pelvis"]] = [0, 0, 1.0]
    P[:, ix["left_hip"]] = [0.1, 0, 0.95]
    P[:, ix["right_hip"]] = [-0.1, 0, 0.95]
    P[:, ix["spine3"]] = [0, 0, 1.4]
    P[:, ix["neck"]] = [0, 0, 1.55]
    P[:, ix["head"]] = [0, 0, 1.7]
    P[:, ix["left_wrist"]] = [hand_gap / 2, -0.4, 1.0]
    P[:, ix["right_wrist"]] = [-hand_gap / 2, -0.4, 1.0]
    P[:, ix["left_wrist"], 0] += swing * np.arange(n)
    P[:, ix["right_wrist"], 0] += swing * np.arange(n)
    P[:, ix["left_ankle"]] = [0.15, 0, 0.08]
    P[:, ix["right_ankle"]] = [-0.15, 0, 0.08]
    return poses.ClipPose(cid, stroke, label_ok, qa, score, contact, P, ix, FPS)


def test_trophy_is_the_end_of_the_toss_apex_before_contact():
    z = np.full(80, 1.0)
    z[30:45] = 1.9 + 0.02 * np.sin(np.linspace(0, math.pi, 15))   # the toss hand hangs at its apex
    z[70:] = 2.3                                                   # higher after contact: outside the window
    assert poses.trophy_frame(z, contact=60, fps=FPS) == 44
    assert poses.trophy_frame(z, contact=0, fps=FPS) == 0
    # only the 2 s before contact count
    z2 = np.full(120, 1.0)
    z2[5] = 2.5
    z2[90] = 1.8
    assert poses.trophy_frame(z2, contact=100, fps=FPS) == 90


def test_moments_and_offsets_are_clamped_to_the_clip():
    c = _clip("serve_a", "serve", n=60, contact=50)
    c.positions[:, c.joints["left_wrist"], 2] = np.where(np.arange(60) == 33, 2.0, 1.0)
    assert poses.moment_frame(c, "contact", 0) == 50
    assert poses.moment_frame(c, "contact", 20) == 59
    assert poses.moment_frame(c, "start", -5) == 0
    assert poses.moment_frame(c, "trophy", 2) == 35
    # a left-hander tosses with the right hand
    c.positions[:, c.joints["right_wrist"], 2] = np.where(np.arange(60) == 20, 2.0, 1.0)
    assert poses.moment_frame(c, "trophy", 0, handedness="left") == 20


def test_reads_as_separates_two_handed_backhands_from_forehands():
    assert poses.reads_as(_clip("b", "backhand", hand_gap=0.12))
    assert not poses.reads_as(_clip("b", "backhand", hand_gap=0.8))
    assert poses.reads_as(_clip("f", "forehand", hand_gap=0.8))
    assert not poses.reads_as(_clip("f", "forehand", hand_gap=0.12))
    assert poses.reads_as(_clip("b", "backhand", hand_gap=0.8), backhand="one_handed")


def test_slots_prefer_valid_readable_unused_clips_and_never_repeat_a_moment():
    clips = [
        _clip("bh_flagged", "backhand", label_ok=False, score=9.0),
        _clip("bh_forehandish", "backhand", hand_gap=0.9, score=5.0),
        _clip("bh_warn", "backhand", qa="WARN", score=3.0),
        _clip("bh_pass", "backhand", qa="PASS", score=2.0),
    ]
    used: dict[str, list[int]] = {}
    spec = {"stroke": "backhand", "at": "contact", "offset": 0}
    a = poses.resolve_slot(Slot("a", (0, 0, 1, 1), spec), clips, used)
    b = poses.resolve_slot(Slot("b", (0, 0, 1, 1), spec), clips, used)
    assert (a["clip"], a["frame"]) == ("bh_pass", 40)       # QA PASS beats a better-scored WARN
    assert (b["clip"], b["frame"]) == ("bh_warn", 40)       # then the next clip, not the same moment again
    # 3 frames later both used clips are too close to their first moment: only the unreadable one is left
    c = poses.resolve_slot(Slot("c", (0, 0, 1, 1), {**spec, "offset": 3}), clips, used)
    assert c["clip"] == "bh_forehandish"
    # far enough from it, a used clip is fine again; the flagged clip is never taken while valid ones exist
    d = poses.resolve_slot(Slot("d", (0, 0, 1, 1), {**spec, "offset": -20}), clips, used)
    assert (d["clip"], d["frame"]) == ("bh_pass", 20)
    assert all(abs(f - g) >= poses.MIN_MOMENT_GAP for fs in used.values() for i, f in enumerate(fs)
               for g in fs[i + 1:])


def test_hero_takes_the_most_dynamic_clip_and_big_slots_choose_first():
    clips = [_clip("bh_calm", "backhand", score=5.0, swing=0.0),
             _clip("bh_fast", "backhand", qa="WARN", score=1.0, swing=0.15)]
    slots = [Slot("small", (0, 0, 0.1, 0.1), {"stroke": "backhand", "at": "contact", "offset": 0}),
             Slot("hero", (0, 0, 0.5, 0.5), {"stroke": "backhand", "at": "contact", "offset": 0}),
             Slot("front", (0, 0, 0.2, 0.2), "idle")]
    out = poses.resolve(slots, clips)
    assert [s["slot"] for s in out] == ["small", "hero", "front"]
    assert out[1]["clip"] == "bh_fast" and out[1]["choice"]["dynamism"]["speed"] > 3
    assert out[0]["clip"] == "bh_calm"
    assert out[2]["pose"] == "idle" and out[2]["clip"] is None and out[2]["frame"] is None


def test_hero_skips_moments_where_her_bowed_head_hides_the_face():
    bowed = _clip("bh_bowed", "backhand", swing=0.25)
    bowed.positions[:, bowed.joints["head"]] = [0, -0.12, 1.65]     # ~53 deg forward: eyes on the ball
    upright = _clip("bh_upright", "backhand", swing=0.1)
    assert poses.head_tilt(bowed, 40) > poses.HERO_MAX_HEAD_TILT_DEG > poses.head_tilt(upright, 40) == 0
    out = poses.resolve_slot(Slot("hero", (0, 0, 0.5, 0.5), {"stroke": "backhand", "at": "contact"}),
                             [bowed, upright], {})
    assert out["clip"] == "bh_upright" and out["choice"]["dynamism"]["head_tilt"] == 0


def test_missing_stroke_leaves_the_slot_empty_and_atlas_keeps_curation_order():
    clips = [_clip("serve_b", "serve", contact=30), _clip("fh_a", "forehand", contact=20, hand_gap=0.8)]
    out = poses.resolve_slot(Slot("x", (0, 0, 1, 1), {"stroke": "volley", "at": "contact"}), clips, {})
    assert out["pose"] is None and out["clip"] is None
    atlas = poses.atlas(clips)
    assert [a["clip"] for a in atlas] == ["serve_b", "fh_a"] and [a["frame"] for a in atlas] == [30, 20]


def test_dynamism_grows_with_speed_reach_stance_and_sitting_down():
    calm = _clip("a", "backhand")
    fast = _clip("b", "backhand", swing=0.2)
    assert poses.dynamism(fast, 40)["score"] > poses.dynamism(calm, 40)["score"]
    low = _clip("c", "backhand")
    low.positions[40, low.joints["pelvis"], 2] = 0.8
    assert poses.dynamism(low, 40)["drop"] == 0.2


# ------------------------------------------------------------------------------- blender/artwork.py (numpy)
def test_facing_and_camera_azimuth_conventions():
    # left hip -> right hip along +X: she faces +Z x +X = +Y
    assert math.isclose(bl.facing_yaw(np.array([-0.1, 0, 1]), np.array([0.1, 0, 1])), math.pi / 2)
    # the moveset's rest pose: left hip on +X, facing -Y
    yaw = bl.facing_yaw(np.array([0.1, 0, 1]), np.array([-0.1, 0, 1]))
    assert math.isclose(yaw, -math.pi / 2)
    target = np.array([0.0, 0.0, 1.0])
    front = bl.camera_matrix(target, yaw, 0, 0, 5.0)[:3, 3]
    left = bl.camera_matrix(target, yaw, 90, 0, 5.0)[:3, 3]
    back = bl.camera_matrix(target, yaw, 180, 0, 5.0)[:3, 3]
    assert np.allclose(front, [0, -5, 1], atol=1e-9)           # in front of her
    assert np.allclose(left, [5, 0, 1], atol=1e-9)             # her left hip is on +X
    assert np.allclose(back, [0, 5, 1], atol=1e-9)
    M = bl.camera_matrix(target, yaw, 0, 30, 5.0)
    assert M[2, 3] > 1.0 and np.allclose(-M[:3, 2], (target - M[:3, 3]) / 5.0)   # above, looking at her
    assert np.isclose(np.linalg.det(M[:3, :3]), 1.0) and abs(M[2, 0]) < 1e-9     # level horizon


def test_fit_frame_puts_the_points_inside_the_margins():
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.4, -0.2, 0.0], [0.5, 0.3, 1.8], size=(500, 3))
    cam = bl.camera_matrix(np.array([0.05, 0.05, 0.9]), 0.3, 25, 5, 7.0)
    frame = bl.fit_frame(bl.project(pts, cam), 1000, (0.04, 0.05, 0.03))
    px = bl.to_pixels(bl.project(pts, cam), frame)
    assert frame["height"] == 1000 and frame["width"] % 2 == 0
    assert math.isclose(px[:, 1].min(), 0.04 * 1000 / 1.07, rel_tol=1e-6)        # top margin
    assert math.isclose(1000 - px[:, 1].max(), 0.03 * 1000 / 1.07, rel_tol=1e-6)  # bottom margin
    assert px[:, 0].min() > 0 and px[:, 0].max() < frame["width"]
    assert math.isclose(px[:, 0].min(), frame["width"] - px[:, 0].max(), abs_tol=1.5)


def test_hanging_arm_points_down_out_and_forward():
    forward = np.array([0.0, -1.0, 0.0])
    d = bl.hanging_direction(np.array([0.7, 0.0, -0.7]), forward, 11.0, 4.0)
    assert np.isclose(np.linalg.norm(d), 1.0)
    assert d[2] < -0.95 and d[0] > 0.15 and d[1] < 0
    R = bl.rotation_between(np.array([0.7, 0.0, -0.7]), d)
    assert np.allclose(R @ (np.array([0.7, 0.0, -0.7]) / math.sqrt(0.98)), d)


def test_clean_shadow_drops_speckle_fades_borders_and_keeps_the_figure(tmp_path):
    import cv2

    from rybakina3d.artwork.render import clean_shadow, matte_bbox

    h, w = 200, 100
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 6                                  # catcher noise over the empty floor
    img[150:200, :, 3] = 120                         # a shadow running into the bottom and side borders
    img[20:160, 30:70] = (200, 200, 200, 255)        # the figure
    matte = np.zeros((h, w, 4), np.uint8)
    matte[20:160, 30:70, 3] = 255
    cv2.imwrite(str(tmp_path / "f.png"), img)
    cv2.imwrite(str(tmp_path / "f_matte.png"), matte)
    clean_shadow(tmp_path / "f.png", tmp_path / "f_matte.png")
    a = cv2.imread(str(tmp_path / "f.png"), cv2.IMREAD_UNCHANGED)[..., 3]
    assert a[5, 5] == 0 and a[100, 10] == 0                    # speckle gone
    assert a[199, 50] == 0 and a[175, 0] == 0                  # faded out at the borders
    assert a[175, 50] == 120 and (a[20:160, 30:70] == 255).all()
    assert matte_bbox(tmp_path / "f_matte.png") == [30, 20, 69, 159]
