"""Unit tests for the finger rig and racket seat: the numpy half of blender/fingers.py (seat geometry, the constant
hand offset, pose blending, weight bands) and the mesh helpers of rybakina3d.character.hands."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rybakina3d.blender import fingers as fg
from rybakina3d.blender import retarget as rt
from rybakina3d.character import hands as hd

# A right hand hanging at rest: long axis down, palm facing +x (towards the body), thumb forward (-y) = a x n.
A, N = np.array([0.0, 0, -1]), np.array([1.0, 0, 0])
T = np.cross(A, N)
WRIST = np.array([-0.4, 0.0, 0.9])


def palm_point(s: float, w: float, h: float) -> np.ndarray:
    return WRIST + A * s + T * w + N * h


def flat_palm(skin_h: float = 0.02, heel_w: float = -0.015) -> tuple[dict, np.ndarray]:
    """A palm frame and a flat palm skin skin_h out of the palm from the wrist joint."""
    palm = {"wrist": WRIST, "long_axis": A, "thumb_side": T, "normal": N, "centre": palm_point(0.08, 0.0, 0.0),
            "heel": palm_point(0.05, heel_w, skin_h), "index_knuckle": palm_point(0.11, 0.035, 0.0),
            "wrist_clear_s": 0.03}
    s, w = np.meshgrid(np.arange(0.02, 0.13, 0.002), np.arange(-0.04, 0.046, 0.002))
    skin = np.stack([palm_point(a, b, skin_h) for a, b in zip(s.ravel(), w.ravel(), strict=True)])
    return palm, skin


def tiny_rig() -> tuple[rt.Rig, dict[str, str]]:
    """Hips and a right hand bone (bone Y down the hand), unit-scale armature."""
    rest = np.tile(np.eye(4), (2, 1, 1))
    rest[0, :3, 3] = [0.0, 0.0, 1.0]
    rest[1, :3, :3] = np.stack([T, A, np.cross(T, A)], 1)   # X = thumb side, Y = long axis, Z = X x Y
    rest[1, :3, 3] = WRIST
    return rt.Rig(["Hips", "RightHand"], [-1, 0], rest, np.eye(4)), {"pelvis": "Hips", "right_wrist": "RightHand"}


def hand_rig_for(palm: dict, skin: np.ndarray, bones: list[dict] | None = None) -> dict:
    return {"version": fg.HAND_RIG_VERSION, "ok": True,
            "sides": {"Right": {"ok": True, "hand_bone": "RightHand", "palm": palm, "bones": bones or []}},
            "arrays": {"Right_co": skin, "Right_shares": np.tile(np.eye(fg.SHARES)[0], (len(skin), 1)),
                       "Right_palm": skin}}


# ------------------------------------------------------------------------------------------------------ seat
def test_seat_runs_from_heel_pad_to_index_knuckle_on_the_skin():
    palm, skin = flat_palm()
    seat = fg.seat_from_palm(palm, skin)
    M = fg.seat_matrix(seat)
    k = palm_point(0.11 - fg.KNUCKLE_SETBACK_M, 0.035, 0.0) - palm["heel"]
    expected = math.degrees(math.atan2(k @ T, k @ A))
    assert seat["angle_deg"] == pytest.approx(expected, abs=0.01)
    assert abs(M[:3, 1] @ np.cross(A * math.cos(math.radians(expected)) + T * math.sin(math.radians(expected)),
                                   N)) < 1e-9                              # the shaft stays over that plan line
    assert np.allclose(M[:3, :3].T @ M[:3, :3], np.eye(3)) and np.linalg.det(M[:3, :3]) == pytest.approx(1.0)
    assert M[:3, 2] @ N > math.cos(math.radians(fg.MAX_TILT_DEG + 0.1))  # string normal = palm normal (tilted)
    # butt: centre past the wrist so its rim clears the narrowest wrist section, on the plan line
    butt_s = 0.03 + fg.BUTT_RADIUS * math.sin(math.radians(expected)) + fg.BUTT_CLEAR_M
    assert seat["butt_past_wrist_m"] == pytest.approx(butt_s, abs=1e-4)
    assert (M[:3, 3] - WRIST) @ A == pytest.approx(butt_s, abs=1e-6)
    # the handle rests on the skin: nothing deeper than the sinks, and some skin within the skin's own sink
    rel = skin - M[:3, 3]
    y = rel @ M[:3, 1]
    gap = np.linalg.norm(rel - np.outer(y, M[:3, 1]), axis=1) - np.where(y < fg.BUTT_LENGTH, fg.BUTT_RADIUS,
                                                                        fg.HANDLE_RADIUS)
    on = (y >= 0) & (y < 0.1)
    assert gap[on].min() > -(fg.SINK_M + fg.PAD_SINK_M) - 1e-4
    assert gap[on].min() < -fg.SINK_M + 5e-4


def test_seat_angle_is_clamped_and_needs_skin():
    palm, skin = flat_palm(heel_w=-0.09)       # a heel far out on the little-finger side: too steep a diagonal
    seat = fg.seat_from_palm(palm, skin)
    assert seat["measured_angle_deg"] > fg.GRIP_ANGLE_RANGE_DEG[1]
    assert seat["angle_deg"] == fg.GRIP_ANGLE_RANGE_DEG[1] and "angle" in seat["clamped"]
    with pytest.raises(ValueError):
        fg.seat_from_palm(palm, skin[:5])


def test_seat_reports_the_limits_it_runs_into():
    palm, skin = flat_palm()
    assert fg.seat_from_palm(palm, skin)["clamped"] == []                   # a flat palm: no limit reached
    q = fg.palm_coords(palm, skin)
    deep = skin - np.outer(np.clip(q[:, 0] - 0.07, 0, None) * 2.0, N)      # the far palm falls away steeply
    tilted = fg.seat_from_palm(palm, deep)
    assert tilted["tilt_deg"] == pytest.approx(-fg.MAX_TILT_DEG, abs=0.05) and "tilt" in tilted["clamped"]
    far = fg.seat_from_palm({**palm, "wrist_clear_s": 0.06}, skin)         # a long wristband: butt held at 5 cm
    assert far["butt_past_wrist_m"] == fg.BUTT_PAST_WRIST_M[1] and "butt_past_wrist" in far["clamped"]


def test_rest_line_rests_on_the_highest_points():
    y = np.array([0.0, 0.02, 0.05, 0.08, 0.1])
    c = np.array([0.010, 0.005, 0.002, 0.005, 0.008])     # a hollow between two pads
    h0, k = fg.rest_line(y, c, 0.5)
    line = h0 + k * y
    assert (line >= c - 1e-9).all()
    assert np.isclose(line[0], 0.010, atol=1e-4) and np.isclose(line[-1], 0.008, atol=1e-4)   # touches both pads
    assert fg.rest_line(y, c, 0.0)[1] == 0.0 and fg.rest_line(y, c, 0.0)[0] == pytest.approx(0.010)
    h0, k = fg.rest_line(y, c, 0.5, at=0.0)                # pressed down at the first end: rests on it
    assert h0 == pytest.approx(0.010) and (h0 + k * y >= c - 1e-9).all()


def test_seat_frame_compensates_the_fixed_axis_and_falls_back():
    rig, mapping = tiny_rig()
    fixed = rt.grip_matrix(rig, mapping)
    a = fixed[:3, 1]                            # the fixed grip: 25 deg from the bone axis towards the thumb
    assert math.degrees(math.acos(a @ A)) == pytest.approx(rt.GRIP_ANGLE_DEG, abs=1e-6)
    for missing in (None, {"version": fg.HAND_RIG_VERSION, "ok": False, "reason": "coarse hand"}):
        out = fg.seat_frame(missing, rig, mapping)
        assert out["source"] == "fixed" and np.allclose(out["matrix"], fixed) and out["warning"]
        assert np.allclose(out["offset_quat"], [1, 0, 0, 0])
    palm, skin = flat_palm()
    out = fg.seat_frame(hand_rig_for(palm, skin), rig, mapping)
    assert out["source"] == "palm" and out["warning"] is None
    M, C = out["matrix"], out["offset_world"]
    assert np.allclose(C @ M[:3, :3], fixed[:3, :3], atol=1e-9)            # shaft and string normal as fixed
    assert out["axis_vs_fixed_deg"] == 0.0 and out["normal_vs_fixed_deg"] == 0.0
    B = fg._bone_world_rot(rig, "RightHand")
    assert np.allclose(B @ rt.quat_to_matrix(out["offset_quat"]) @ B.T, C, atol=1e-9)   # same rotation, bone frame
    # mostly a turn about the palm normal (the seat is steeper than the fixed 25 deg): an ulnar offset of the hand
    about = out["offset_about_deg"]
    assert abs(about["palm_normal"]) > max(abs(about["long_axis"]), abs(about["thumb_side"]))
    assert np.allclose(out["hand_seat"], np.linalg.inv(np.block([[B, WRIST[:, None]], [np.zeros((1, 3)), 1]])) @ M)


def test_inside_handle_is_a_cylinder_from_the_butt():
    M = np.eye(4)
    pts = np.array([[0.0, 0.05, 0.0], [0.015, 0.1, 0.0], [0.017, 0.1, 0.0], [0.0, -0.01, 0.0], [0.0, 0.21, 0.0]])
    assert fg.inside_handle(pts, M).tolist() == [True, True, False, False, False]


def test_inside_racket_covers_butt_grip_shaft_and_throat_arms():
    M = np.eye(4)
    M[:3, 3] = [0.3, -0.2, 1.0]                                           # anywhere: the racket frame decides
    arm = np.array([fg.THROAT_ARM_END[0], fg.THROAT_ARM_END[1] - fg.THROAT_Y])
    local = np.array([
        [0.017, 0.005, 0.0],                                              # butt cap (1.85 cm): inside
        [0.017, 0.05, 0.0],                                               # past the grip's 1.6 cm: outside
        [0.0, 0.10, 0.015],                                               # in the grip
        [0.009, 0.22, 0.0], [0.0, 0.22, 0.012],                           # shaft box: inside, then past its depth
        [0.5 * arm[0], fg.THROAT_Y + 0.5 * arm[1], 0.0],                  # mid-way up one arm
        [-0.5 * arm[0], fg.THROAT_Y + 0.5 * arm[1], 0.005],               # ... and the other
        [0.0, fg.THROAT_Y + 0.5 * arm[1], 0.0],                           # between the arms (the open throat)
        [0.0, 0.5, 0.0],                                                  # the string bed: not counted
    ])
    pts = local @ M[:3, :3].T + M[:3, 3]
    assert fg.inside_racket(pts, M).tolist() == [True, False, True, True, False, True, True, False, False]


# ------------------------------------------------------------------------------------------------- finger pose
def bones_of_a_finger() -> tuple[list[dict], list[np.ndarray]]:
    """Three bones down the hand with known world rotations about their heads (grip) and none relaxed."""
    heads = [WRIST + A * d for d in (0.10, 0.145, 0.175, 0.195)]
    R = [rt.axis_angle_to_matrix(np.cross(A, N) * math.radians(v)) for v in (60.0, 80.0, 30.0)]
    names = fg.bone_names("RightHand")[3:6]
    bones = [{"name": names[i], "parent": "RightHand" if i == 0 else names[i - 1], "head": heads[i],
              "tail": heads[i + 1], "z_axis": N, "grip_world": rt.matrix_to_quat(R[i]),
              "relaxed_world": np.array([1.0, 0, 0, 0])} for i in range(3)]
    return bones, R


def test_finger_pose_blends_from_relaxed_to_grip():
    bones, R = bones_of_a_finger()
    palm, skin = flat_palm()
    hr = hand_rig_for(palm, skin, bones)
    q0, q1 = fg.finger_pose_quats(hr, "Right", 0.0), fg.finger_pose_quats(hr, "Right", 1.0)
    half = fg.finger_pose_quats(hr, "Right", np.array([0.0, 0.5, 1.0]))
    for b, Rw in zip(bones, R, strict=True):
        B = fg.bone_frame(b["head"], b["tail"], b["z_axis"])
        assert np.allclose(q0[b["name"]], [1, 0, 0, 0])
        assert np.allclose(B @ rt.quat_to_matrix(q1[b["name"]]) @ B.T, Rw, atol=1e-9)   # local -> world again
        track = half[b["name"]]
        assert track.shape == (3, 4) and np.allclose(track[0], q0[b["name"]]) and np.allclose(track[2], q1[b["name"]])
        full, mid = rt.rotation_angle_deg(Rw), rt.rotation_angle_deg(rt.quat_to_matrix(track[1]))
        assert mid == pytest.approx(full / 2, abs=1e-6)
    # a flexion about the bone's own X axis: the bones' Z axes point to the palm side
    assert abs(q1[bones[0]["name"]][1]) > 0.4 and np.allclose(q1[bones[0]["name"]][2:], 0, atol=1e-9)
    names, tracks = fg.finger_tracks(hr, {"Right": np.linspace(0, 1, 5), "Left": 1.0}, 5)
    assert names == [b["name"] for b in bones] and tracks.shape == (5, 3, 4)   # no usable left hand: left out


def test_quaternion_helpers():
    rng = np.random.default_rng(1)
    q, r = rt.matrix_to_quat(rt.axis_angle_to_matrix(rng.normal(size=(2, 10, 3))))
    assert np.allclose(rt.quat_to_matrix(rt.quat_mul(q, r)), rt.quat_to_matrix(q) @ rt.quat_to_matrix(r))
    assert np.allclose(fg.quat_slerp(q[0], r[0], 0.0), q[0]) and np.allclose(np.abs(fg.quat_slerp(q[0], r[0], 1.0)
                                                                                    @ r[0]), 1.0)
    # a constant offset right-multiplied into a track commutes with smoothing (retarget.smooth_quats)
    track = rt.smooth_quats(rt.matrix_to_quat(rt.axis_angle_to_matrix(np.cumsum(rng.normal(size=(30, 3)) * 0.1, 0))),
                            0)
    c = r[3]
    assert np.allclose(rt.smooth_quats(rt.quat_mul(track, c), 5), rt.quat_mul(rt.smooth_quats(track, 5), c))


# ---------------------------------------------------------------------------------------------- weight bands
def straight_hand() -> tuple[np.ndarray, dict, dict, np.ndarray]:
    """Five straight fingers along -z from knuckles 2 cm apart, vertices on rings around them plus a palm block."""
    joints, pts, owner = {}, [], []
    for k, f in enumerate(fg.FINGERS):
        base = palm_point(0.10, 0.04 - 0.02 * k, 0.0)
        joints[f] = np.stack([base + A * d for d in (0.0, 0.045, 0.075, 0.095)])
        for d in np.arange(-0.01, 0.1, 0.004):
            for ang in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                pts.append(base + A * d + 0.008 * (math.cos(ang) * T + math.sin(ang) * N))
                owner.append(k)
    for s in np.arange(0.02, 0.08, 0.005):
        for w in np.arange(-0.05, 0.05, 0.005):
            pts.append(palm_point(s, w, 0.0))
            owner.append(-1)
    owner = np.array(owner)
    labels = np.zeros((len(pts), 5))
    labels[owner >= 0, owner[owner >= 0]] = 1.0
    labels[owner < 0] = 0.2
    return np.array(pts), joints, {f: 0.008 for f in fg.FINGERS}, labels


def test_band_shares_partition_the_hand_weight():
    P, joints, radius, labels = straight_hand()
    S = fg.band_shares(P, joints, radius, labels, {f: 0.08 for f in fg.FINGERS})
    assert S.shape == (len(P), fg.SHARES) and (S >= 0).all() and np.allclose(S.sum(1), 1.0)
    palm = (P - WRIST) @ A < 0.08
    assert np.allclose(S[palm, 0], 1.0)                                    # the palm stays on the hand
    idx = fg.FINGERS.index("middle")
    J = joints["middle"]
    on_chain = np.argmin(np.linalg.norm(P - (J[3] - A * 0.002 + 0.008 * T), axis=1))   # just before the tip
    assert S[on_chain, 3 + 3 * idx] > 0.99                                 # the last bone
    pip = np.argmin(np.linalg.norm(P - (J[1] + 0.008 * T), axis=1))       # a ring at the middle joint
    assert 0.25 < S[pip, 1 + 3 * idx] < 0.75 and S[pip, 1 + 3 * idx] + S[pip, 2 + 3 * idx] > 0.99   # shared
    w_hand = np.linspace(0.0, 1.0, len(P))
    W = fg.split_hand_weight(w_hand, S)
    assert np.allclose(W.sum(1), w_hand)                                  # the wrist blend is untouched


def test_shares_match_by_position_and_default_to_the_hand():
    P, joints, radius, labels = straight_hand()
    S = fg.band_shares(P, joints, radius, labels)
    query = np.concatenate([P[::7] + 1e-4, [[5.0, 5.0, 5.0]]])
    shares, dist = fg.match_shares(query, P, S)
    assert np.allclose(shares[:-1], S[::7]) and dist[:-1].max() < 2e-4
    assert np.allclose(shares[-1], np.eye(fg.SHARES)[0])


# --------------------------------------------------------------------------------------- character/hands.py
def cube(size: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (0, size) for y in (0, size) for z in (0, size)], np.float64)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return v, np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])


def test_winding_number_tells_inside_from_outside():
    v, tri = cube()
    w = hd.winding_number(v, tri, np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2], [1.5, 0.5, 0.5], [-0.2, -0.2, 2.0]]))
    assert np.allclose(np.abs(w), [1, 1, 0, 0], atol=1e-6)
    open_box = tri[2:]                                     # one face missing: deep inside still counts as inside
    assert abs(hd.winding_number(v, open_box, np.array([[0.5, 0.5, 0.9]]))[0]) > 0.5


def test_dedupe_merges_split_vertices_for_the_surface_graph():
    v, tri = cube()
    split = np.concatenate([v, v])                         # every triangle on its own copy of the corners
    tri2 = np.concatenate([tri[:6], tri[6:] + 8])
    co, t, inv = hd.dedupe(split, tri2)
    assert len(co) == 8 and np.allclose(co[inv], split)
    G = hd.surface_graph(co, t)
    assert G.nnz == 2 * 18 and np.isclose(G.max(), math.sqrt(2))          # 12 edges + 6 face diagonals, each once


def test_band_shares_fit_gltf_influences():
    P, joints, radius, labels = straight_hand()
    labels = 0.5 * labels + 0.5 * np.roll(labels, 1, axis=1)     # every vertex shared by two fingers
    S = fg.band_shares(P, joints, radius, labels)
    assert ((S > 0).sum(1) <= fg.MAX_SHARES).all() and np.allclose(S.sum(1), 1.0)


# ----------------------------------------------------------------------------------------- grip metrics / IK palm
def test_palm_gap_measures_to_the_handle_surface():
    M = np.eye(4)                                             # the butt at the origin, the shaft along +y
    pts = np.array([[0.020, 0.05, 0.0], [0.0, 0.05, 0.010], [0.019, 0.005, 0.0],    # over the grip, the butt cap
                    [0.020, 0.13, 0.0], [0.040, 0.05, 0.0], [0.020, -0.01, 0.0]])   # past the span / too far / behind
    assert np.allclose(fg.palm_gap(pts, M), [0.020 - fg.HANDLE_RADIUS, 0.010 - fg.HANDLE_RADIUS,
                                             0.019 - fg.BUTT_RADIUS])


def test_palm_in_bone_and_the_left_ik_palm_follow_the_skin_and_the_seat():
    rig, mapping = tiny_rig()
    palm, skin = flat_palm(skin_h=0.02)
    centre, normal = fg.palm_in_bone(hand_rig_for(palm, skin), rig, "Right")
    B = fg._bone_world_rot(rig, "RightHand")
    assert np.allclose(B @ centre + WRIST, palm_point(0.08, 0.0, 0.02))   # on the skin over the stored centre
    assert np.allclose(B @ normal, N) and fg.palm_in_bone(None, rig, "Right") is None
    # the same hand as a left one: the IK's palm puts the handle on that hand's seat
    left = rt.Rig(["Hips", "LeftHand"], rig.parents, rig.rest, rig.obj)
    hr = hand_rig_for(palm, skin)
    hr["sides"] = {"Left": {**hr["sides"]["Right"], "hand_bone": "LeftHand"}}
    hr["arrays"] = {k.replace("Right", "Left"): v for k, v in hr["arrays"].items()}
    lmap = {"pelvis": "Hips", "left_wrist": "LeftHand"}
    out = fg.left_ik_palm(hr, left, lmap, 0.015)
    S = fg.seat_frame(hr, left, lmap, "Left")["hand_seat"]
    assert np.allclose(out["palm_normal"], S[:3, 2]) and np.allclose(out["handle_dir"], S[:3, 1])
    q = out["palm_centre"] + 0.015 * out["palm_normal"]                  # twohand's handle axis point
    rel = q - S[:3, 3]
    assert np.linalg.norm(rel - (rel @ S[:3, 1]) * S[:3, 1]) < 1e-9       # on the seat's axis ...
    c = fg.palm_in_bone(hr, left, "Left")[0]
    assert abs((c - q) @ S[:3, 1]) < 1e-9                                 # ... level with the palm centre
    assert fg.left_ik_palm(hand_rig_for(palm, skin), rig, mapping, 0.015) is None   # no left hand
