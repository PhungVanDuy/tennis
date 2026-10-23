"""Unit tests for blender/twohand.py: the left hand onto the racket by two-bone IK (numpy only)."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from rybakina3d.blender import fingers as fg
from rybakina3d.blender import retarget as rt
from rybakina3d.blender import twohand as th

# A small Z-up rig facing -Y (Blender's front), so her left is +X. Bones: name, parent, head, tail.
BONES = [
    ("Hips", -1, (0, 0, 1.0), (0, 0, 1.1)),
    ("Spine", 0, (0, 0, 1.25), (0, 0, 1.4)),
    ("LeftShoulder", 1, (0.03, 0, 1.4), (0.17, 0, 1.4)),
    ("LeftArm", 2, (0.17, 0, 1.4), (0.361, 0, 1.209)),
    ("LeftForeArm", 3, (0.361, 0, 1.209), (0.538, 0, 1.032)),
    ("LeftHand", 4, (0.538, 0, 1.032), (0.587, 0, 0.983)),
    ("RightShoulder", 1, (-0.03, 0, 1.4), (-0.17, 0, 1.4)),
    ("RightArm", 6, (-0.17, 0, 1.4), (-0.361, 0, 1.209)),
    ("RightForeArm", 7, (-0.361, 0, 1.209), (-0.538, 0, 1.032)),
    ("RightHand", 8, (-0.538, 0, 1.032), (-0.587, 0, 0.983)),
    ("LeftUpLeg", 0, (0.1, 0, 1.0), (0.1, 0, 0.55)),
]
NAMES = [b[0] for b in BONES]
PARENTS = [b[1] for b in BONES]
IX = {n: i for i, n in enumerate(NAMES)}
ARM = th.LeftArm(collar=IX["LeftShoulder"], upper=IX["LeftArm"], fore=IX["LeftForeArm"], hand=IX["LeftHand"],
                 right_hand=IX["RightHand"])
PALM_C, PALM_N = np.array([0.0, 0.075, 0.03]), np.array([0.0, 0.0, 1.0])


def _rest() -> np.ndarray:
    rest = np.tile(np.eye(4), (len(BONES), 1, 1))
    for i, (_, _, h, t) in enumerate(BONES):
        y = np.subtract(t, h) / np.linalg.norm(np.subtract(t, h))
        x = np.cross(y, [0, 0, 1.0]) if abs(y[2]) < 0.99 else np.array([1.0, 0, 0])
        x /= np.linalg.norm(x)
        rest[i, :3, :3] = np.stack([x, y, np.cross(x, y)], 1)
        rest[i, :3, 3] = h
    return rest


REST = _rest()
RIG = rt.Rig(NAMES, PARENTS, REST, np.eye(4))


def _world_turn(b: int, R: np.ndarray) -> np.ndarray:
    """Basis that turns bone b by the world rotation R (its ancestors at rest)."""
    return REST[b, :3, :3].T @ R @ REST[b, :3, :3]


def _clip(L: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Both arms swung forward, the right one sweeping slowly with the racket across the front of the body, the left
    hand held ~10 cm off the throat. Returns (basis, pose, racket_in_hand)."""
    basis = np.tile(np.eye(3), (L, len(BONES), 1, 1))
    for k in range(L):
        z = math.radians(55 + 8 * math.sin(2 * math.pi * k / L))
        basis[k, IX["RightArm"]] = _world_turn(IX["RightArm"], Rotation.from_euler("z", z).as_matrix())
        basis[k, IX["RightForeArm"]] = Rotation.from_euler("x", math.radians(-50)).as_matrix()
        basis[k, IX["LeftArm"]] = _world_turn(IX["LeftArm"], Rotation.from_euler("z", math.radians(-40)).as_matrix())
        basis[k, IX["LeftForeArm"]] = Rotation.from_euler("x", math.radians(-35 - 10 * math.cos(k / 9))).as_matrix()
        basis[k, IX["LeftUpLeg"]] = Rotation.from_euler("x", math.radians(10 * math.sin(k / 7))).as_matrix()
    pose = rt.forward_kinematics(RIG, basis, 0, np.zeros((L, 3)))
    # racket at frame 0: the throat (0.27 m from the butt) 10 cm in front of the left palm, the shaft coming from
    # the right hand; then rigid on the right hand
    hand_l, hand_r = pose[0, IX["LeftHand"]], pose[0, IX["RightHand"]]
    palm = hand_l[:3, :3] @ PALM_C + hand_l[:3, 3]
    throat = palm + np.array([0.0, -0.10, 0.0])
    y = (throat - hand_r[:3, 3]) / np.linalg.norm(throat - hand_r[:3, 3])
    x = np.cross(y, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    world = np.eye(4)
    world[:3, :3] = np.stack([x, y, np.cross(x, y)], 1)
    world[:3, 3] = throat - 0.27 * y
    return basis, pose, np.linalg.inv(hand_r) @ world


def _run(L: int = 60, wrist_dist: np.ndarray | None = None, **params) -> tuple:
    basis, pose, racket = _clip(L)
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N,
                                                  wrist_dist=wrist_dist, params=th.Params(**params))
    after = rt.forward_kinematics(RIG, out, 0, np.zeros((L, 3)))
    return basis, pose, racket, out, w, closed, info, after


# ------------------------------------------------------------------------------------------ helpers
def test_rotation_helpers_match_scipy():
    rng = np.random.default_rng(0)
    R0 = Rotation.random(20, random_state=1).as_matrix()
    R1 = Rotation.random(20, random_state=2).as_matrix()
    q = th.quat_from_matrix(R0)
    assert np.allclose(np.abs(np.sum(q * Rotation.from_matrix(R0).as_quat()[:, [3, 0, 1, 2]], -1)), 1, atol=1e-9)
    assert np.allclose(th.matrix_from_quat(q), R0, atol=1e-9)
    t = rng.uniform(0, 1, 20)
    mid = th.slerp(R0, R1, t)
    for k in range(20):   # the slerp lies on the geodesic: angle(R0, mid) = t * angle(R0, R1)
        full = rt.rotation_angle_deg(R0[k].T @ R1[k])
        assert math.isclose(rt.rotation_angle_deg(R0[k].T @ mid[k]), t[k] * full, abs_tol=1e-6)
        assert math.isclose(rt.rotation_angle_deg(mid[k].T @ R1[k]), (1 - t[k]) * full, abs_tol=1e-6)
    assert np.allclose(th.slerp(R0, R1, 0.0), R0) and np.allclose(th.slerp(R0, R1, 1.0), R1)
    u, v = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
    M = th.rotation_between(u, v)
    assert np.allclose(np.einsum("lij,lj->li", M, u / np.linalg.norm(u, axis=1, keepdims=True)),
                       v / np.linalg.norm(v, axis=1, keepdims=True), atol=1e-9)
    anti = th.rotation_between(np.array([[0, 0, 1.0]]), np.array([[0, 0, -1.0]]))[0]
    assert np.allclose(anti @ [0, 0, 1.0], [0, 0, -1.0]) and np.isclose(np.linalg.det(anti), 1)


def test_twist_bend_split():
    R = Rotation.from_euler("y", 40, degrees=True).as_matrix()[None]
    tw, bend = th.twist_bend_deg(R)
    assert np.isclose(tw[0], 40) and np.isclose(bend[0], 0, atol=1e-6)
    R = Rotation.from_euler("x", 30, degrees=True).as_matrix()[None]
    tw, bend = th.twist_bend_deg(R)
    assert np.isclose(tw[0], 0, atol=1e-6) and np.isclose(bend[0], 30)


def test_grip_frame_puts_the_palm_on_the_opposite_face():
    G = th.left_grip_frame(PALM_C, PALM_N, None, 45.0, 0.015)
    assert np.allclose(G[:3, :3].T @ G[:3, :3], np.eye(3)) and np.isclose(np.linalg.det(G[:3, :3]), 1)
    assert np.allclose(G[:3, 2], -PALM_N)                                  # palm pushes on the racket's +Z face
    assert np.allclose(np.linalg.inv(G) @ np.append(PALM_C, 1), [0, 0, 0.015, 1])   # palm 15 mm off the axis
    # 45 deg from the long axis (bone Y) towards the left thumb (normal x long axis)
    thumb = np.cross(PALM_N, [0, 1.0, 0])
    assert np.isclose(G[:3, 1] @ [0, 1.0, 0], math.cos(math.radians(45)))
    assert G[:3, 1] @ thumb > 0
    c, n = th.default_palm(REST, IX["LeftHand"], IX["RightHand"])
    assert np.isclose(np.linalg.norm(n), 1) and abs(n[1]) < 1e-9
    world_n = REST[IX["LeftHand"], :3, :3] @ n
    assert world_n[0] < 0                                                  # faces the body (-X) at rest


# --------------------------------------------------------------------------------------------- IK
def test_two_bone_ik_reaches_keeps_lengths_and_the_bend_plane():
    rng = np.random.default_rng(3)
    n = 200
    S = np.tile([0, 0, 1.4], (n, 1))
    E = np.tile([0.25, 0.05, 1.2], (n, 1))
    W = np.tile([0.3, 0.27, 1.15], (n, 1))                               # elbow bent ~70 deg
    l1, l2 = np.linalg.norm(E[0] - S[0]), np.linalg.norm(W[0] - E[0])
    d = rng.uniform(0.12, 0.50, n)                                       # inside the 165 deg cap (0.55 m)
    dirs = rng.normal(size=(n, 3))
    target = S + dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * d[:, None]
    E2, W2, R1, R12, short = th.two_bone_ik(S, E, W, target, E, max_elbow_deg=165, soft_m=0.01)
    assert np.abs(np.linalg.norm(E2 - S, axis=1) - l1).max() < 1e-9
    assert np.abs(np.linalg.norm(W2 - E2, axis=1) - l2).max() < 1e-9
    assert short.max() < 0.005 and np.linalg.norm(W2 - target, axis=1).max() < 0.005
    # the rotations carry the bones there
    assert np.allclose(S + np.einsum("lij,lj->li", R1, E - S), E2, atol=1e-9)
    assert np.allclose(E2 + np.einsum("lij,lj->li", R12, W - E), W2, atol=1e-9)
    # the elbow bends the way the arm's own elbow does off its shoulder-wrist line, square to the target line
    u = (target - S) / d[:, None]
    a = (W - S) / np.linalg.norm(W - S, axis=1, keepdims=True)
    bend = (E - S) - np.sum((E - S) * a, 1, keepdims=True) * a
    pole = bend - np.sum(bend * u, 1, keepdims=True) * u
    assert np.all(np.sum((E2 - S) * pole, 1) > 0)
    assert np.abs(np.sum(np.cross(u, pole) * (E2 - S), 1)).max() < 1e-9
    # the elbow keeps bending about the same forearm axis: both bones take the arm plane's normal along
    n0, n2 = np.cross(E - S, W - E), np.cross(E2 - S, W2 - E2)
    n0, n2 = n0 / np.linalg.norm(n0, axis=1, keepdims=True), n2 / np.linalg.norm(n2, axis=1, keepdims=True)
    assert np.allclose(np.einsum("lij,lj->li", R1, n0), n2, atol=1e-6)
    assert np.allclose(np.einsum("lij,lj->li", R12, n0), n2, atol=1e-6)


def test_the_bend_plane_holds_as_the_target_crosses_a_nearly_straight_arm():
    """A nearly straight arm (bent ~20 deg) with its wrist target swept across the arm's own line: the elbow stays
    on the side it bends to and moves smoothly (taking the pole's offset from the shoulder-target line alone, it
    flipped to the other side once the target passed the elbow's lean, a 15-20 cm step in one frame)."""
    n = 41
    S = np.tile([0.0, 0.0, 1.4], (n, 1))
    E = np.tile([0.04, 0.0, 1.13], (n, 1))                                # bent towards +X
    W = np.tile([0.0, 0.0, 0.88], (n, 1))
    target = W + np.c_[np.linspace(-0.12, 0.12, n), np.zeros(n), np.zeros(n)]
    E2, W2, *_ = th.two_bone_ik(S, E, W, target, E, max_elbow_deg=170)
    assert np.all(E2[:, 0] - (target[:, 0] * (E2[:, 2] - 1.4) / (target[:, 2] - 1.4)) > 0.02)   # off the line, +X
    assert np.linalg.norm(np.diff(E2, axis=0), axis=1).max() < 0.01        # target steps 6 mm a frame


def test_two_bone_ik_caps_the_elbow_and_swivels():
    S, E, W = np.array([[0, 0, 1.4]]), np.array([[0.25, 0.05, 1.2]]), np.array([[0.45, 0.1, 1.05]])
    for far in (0.48, 0.6, 2.0):
        target = S + np.array([[far, 0, 0]])
        E2, W2, *_ = th.two_bone_ik(S, E, W, target, E, max_elbow_deg=165, soft_m=0.01)
        a, b = S[0] - E2[0], W2[0] - E2[0]
        included = math.degrees(math.acos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
        assert included <= 165 + 1e-6
    # swivel: the elbow turns about the shoulder->target axis, the wrist stays
    target = np.array([[0.2, -0.2, 1.1]])
    E0, W0, *_ = th.two_bone_ik(S, E, W, target, E)
    E1, W1, *_ = th.two_bone_ik(S, E, W, target, E, swivel=np.array([math.radians(90)]))
    u = (target - S)[0] / np.linalg.norm(target - S)
    assert np.allclose(W0, W1)
    r0, r1 = E0[0] - S[0] - ((E0[0] - S[0]) @ u) * u, E1[0] - S[0] - ((E1[0] - S[0]) @ u) * u
    assert math.isclose(math.degrees(math.acos(r0 @ r1 / np.linalg.norm(r0) / np.linalg.norm(r1))), 90, abs_tol=1e-6)


# ------------------------------------------------------------------------------------------- weight
def test_weight_thresholds_ignore_pops_and_ramp_inside_the_window():
    d = np.array([0.10, 0.20, 0.25, 0.30, 0.40])
    assert np.allclose(th.two_hand_weight(d, median_frames=1), [1, 1, 0.5, 0, 0])
    spike = np.full(20, 0.1)
    spike[10] = 0.6                                                        # one-frame GVHMR pop
    assert np.allclose(th.two_hand_weight(spike), 1)
    step = np.r_[np.ones(20), np.zeros(20)]
    w = th.limit_ramps(step, np.full(39, 0.25))
    assert np.abs(np.diff(w)).max() <= 0.25 + 1e-12 and np.all(w <= step)   # never raised ...
    assert np.allclose(w[16:20], [1.0, 0.75, 0.5, 0.25]) and np.all(w[20:] == 0)   # ... down before it is off
    assert np.allclose(th.limit_ramps(np.full(10, 0.7), np.full(9, 0.1)), 0.7)
    short = th.limit_ramps(np.r_[np.zeros(5), np.ones(3), np.zeros(5)], np.full(12, 0.25))
    assert np.isclose(short.max(), 0.5)                                    # too short for its ramps: a lower peak
    slow = th.limit_ramps(step, np.full(39, 0.05))                          # far pulls: longer ramps
    assert np.abs(np.diff(slow)).max() <= 0.05 + 1e-12
    wide = th.widen(step, 3)                                               # room for the ramps past the edge
    assert np.array_equal(wide[:20], step[:20]) and np.allclose(wide[20:24], [0.75, 0.5, 0.25, 0.0])
    assert np.array_equal(th.widen(step, 0), step)


def test_two_hand_state_holds_between_the_thresholds():
    d = np.array([0.25, 0.22, 0.19, 0.25, 0.29, 0.31, 0.25, 0.21, 0.20, 0.26, 0.40])
    assert np.array_equal(th.two_hand_state(d, median_frames=1), [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0])
    assert th.two_hand_state(np.full(5, 0.26), median_frames=1).sum() == 0     # starts nearer off_m: off
    assert th.two_hand_state(np.full(5, 0.24), median_frames=1).sum() == 5     # nearer on_m: on
    spike = np.full(20, 0.1)
    spike[10] = 0.6                                                        # one-frame GVHMR pop
    assert np.all(th.two_hand_state(spike) == 1) and th.two_hand_state(np.zeros(0)).shape == (0,)


def test_ramp_limit_slows_only_the_costly_levels():
    levels = np.linspace(0.0, 1.0, 5)
    step = np.r_[np.zeros(10), np.ones(30)]
    flat = th.ramp_limit(step, np.full((39, 4), 4.0), levels)              # 4 frames per unit everywhere ...
    assert np.allclose(flat, th.limit_ramps(step, np.full(39, 0.25)))      # ... is limit_ramps at 0.25
    assert np.isclose(th.advance(0.1, np.full(4, 4.0), levels), 0.35) and th.advance(0.9, np.full(4, 4.0), levels) == 1
    cost = np.tile([4.0, 4.0, 40.0, 4.0], (39, 1))                          # the stretch from 0.5 to 0.75 costs 10x
    w = th.ramp_limit(step, cost, levels)
    assert np.all(w <= step) and w[39] == 1.0
    rise = np.diff(w[10:])
    mid = (w[10:-1] >= 0.5) & (w[11:] <= 0.75)                              # the steps inside the costly stretch
    assert mid.sum() >= 9 and rise[mid].max() <= 1 / 40 + 1e-9 and rise.max() <= 0.25 + 1e-9
    assert np.argmax(w >= 1.0) - np.argmax(flat >= 1.0) >= 8               # the ramp takes ~9 frames longer


def test_runs_unwrap_and_quaternion_signs_keep_paths_continuous():
    on = np.array([0, 1, 1, 1, 0, 0, 1, 1, 0], bool)
    assert th.runs(on) == [(1, 4), (6, 8)] and th.runs(np.zeros(3, bool)) == []
    x = np.radians([5.0, 170, -170, -150, 50, 0, 350, 355, 7])
    u = np.degrees(th.unwrap_runs(x, on))
    assert np.allclose(u[1:4], [-190, -170, -150]) and np.allclose(u[6:8], [-10, -5])   # continuous, near 0
    assert np.allclose(u[[0, 4, 5, 8]], np.degrees(x[[0, 4, 5, 8]]))                   # off the runs: as given
    rng = np.random.default_rng(4)
    R = Rotation.from_rotvec(np.cumsum(rng.normal(size=(9, 3)) * 0.2, 0)).as_matrix()
    q = th.quat_from_matrix(R) * rng.choice([-1.0, 1.0], size=(9, 1))                  # random signs
    ref = np.tile(-th.quat_from_matrix(R[:1])[0], (9, 1))
    c = th.continuous_quats(q, on, ref)
    assert np.allclose(np.abs(np.sum(c * q, -1)), 1)                                    # the same rotations
    assert all(c[t] @ c[t - 1] > 0 for a, b in th.runs(on) for t in range(a + 1, b))
    assert all(np.sum(c[a:b] * ref[a:b]) > 0 for a, b in th.runs(on))
    # slerp_quat keeps the arc it is given: the long way round where the signs say so
    q0, q1 = th.quat_from_matrix(R[:1]), th.quat_from_matrix(R[3:4])
    assert np.allclose(th.matrix_from_quat(th.slerp_quat(q0, q1, 0.3)), th.slerp(R[:1], R[3:4], 0.3))
    long = th.matrix_from_quat(th.slerp_quat(q0, -q1, 0.5))[0]
    full = rt.rotation_angle_deg(R[0].T @ R[3])
    assert math.isclose(rt.rotation_angle_deg(R[0].T @ long), (360 - full) / 2, abs_tol=1e-6)


def test_fill_dips_closes_short_gaps_only():
    w = np.ones(40)
    w[10:14] = 0.3                                                         # 4 frames: GVHMR's wrists drifting apart
    w[25:37] = 0.0                                                         # 12 frames: the hand really lets go
    out = th.fill_dips(w, 9)
    assert np.all(out[:20] == 1) and np.array_equal(out[20:], w[20:])
    ramp = np.clip(np.linspace(-1, 2, 30), 0, 1)
    assert np.allclose(th.fill_dips(ramp, 9), ramp)                         # ramps and ends keep their values
    assert np.array_equal(th.fill_dips(w, 1), w)


# ------------------------------------------------------------------------------------------ the pass
def test_left_hand_lands_on_the_handle_and_nothing_else_moves():
    basis, pose, racket, out, w, closed, info, after = _run(wrist_dist=np.full(60, 0.12))
    others = [b for b in range(len(NAMES)) if b not in (ARM.collar, ARM.upper, ARM.fore, ARM.hand)]
    assert all(np.array_equal(out[:, b], basis[:, b]) for b in others)
    assert np.allclose(after[:, others], pose[:, others])
    assert np.all(w > 0.99)
    dist, facing = th.palm_on_handle(after, racket, ARM, PALM_C, PALM_N)
    before, _ = th.palm_on_handle(pose, racket, ARM, PALM_C, PALM_N)
    assert np.median(before) > 0.05                                        # it really was off the racket
    assert np.abs(dist - 0.015).max() < 0.002                              # palm on the handle (15 mm off the axis)
    assert facing.max() < 2.0
    assert info["reach_short_cm_max"] < 0.5 and info["elbow_deg_max"]["after"] <= 165.0 + 1e-6
    assert np.allclose(np.einsum("lij,lik->ljk", out[:, ARM.hand], out[:, ARM.hand]), np.eye(3), atol=1e-9)
    assert np.all(closed > 0.99) and info["frames_gripped"] == 60 and info["frames_open_on"] == []   # held: closed


def test_weight_follows_the_wrist_distance_without_pops():
    L = 90
    d = np.interp(np.arange(L), [0, 20, 30, 55, 65, 89], [0.1, 0.1, 0.45, 0.45, 0.1, 0.1])
    basis, pose, racket, out, w, closed, info, after = _run(L, wrist_dist=d)
    assert np.all(w[:5] > 0.99) and np.all(w[85:] > 0.99)                    # on while the wrists are close
    assert np.all(closed[40:46] == 0)                                       # the fingers close only on the racket
    assert closed[(w > 0.05) & (w < 0.5)].max() < 0.1                      # the hand half-way there: still open
    assert np.all(w[40:46] == 0)                                            # off while they are far apart ...
    assert np.array_equal(out[40:46], basis[40:46])                         # ... and the motion is untouched there
    assert np.abs(np.diff(w)).max() <= 1 / 4 + 1e-9                          # ramps of at least blend_frames
    assert np.abs(np.diff(closed)).max() <= th.Params().max_close_step + 1e-9   # the fingers never snap shut
    assert info["closing_max_step"] <= th.Params().max_close_step and info["pop_frames"] == []
    assert info["approach_wrist_bend_deg_max"] <= th.Params().max_wrist_bend_deg + 0.1
    assert abs(info["swivel_deg_max"]) <= 180.0
    # the palm never steps more than max_pop_m beyond the motion's and the full IK's steps (path ~15-25 cm here)
    assert info["max_pop_cm"] <= th.Params().max_pop_m * 100 + 0.1
    hand = after[:, ARM.hand]
    palm = np.einsum("lij,j->li", hand[:, :3, :3], PALM_C) + hand[:, :3, 3]
    hand0 = pose[:, ARM.hand]
    palm0 = np.einsum("lij,j->li", hand0[:, :3, :3], PALM_C) + hand0[:, :3, 3]
    assert np.linalg.norm(np.diff(palm, axis=0), axis=1).max() < np.linalg.norm(np.diff(palm0, axis=0), axis=1).max() \
        + 0.04
    # without the limiter (and its repair) the same blend would jump much further in one frame
    *_, info_fast, _ = _run(L, wrist_dist=d, max_blend_step_m=10.0, repair_rounds=0)
    dip = np.full(L, 0.1)
    dip[40:44] = 0.45                                                       # a 4-frame drift apart mid-stroke
    assert np.all(_run(L, wrist_dist=dip)[4] > 0.99)
    assert info_fast["max_pop_cm"] > 1.5 * info["max_pop_cm"]


def test_out_of_reach_turns_the_collar_within_its_limit():
    basis, pose, racket = _clip(40)
    far = racket.copy()
    far[:3, 3] += [0.0, 0.0, 0.25]                                          # handle 25 cm further from the hand
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, far, ARM, PALM_C, PALM_N,
                                                  wrist_dist=np.full(40, 0.1))
    turn = [rt.rotation_angle_deg(basis[k, ARM.collar].T @ out[k, ARM.collar]) for k in range(40)]
    assert max(turn) <= th.Params().max_collar_deg + 1e-6
    assert info["elbow_deg_max"]["after"] <= 165.0 + 1e-6
    no_collar = th.LeftArm(upper=ARM.upper, fore=ARM.fore, hand=ARM.hand, right_hand=ARM.right_hand)
    out2, _, closed2, info2 = th.left_hand_on_racket(REST, PARENTS, basis, pose, far, no_collar, PALM_C, PALM_N,
                                                     wrist_dist=np.full(40, 0.1))
    assert np.array_equal(out2[:, ARM.collar], basis[:, ARM.collar])
    if info2["reach_short_cm_max"] > 0.5:                                   # the collar buys reach when it is short
        assert info["reach_short_cm_max"] < info2["reach_short_cm_max"]
    short = np.array([k in info2["reach_short_frames"] for k in range(40)])
    if short.any():                                                         # the fingers stay open off the handle
        assert closed2[short].max() < 1.0 and closed2[short].max() <= closed[short].max() + 1e-9


def test_far_apart_wrists_leave_the_clip_as_it_was():
    basis, pose, racket = _clip(30)
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N,
                                                  wrist_dist=np.full(30, 0.5))
    assert np.array_equal(out, basis) and np.all(w == 0) and np.all(closed == 0) and info["frames_on"] == 0


def test_letting_go_to_toss_swings_the_arm_up_without_a_pop():
    """A serve: the left hand holds the racket, then lets go while the motion's left arm swings up overhead to toss,
    its upper arm and forearm turning far (over 150 deg) from where the IK holds them. The weight ramps off as the
    arm goes up; the hand must follow without flipping between frames."""
    L = 60
    basis, pose, racket = _clip(L)
    for k in range(25, L):
        t = min((k - 25) / 15, 1.0)                                        # over 15 frames: straight up and twisted
        up = Rotation.from_euler("y", -165 * t, degrees=True).as_matrix()   # about the body's front-back axis
        basis[k, IX["LeftArm"]] = _world_turn(IX["LeftArm"], up)
        basis[k, IX["LeftForeArm"]] = Rotation.from_euler("y", 160 * t, degrees=True).as_matrix()   # forearm roll
    pose = rt.forward_kinematics(RIG, basis, 0, np.zeros((L, 3)))
    d = np.interp(np.arange(L), [0, 24, 32, 59], [0.1, 0.1, 0.8, 0.9])
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N,
                                                  wrist_dist=d, hand_points=_hand_cloud())
    after = rt.forward_kinematics(RIG, out, 0, np.zeros((L, 3)))
    fk = th.chain_fk(pose, REST, PARENTS, {b: basis[:, b] for b in (ARM.collar, ARM.upper, ARM.fore, ARM.hand)})
    angle = [rt.rotation_angle_deg(fk[ARM.upper][k, :3, :3].T @ after[k, ARM.upper, :3, :3]) for k in range(10)]
    assert np.all(w[:5] > 0.99) and np.all(w[40:] == 0)                     # held, then off for the toss
    assert info["max_pop_cm"] <= th.Params().max_pop_m * 100 + 0.1 and info["pop_frames"] == []
    hand = after[:, ARM.hand]
    palm = np.einsum("lij,j->li", hand[:, :3, :3], PALM_C) + hand[:, :3, 3]
    palm0 = np.einsum("lij,j->li", pose[:, ARM.hand, :3, :3], PALM_C) + pose[:, ARM.hand, :3, 3]
    step, step0 = np.linalg.norm(np.diff(palm, axis=0), axis=1), np.linalg.norm(np.diff(palm0, axis=0), axis=1)
    assert step.max() <= step0.max() + 0.03                                 # never the long way round in a frame
    assert np.array_equal(out[40:], basis[40:]) and max(angle) > 5           # the toss untouched; the IK did act
    assert abs(info["swivel_deg_max"]) <= 180.0


def test_grip_amount_closes_as_the_hand_comes_down_onto_the_handle():
    p = th.Params()
    grip = th.left_grip_frame(PALM_C, PALM_N, None, 45.0, 0.015)
    racket = np.tile(np.eye(4), (6, 1, 1))
    held = np.linalg.inv(grip)                                              # the grip frame on the racket's axis ...
    hand = np.tile(held, (6, 1, 1))
    hand[:, 1, 3] += [0.14, 0.14, 0.14, 0.14, 0.14, 0.30]                   # ... 14 cm up (the last on the throat)
    hand[1, 0, 3] += 0.015                                                  # 1.5 cm off the axis
    hand[2, 0, 3] += 0.03                                                   # 3 cm off
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    hand[3] = turn @ hand[3]                                                # the handle 30 deg off the hand's
    spin = np.eye(4)
    spin[:3, :3] = Rotation.from_euler("y", 70, degrees=True).as_matrix()
    hand[4] = spin @ hand[4]                                                # rolled round the handle: still held
    c = th.grip_amount(hand, racket, grip, np.ones(6), p)
    # on its spot: closed; 1.5 cm off: three-quarters (closed would poke through the handle, relaxed curl into
    # it); 3 cm off: opening; turned off the shaft or up on the throat: relaxed
    assert np.isclose(c[0], 1.0) and np.isclose(c[1], p.close_hold) and 0.0 < c[2] < p.close_hold
    assert c[3] == 0 and np.isclose(c[4], 1.0) and c[5] == 0
    assert np.all(th.grip_amount(hand, racket, grip, np.zeros(6), p) == 0)   # the approach not under way: relaxed
    lift = np.linspace(0.0, 0.06, 61)
    curve = th.closing_for_lift(lift, p)
    assert np.all(np.diff(curve) <= 1e-12) and curve[0] == 1.0 and curve[-1] == 0.0
    assert np.allclose(curve[(lift >= p.close_near_m[1]) & (lift <= p.close_open_m[0])], p.close_hold)
    radial, s, angle = th.grip_lift(hand, racket, grip)
    assert np.allclose(radial[:3], [0.0, 0.015, 0.03]) and np.isclose(s[0], 0.14) and np.isclose(angle[3], 30)


# --------------------------------------------------------------------------------- the push off the racket
def _hand_cloud() -> np.ndarray:
    """(2,K,3) skin points of a flat left hand in its bone frame (the palm skin at PALM_C's height, the back 2 cm
    under it, fingers out to 17 cm from the wrist), the same at both closings."""
    x, y, z = np.meshgrid(np.linspace(-0.03, 0.03, 5), np.linspace(0.04, 0.17, 9), [PALM_C[2], PALM_C[2] - 0.02])
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], -1)
    return np.stack([pts, pts])


def test_racket_shape_is_the_one_fingers_tests():
    shape = th.RacketShape()
    assert shape == rt.racket_shape(fg)                                   # the defaults mirror racket.py's numbers
    rng = np.random.default_rng(5)
    q = rng.uniform([-0.1, -0.02, -0.03], [0.1, 0.45, 0.03], size=(20000, 3))
    world = np.eye(4)
    world[:3, :3] = Rotation.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix()
    world[:3, 3] = [0.3, -0.2, 1.1]
    inside = th.inside_racket(q, shape)
    assert inside.sum() > 500
    assert np.array_equal(inside, fg.inside_racket(q @ world[:3, :3].T + world[:3, 3], world))
    assert np.all(inside >= th.inside_racket(q, shape, 0.003))           # a margin only lets more points be


def test_hand_skin_poses_between_the_closings():
    pts = np.stack([np.zeros((4, 3)), np.ones((4, 3)), np.full((4, 3), 3.0)])   # closings 0, 0.5, 1
    out = th.hand_skin(pts, np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5]))
    assert out.shape == (6, 4, 3) and np.allclose(out[:, 0, 0], [0.0, 0.5, 1.0, 2.0, 3.0, 3.0])


def test_push_off_moves_the_skin_out_square_to_the_shaft_and_keeps_its_side():
    p, L = th.Params(), 40
    racket = np.tile(np.eye(4), (L, 1, 1))
    lifts = np.arange(0.0, p.max_push_m + 1e-9, p.push_step_m)
    blob = np.stack(np.meshgrid(np.linspace(-0.004, 0.004, 3), np.linspace(0.09, 0.11, 3),
                                np.linspace(0.008, 0.014, 3)), -1).reshape(-1, 3)   # just inside the grip, +Z side
    pts = np.tile(blob, (L, 1, 1)) + np.array([0.0, 0.0, 0.05])            # clear of the racket ...
    pts[15:25] -= [0.0, 0.0, 0.05]                                         # ... but for 10 frames
    push = th.push_off(pts, racket, th.RacketShape(), np.zeros(L), lifts, 7, p)
    moved = pts + push[:, None]
    assert not th.inside_racket(moved, th.RacketShape()).any()
    assert np.allclose(push[:, 1], 0) and np.all(push[15:25, 2] >= 0.005)  # out of the +Z face, square to the shaft
    assert np.all(push[:9] == 0) and np.all(push[31:] == 0)                # only around the frames that need it
    assert np.linalg.norm(np.diff(push, axis=0), axis=-1).max() < 0.01     # ramps in and out
    # a hand sweeping through the handle from +X to -X keeps to one side instead of swapping halfway
    sweep = np.tile(blob - [0.0, 0.0, 0.011], (L, 1, 1)) + np.linspace(0.03, -0.03, L)[:, None, None] * [1, 0, 0]
    push = th.push_off(sweep, racket, th.RacketShape(), np.zeros(L), lifts, 7, p)
    assert not th.inside_racket(sweep + push[:, None], th.RacketShape()).any()
    on = np.linalg.norm(push, axis=-1) > 0.005
    assert on.sum() > 5 and np.all(np.sum(push[on][1:] * push[on][:-1], -1) > 0)
    assert np.all(th.push_off(pts[:9], racket[:9], th.RacketShape(), np.zeros(9), lifts, 7, p) == 0)


def test_the_motions_own_hand_is_pushed_out_of_the_racket():
    L = 60
    basis, pose, racket = _clip(L)
    # the throat moved onto the left hand at frame 0: the sweeping right arm takes it through and off the hand
    hand_r = pose[0, IX["RightHand"]]
    shift = np.eye(4)
    shift[:3, 3] = [0.0, 0.10, -0.02]
    racket = np.linalg.inv(hand_r) @ shift @ hand_r @ racket
    cloud = _hand_cloud()
    far = np.full(L, 0.5)                                                   # the wrists apart: no weight at all

    def skin_inside(P: np.ndarray) -> np.ndarray:
        H = P[:, IX["LeftHand"]]
        R = th._frames(P[:, IX["RightHand"]] @ racket)
        q = np.einsum("lij,lkj->lki", H[:, :3, :3], np.broadcast_to(cloud[0], (L,) + cloud.shape[1:])) \
            + H[:, None, :3, 3]
        return th.inside_racket(np.einsum("lji,lkj->lki", R[:, :3, :3], q - R[:, None, :3, 3]),
                                th.RacketShape()).sum(-1)

    before = skin_inside(pose)
    assert before.max() >= 5 and (before > 0).sum() >= 20 and (before == 0).sum() > 10
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N,
                                                  wrist_dist=far, hand_points=cloud)
    after = rt.forward_kinematics(RIG, out, 0, np.zeros((L, 3)))
    assert np.all(w == 0) and np.all(closed == 0) and info["frames_pushed"] > 0
    left = skin_inside(after)                                               # (where the arm reaches the push)
    assert left.max() <= 1 and (left > 0).sum() <= 2
    untouched = np.linalg.norm(after[:, IX["LeftHand"], :3, 3] - pose[:, IX["LeftHand"], :3, 3], axis=-1) == 0
    assert untouched.sum() > 5 and np.array_equal(out[untouched], basis[untouched])    # bit for bit off the push
    assert info["max_pop_cm"] <= th.Params().max_pop_m * 100 + 0.1
    # without the skin there is no push: the clip stays as it was
    out2, *_ = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N, wrist_dist=far)
    assert np.array_equal(out2, basis)


def test_the_blend_keeps_the_skin_out_of_the_racket():
    L = 90
    d = np.interp(np.arange(L), [0, 20, 30, 55, 65, 89], [0.1, 0.1, 0.45, 0.45, 0.1, 0.1])
    basis, pose, racket = _clip(L)
    cloud = _hand_cloud()
    out, w, closed, info = th.left_hand_on_racket(REST, PARENTS, basis, pose, racket, ARM, PALM_C, PALM_N,
                                                  wrist_dist=d, hand_points=cloud)
    after = rt.forward_kinematics(RIG, out, 0, np.zeros((L, 3)))
    H = after[:, IX["LeftHand"]]
    R = th._frames(after[:, IX["RightHand"]] @ racket)
    q = np.einsum("lij,lkj->lki", H[:, :3, :3], np.broadcast_to(cloud[0], (L,) + cloud.shape[1:])) + H[:, None, :3, 3]
    q = np.einsum("lji,lkj->lki", R[:, :3, :3], q - R[:, None, :3, 3])
    deep = th.inside_racket(q, th.RacketShape(), th.Params().push_margin_m).sum(-1)
    assert deep.max() == 0                                                  # no skin deeper than the grip's sink
    palm = np.einsum("lji,lj->li", R[:, :3, :3], np.einsum("lij,j->li", H[:, :3, :3], PALM_C) + H[:, :3, 3]
                     - R[:, :3, 3])
    radial = np.hypot(palm[:, 0], palm[:, 2])
    assert radial.min() > th.Params().handle_above_palm_m - 0.002           # the palm never inside the handle
    assert np.all(w[:5] > 0.99) and np.all(w[85:] > 0.99) and np.all(w[40:46] == 0)
    assert info["max_pop_cm"] <= th.Params().max_pop_m * 100 + 0.1


def test_hand_points_pose_the_skin_with_the_fingers():
    # a left hand bone with one three-bone finger (the fingers test's finger, as a left one), unit-scale armature
    A, N = np.array([0.0, 0, -1]), np.array([1.0, 0, 0])
    T = np.cross(N, A)
    wrist = np.array([0.4, 0.0, 0.9])
    heads = [wrist + A * k for k in (0.10, 0.145, 0.175, 0.195)]
    names = ["Hips", "LeftHand", *fg.bone_names("LeftHand")[3:6]]
    rest = np.tile(np.eye(4), (5, 1, 1))
    rest[1, :3, :3] = np.stack([T, A, np.cross(T, A)], 1)
    rest[1, :3, 3] = wrist
    bones = []
    for i in range(3):
        rest[2 + i, :3, :3] = fg.bone_frame(heads[i], heads[i + 1], N)
        rest[2 + i, :3, 3] = heads[i]
        R = rt.axis_angle_to_matrix(np.cross(A, N) * math.radians((60.0, 80.0, 30.0)[i]))
        bones.append({"name": names[2 + i], "parent": names[1 + i], "head": heads[i], "tail": heads[i + 1],
                      "z_axis": N, "grip_world": rt.matrix_to_quat(R), "relaxed_world": np.array([1.0, 0, 0, 0])})
    rig = rt.Rig(names, [-1, 0, 1, 2, 3], rest, np.eye(4))
    hand_rig = {"version": fg.HAND_RIG_VERSION, "ok": True,
                "sides": {"Left": {"ok": True, "hand_bone": "LeftHand", "palm": {}, "bones": bones}},
                "arrays": {"Left_co": np.zeros((0, 3))}}
    co = np.stack([wrist + A * 0.05, heads[1] + N * 0.01, wrist - A * 0.02])   # palm, on the first finger bone,
    W = np.zeros((3, 5))                                                    # on the forearm (follows the hand)
    W[0, 1], W[1, 2], W[2, 0] = 1.0, 1.0, 1.0
    pts = fg.hand_points(hand_rig, rig, "Left", co, W, closings=(0.0, 1.0), every=1)
    local = (co - wrist) @ rest[1, :3, :3]                                 # the rest skin in the hand's frame
    assert pts.shape == (2, 3, 3) and np.allclose(pts[0], local)          # relaxed = rest here
    assert np.allclose(pts[1, [0, 2]], local[[0, 2]])                      # the palm and the wrist stay
    R0 = rt.axis_angle_to_matrix(np.cross(A, N) * math.radians(60.0))
    bent = heads[0] + R0 @ (co[1] - heads[0])                              # the finger turned about its knuckle
    assert np.allclose(pts[1, 1], (bent - wrist) @ rest[1, :3, :3])
    assert fg.hand_points(None, rig, "Left", co, W) is None and fg.hand_points(hand_rig, rig, "Right", co, W) is None


def test_grip_track_metrics_read_the_pose():
    L = 40
    basis, pose, racket = _clip(L)
    mapping = {"left_collar": "LeftShoulder", "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm",
               "left_wrist": "LeftHand", "right_wrist": "RightHand"}
    dist = np.linalg.norm(pose[:, IX["LeftHand"], :3, 3] - pose[:, IX["RightHand"], :3, 3], axis=-1)
    w = np.r_[np.zeros(25), np.ones(15)]
    g = rt.grip_track_metrics(RIG, mapping, pose, racket, 20, dist, (PALM_C, PALM_N), weight=w)
    assert g["butt_past_wrist_cm"] == round(racket[1, 3] * 100, 2)        # the butt along the hand bone (its Y)
    shaft = th._frames(pose[:, IX["RightHand"]] @ racket)[:, :3, 1]
    assert list(g["racket_elevation_deg"]) == ["19", "20", "21"]
    assert g["racket_elevation_deg"]["20"] == round(math.degrees(math.asin(shaft[20, 2])), 1)
    d, f = th.palm_on_handle(pose, racket, ARM, PALM_C, PALM_N)
    assert g["left_weight"] == "ik" and g["left_on_frames"] == 15
    assert g["left_palm_to_axis_cm"]["max"] == round(d[25:].max() * 100, 1)
    assert g["left_palm_facing_deg"]["p50"] == round(float(np.median(f[25:])))
    palm = np.einsum("lij,j->li", pose[:, IX["LeftHand"], :3, :3], PALM_C) + pose[:, IX["LeftHand"], :3, 3]
    step = np.linalg.norm(np.diff(palm, axis=0), axis=1)
    assert g["left_palm_max_step_cm"] == round(step.max() * 100, 1) and g["left_palm_max_step_at"] == step.argmax() + 1
    wrists = rt.grip_track_metrics(RIG, mapping, pose, racket, 20, np.full(L, 0.5), (PALM_C, PALM_N))
    assert wrists["left_weight"] == "wrists" and wrists["left_on_frames"] == 0
    assert wrists["left_palm_to_axis_cm"] is None
