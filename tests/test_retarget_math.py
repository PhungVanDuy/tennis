"""Unit tests for the pure-numpy half of blender/retarget.py (rotations, bone mapping, retargeting)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from rybakina3d.blender import retarget as rt
from rybakina3d.motion.smpl import JOINTS, PARENTS, Y_UP_TO_Z_UP, global_rotations

# SMPL rest joints (pelvis-relative, Y-up) of one GVHMR clip of this run
REST = np.array([
    [0, 0, 0], [0.065, -0.092, -0.014], [-0.064, -0.09, -0.01], [-0.002, 0.105, -0.015],
    [0.097, -0.47, -0.023], [-0.099, -0.476, -0.021], [0.002, 0.238, -0.009], [0.082, -0.867, -0.064],
    [-0.081, -0.875, -0.064], [0.002, 0.287, 0.015], [0.108, -0.924, 0.052], [-0.109, -0.922, 0.054],
    [-0.001, 0.492, -0.03], [0.074, 0.404, -0.02], [-0.073, 0.402, -0.024], [0.004, 0.555, 0.02],
    [0.155, 0.427, -0.027], [-0.159, 0.427, -0.029], [0.402, 0.381, -0.051], [-0.399, 0.38, -0.048],
    [0.64, 0.384, -0.053], [-0.643, 0.384, -0.054],
])
MESHY = {
    "pelvis": "Hips", "spine1": "Spine02", "spine2": "Spine01", "spine3": "Spine", "neck": "neck",
    "head": "Head",
    "left_collar": "LeftShoulder", "left_shoulder": "LeftArm", "left_elbow": "LeftForeArm",
    "left_wrist": "LeftHand", "right_collar": "RightShoulder", "right_shoulder": "RightArm",
    "right_elbow": "RightForeArm", "right_wrist": "RightHand", "left_hip": "LeftUpLeg",
    "left_knee": "LeftLeg",
    "left_ankle": "LeftFoot", "left_foot": "LeftToeBase", "right_hip": "RightUpLeg", "right_knee": "RightLeg",
    "right_ankle": "RightFoot", "right_foot": "RightToeBase",
}


def test_rotation_helpers_match_scipy():
    rng = np.random.default_rng(0)
    aa = rng.normal(size=(50, 3))
    R = Rotation.from_rotvec(aa).as_matrix()
    assert np.allclose(rt.axis_angle_to_matrix(aa), R, atol=1e-10)
    q = rt.matrix_to_quat(R)
    q_ref = Rotation.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    assert np.allclose(np.abs(np.sum(q * q_ref, -1)), 1, atol=1e-9)
    assert np.allclose(rt.quat_to_matrix(q), R, atol=1e-10)
    pairs = [(rng.normal(size=3), rng.normal(size=3)), ([0, 0, 1.0], [0, 0, -1.0]), ([1.0, 0, 0], [1, 0, 0])]
    for u, v in pairs:
        M = rt.rotation_between(u, v)
        assert np.allclose(M @ (np.array(u) / np.linalg.norm(u)), np.array(v) / np.linalg.norm(v), atol=1e-9)
        assert np.allclose(M @ M.T, np.eye(3), atol=1e-9)


def test_smpl_skeleton_and_fk_match_the_package():
    assert rt.SMPL_JOINTS == JOINTS and rt.SMPL_PARENTS == PARENTS
    assert np.allclose(rt.Y_UP_TO_Z_UP, Y_UP_TO_Z_UP)
    rng = np.random.default_rng(1)
    go, bp = rng.normal(size=(4, 3)), rng.normal(scale=0.4, size=(4, 63))
    assert np.allclose(rt.smpl_global_rotations(go, bp), global_rotations(go, bp), atol=1e-10)


def test_savgol_keeps_polynomials_and_smooth_quats_handles_sign_flips():
    t = np.arange(20, dtype=float)
    x = np.stack([0.3 * t**2 - t, np.full(20, 2.0)], 1)
    assert np.allclose(rt.savgol(x, 5), x)
    noisy = x[:, 0] + np.random.default_rng(2).normal(scale=1.0, size=20)
    assert np.std(rt.savgol(noisy, 5) - x[:, 0]) < np.std(noisy - x[:, 0])
    q = rt.matrix_to_quat(Rotation.from_rotvec(np.outer(np.linspace(0, 1, 12), [0, 0, 1])).as_matrix())
    q[::2] *= -1
    s = rt.smooth_quats(q, 5)
    assert np.allclose(np.abs(np.sum(s * q, -1)), 1, atol=1e-3)


def test_map_bones_meshy_mixamo_unreal():
    meshy = ["Hips", "Spine02", "Spine01", "Spine", "neck", "Head", "head_end", "headfront", "LeftShoulder",
             "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
             "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot",
             "RightToeBase"]
    parents = [-1, 0, 1, 2, 3, 4, 5, 5, 3, 8, 9, 10, 3, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22]
    mapping, info = rt.map_bones(meshy, parents)
    assert mapping == MESHY
    assert info["unmapped_bones"] == ["head_end", "headfront"] and not info["unmapped_joints"]

    mixamo = [f"mixamorig:{n}" for n in [
        "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "HeadTop_End", "LeftShoulder", "LeftArm",
        "LeftForeArm", "LeftHand", "LeftHandIndex1", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "LeftToe_End"]]
    mparents = [-1, 0, 1, 2, 3, 4, 5, 3, 7, 8, 9, 10, 0, 12, 13, 14, 15]
    mapping, info = rt.map_bones(mixamo, mparents)
    assert [mapping[j] for j in ("spine1", "spine2", "spine3")] == [f"mixamorig:{n}" for n in
                                                                    ("Spine", "Spine1", "Spine2")]
    assert mapping["left_foot"] == "mixamorig:LeftToeBase" and mapping["left_wrist"] == "mixamorig:LeftHand"
    assert set(info["unmapped_bones"]) == {"mixamorig:HeadTop_End", "mixamorig:LeftHandIndex1",
                                           "mixamorig:LeftToe_End"}

    ue = ["root", "pelvis", "spine_01", "spine_02", "spine_03", "neck_01", "head", "clavicle_r", "upperarm_r",
          "upperarm_twist_01_r", "lowerarm_r", "hand_r", "thigh_r", "calf_r", "foot_r", "ball_r"]
    uparents = [-1, 0, 1, 2, 3, 4, 5, 4, 7, 8, 8, 10, 1, 12, 13, 14]
    mapping, info = rt.map_bones(ue, uparents)
    assert mapping["pelvis"] == "pelvis" and mapping["neck"] == "neck_01" and mapping["spine2"] == "spine_02"
    assert [mapping[f"right_{j}"] for j in ("collar", "shoulder", "elbow", "wrist", "hip", "knee", "ankle",
                                             "foot")] == ["clavicle_r", "upperarm_r", "lowerarm_r", "hand_r",
                                                          "thigh_r", "calf_r", "foot_r", "ball_r"]
    assert "root" in info["unmapped_bones"] and "upperarm_twist_01_r" in info["unmapped_bones"]


def _synthetic_rig(arm_drop_deg: float = 50.0, seed: int = 3) -> rt.Rig:
    """Meshy-named rig built from the SMPL rest joints with the arms lowered into an A-pose, random bone
    rolls, and an armature object that is scaled (cm) and turned (its world rest still faces -Y)."""
    C = Y_UP_TO_Z_UP
    heads = REST @ C.T + [0, 0, 0.95]
    for side, sign in (("left", 1), ("right", -1)):
        s = heads[rt.J[f"{side}_shoulder"]].copy()
        R = Rotation.from_rotvec([0, sign * np.radians(arm_drop_deg), 0]).as_matrix()
        for j in (f"{side}_elbow", f"{side}_wrist"):
            heads[rt.J[j]] = s + R @ (heads[rt.J[j]] - s)
    names = [MESHY[j] for j in rt.SMPL_JOINTS] + ["head_end"]
    parents = list(rt.SMPL_PARENTS) + [rt.J["head"]]
    heads = np.vstack([heads, heads[rt.J["head"]] + [0, 0, 0.2]])
    obj = np.eye(4)
    obj[:3, :3] = 0.01 * Rotation.from_rotvec([0, 0, 0.7]).as_matrix()
    obj[:3, 3] = [0.3, -0.2, 0.0]
    rolls = Rotation.random(len(names), random_state=seed).as_matrix()
    rest = np.tile(np.eye(4), (len(names), 1, 1))
    rest[:, :3, :3] = (obj[:3, :3] / 0.01).T @ rolls
    rest[:, :3, 3] = (heads - obj[:3, 3]) @ np.linalg.inv(obj[:3, :3]).T
    return rt.Rig(names, parents, rest, obj)


def _motion(go: np.ndarray, bp: np.ndarray, static: np.ndarray | None = None) -> dict:
    L = len(go)
    G = global_rotations(go, bp)
    jw = np.zeros((L, 24, 3))
    jw[:, 0] = [0.5, 0.9, 2.0]
    for j, p in enumerate(PARENTS):
        if p >= 0:
            jw[:, j] = jw[:, p] + G[:, p] @ (REST[j] - REST[p])
    return {"global_orient": go, "body_pose": bp, "joints_world": jw,
            "rest_joints": np.vstack([REST, np.zeros((2, 3))]),
            "static_conf": static if static is not None else np.zeros((L, 6)), "fps": 25.0}


def test_zero_pose_gives_a_t_pose_and_upright_trunk():
    rig = _synthetic_rig()
    mapping, _ = rt.map_bones(rig.names, rig.parents)
    res = rt.retarget(rig, mapping, _motion(np.zeros((3, 3)), np.zeros((3, 63))))
    h = res["heads"][0]
    i = rig.index
    for side in ("Left", "Right"):
        arm = h[i[f"{side}ForeArm"]] - h[i[f"{side}Arm"]]
        assert abs(arm[2]) / np.linalg.norm(arm) < 0.2          # horizontal again (SMPL's T-pose arm)
    rest = rig.rest_heads_world()
    neck = h[i["Head"]] - h[i["neck"]]
    assert np.allclose(neck / np.linalg.norm(neck), (rest[i["Head"]] - rest[i["neck"]]) /
                       np.linalg.norm(rest[i["Head"]] - rest[i["neck"]]), atol=1e-9)   # trunk not re-aligned
    assert abs(res["scale"] - 1.0) < 1e-6                         # the rig was built from these joints


def test_world_rotation_transfer_reproduces_smpl_limb_directions():
    rng = np.random.default_rng(4)
    L = 6
    go, bp = rng.normal(scale=0.6, size=(L, 3)), rng.normal(scale=0.5, size=(L, 63))
    rig = _synthetic_rig()
    mapping, _ = rt.map_bones(rig.names, rig.parents)
    res = rt.retarget(rig, mapping, _motion(go, bp))
    C = Y_UP_TO_Z_UP
    G = C @ global_rotations(go, bp) @ C.T
    rest_z = REST @ C.T
    i = rig.index
    for j in rt.ALIGN_JOINTS:
        c = rt.SMPL_CHILD[j]
        want = G[:, rt.J[j]] @ (rest_z[rt.J[c]] - rest_z[rt.J[j]])
        got = res["heads"][:, i[mapping[c]]] - res["heads"][:, i[mapping[j]]]
        cos = np.sum(want * got, -1) / np.linalg.norm(want, axis=-1) / np.linalg.norm(got, axis=-1)
        assert np.allclose(cos, 1, atol=1e-9), j
    # the pelvis follows joints_world[:, 0], starting above the origin
    assert np.allclose(res["heads"][0, i["Hips"], :2], 0, atol=1e-9)


def test_grounding_puts_static_feet_at_rest_height():
    L = 5
    static = np.zeros((L, 6))
    static[:, 0] = 0.9                                            # left ankle planted
    rig = _synthetic_rig()
    mapping, _ = rt.map_bones(rig.names, rig.parents)
    m = _motion(np.zeros((L, 3)), np.zeros((L, 63)), static)
    m["joints_world"][:, :, 1] += 0.13                            # the motion floats 8 cm above the rig
    res = rt.retarget(rig, mapping, m, smooth_window=5)
    b = rig.index["LeftFoot"]
    assert np.allclose(res["heads"][:, b, 2], rig.rest_heads_world()[b, 2], atol=1e-6)
    assert res["ground_method"] == "contacts" and abs(res["ground"] + 0.08) < 1e-6  # 0.90 + 0.13 vs 0.95


def test_camera_from_joints_recovers_the_rigid_transform():
    rng = np.random.default_rng(5)
    jw = rng.normal(size=(22, 3))
    R = Rotation.random(random_state=6).as_matrix()
    t = np.array([0.2, -1.0, 15.0])
    R2, t2 = rt.camera_from_joints(jw, jw @ R.T + t)
    assert np.allclose(R2, R, atol=1e-9) and np.allclose(t2, t, atol=1e-9)


def test_clamp_speed_removes_pops_and_keeps_slow_steps():
    path = np.zeros((6, 3))
    path[:, 0] = [0, 0.1, 0.2, 2.2, 2.3, 2.4]                     # one 2 m pop
    path[:, 2] = np.linspace(1, 1.5, 6)
    out = rt.clamp_speed(path, 0.3)
    assert np.allclose(np.diff(out[:, 0]), [0.1, 0.1, 0.3, 0.1, 0.1]) and np.allclose(out[:, 2], path[:, 2])


def test_plant_feet_pins_planted_feet_and_keeps_flight_steps():
    L = 8
    rng = np.random.default_rng(7)
    root = np.zeros((L, 3))
    root[:, 0] = np.arange(L) * 0.05 + rng.normal(scale=0.05, size=L)  # noisy depth
    root[:, 2] = 0.9 + rng.normal(scale=0.03, size=L)
    feet = np.zeros((L, 2, 3))
    feet[:, 0] = [0.1, 0, -0.85]
    feet[:, 1] = [-0.1, 0, -0.85]
    feet[:, 1, 0] -= np.arange(L) * 0.04                              # foot 1 swings relative to the root
    contact = np.zeros((L, 2), bool)
    contact[:5, 0] = True                                             # foot 0 planted, then a flight phase
    rest_z = np.array([0.08, 0.08])
    path, g, method = rt.plant_feet(root, feet, contact, rest_z)
    world0 = path[:, None] + feet
    assert method == "contacts"
    assert np.allclose(world0[:5, 0, :2], world0[0, 0, :2])          # planted foot does not slide
    assert np.allclose(world0[:5, 0, 2], 0.08)                        # ... and sits at its rest height
    assert np.allclose(np.diff(path[4:, :2], axis=0), np.diff(root[4:, :2], axis=0))  # flight: own steps


def test_plant_feet_floor_lifts_a_sinking_swing_foot():
    L = 5
    root = np.tile([0.0, 0.0, 0.9], (L, 1))
    feet = np.tile([[0.1, 0, -0.82], [-0.1, 0, -0.82]], (L, 1, 1))
    feet[2, 1, 2] = -1.0                                              # swing foot dips 18 cm on frame 2
    contact = np.zeros((L, 2), bool)
    contact[:, 0] = True
    path, g, _ = rt.plant_feet(root, feet, contact, np.array([0.08, 0.08]))
    z = path[:, None, 2] + feet[..., 2]
    assert (z - 0.08 >= -rt.FLOOR_TOLERANCE_M - 1e-9).all()
    assert np.allclose(z[[0, 1, 3, 4], 0], 0.08)                     # planted foot untouched elsewhere


def test_smooth_quats_is_hemisphere_continuous_even_unsmoothed():
    q = rt.matrix_to_quat(Rotation.from_rotvec(np.outer(np.linspace(2.8, 3.6, 9), [0, 1.0, 0])).as_matrix())
    assert (np.sum(q[1:] * q[:-1], -1) < 0).any()                    # w >= 0 canonical form flips sign
    s = rt.smooth_quats(q, 0)
    assert (np.sum(s[1:] * s[:-1], -1) > 0).all() and np.allclose(np.abs(np.sum(s * q, -1)), 1)


def test_plant_feet_caps_a_floating_root_between_sparse_contacts():
    L = 30
    root = np.tile([0.0, 0.0, 0.9], (L, 1))
    root[10:, 2] += np.linspace(0, 0.4, L - 10)                       # GVHMR's root drifts 40 cm up
    feet = np.tile([[0.1, 0, -0.82], [-0.1, 0, -0.82]], (L, 1, 1))
    contact = np.zeros((L, 2), bool)
    contact[:3, 0] = True                                             # static only at the start
    rest_z = np.array([0.08, 0.08])
    _, g_free, _ = rt.plant_feet(root, feet, contact, rest_z)
    path, g, _ = rt.plant_feet(root, feet, contact, rest_z, smooth_window=5, max_float=0.12)
    low_free = (root[:, 2] + g_free)[:, None] + feet[..., 2] - rest_z
    low = (path[:, 2])[:, None] + feet[..., 2] - rest_z
    assert low_free.min(1).max() > 0.35                               # without the cap she floats
    assert low.min(1).max() <= 0.12 + 1e-9 and low.min(1).min() >= -rt.FLOOR_TOLERANCE_M - 1e-9
    assert np.allclose(low[:3, 0], 0, atol=1e-9)                       # the planted frames are untouched


def test_smooth_along_removes_depth_pops_and_keeps_lateral_motion():
    L = 30
    path = np.zeros((L, 3))
    path[:, 0] = np.linspace(0, 1.5, L)                               # lateral run, across the ray
    path[:, 1] = -0.02 * np.arange(L)                                 # slow drift along the ray
    noisy = path.copy()
    noisy[12, 1] += 1.2                                               # a 1.2 m single-frame depth pop
    ray = np.array([0.0, -1.0, 0.0])
    out = rt.smooth_along(noisy, ray, 13)
    assert np.allclose(out[:, 0], path[:, 0]) and np.allclose(out[:, 2], 0)
    assert np.abs(out[:, 1] - path[:, 1]).max() < 0.015               # the pop is gone (1 cm of 1.2 m left)
    assert np.allclose(rt.smooth_along(path, ray, 13), path)          # a smooth path passes unchanged


def test_camera_ray_points_from_the_camera_to_the_player():
    rng = np.random.default_rng(8)
    L = 6
    jw = rng.normal(scale=0.3, size=(L, 24, 3)) + [0, 0.9, 0]
    centre = np.array([0.0, 6.0, -20.0])                              # Y-up: behind and above the player
    R = Rotation.from_rotvec([0.25, 0, 0]).as_matrix()                # looking down the court (+Z), pitched
    jc = (jw - centre) @ R.T
    ray = rt.camera_ray({"joints_world": jw, "joints_camera": jc})
    want = Y_UP_TO_Z_UP @ (np.median(jw[:, 0], 0) - centre)
    want[2] = 0
    assert np.allclose(ray, want / np.linalg.norm(want), atol=1e-6) and abs(ray[1] + 1) < 0.01
