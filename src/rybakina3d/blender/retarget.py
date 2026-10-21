"""Retarget GVHMR SMPL motions onto the rigged character and export the moveset (runs INSIDE Blender).

    blender -b --factory-startup --python-exit-code 1 --python retarget.py -- job.json [--verify]

`rybakina3d.assemble` writes the job. Only bpy / mathutils / numpy are used (Blender's bundled Python);
the numpy half (rotations, bone mapping, retargeting, FK, grounding, camera) imports without Blender.

Retargeting is world-rotation transfer with rest-pose alignment, in Blender's Z-up world:
  G_j(t)   world rotation of SMPL joint j (FK of global_orient / body_pose), Y-up -> Z-up
  A_j      minimal rotation taking the rig bone's rest direction (head -> mapped child head) onto the SMPL
           T-pose bone direction. Applied to the limbs only: there SMPL's T-pose and the rig's A-pose
           really differ, while trunk, collars and feet stand the same way in both rest poses and their
           joint offsets differ only by where each skeleton puts its joints (aligning those pitches the
           head ~40 deg down and lifts the toes ~30 deg). Hands reuse the forearm's alignment.
  pose_b(t) = G_j(t) @ A_j @ rest_b      armature-space rotation of the bone b mapped to j
Unmapped bones keep their rest pose relative to their parent; pose-bone basis = rest_rel^-1 @
parent_pose^-1 @ pose. The hips follow joints_world[:, 0] scaled by the rig/SMPL leg length, starting at
the origin, with GVHMR's depth pops clamped; feet GVHMR marks static (static_conf > 0.5) then pin the root
path, so they neither slide nor sink: they sit at their rest height, the rig standing on z=0 at rest.
Between those frames the lowest foot stays between 3 cm under the floor and a per-stroke float cap.
GVHMR's world heading follows the camera: every clip starts with the broadcast camera looking along -Y.
The rig's armature scale (Meshy: 0.01, centimetre bones) is applied on import: the moveset is unit-scale.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Matrix
except ImportError:  # imported outside Blender (unit tests of the numpy part)
    bpy = None
    Matrix = None

# SMPL body skeleton, mirrored from rybakina3d.motion.smpl (this file must not import the package).
SMPL_JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
J = {n: i for i, n in enumerate(SMPL_JOINTS)}

# Child that defines each joint's rest direction (pelvis and spine3 have several children).
SMPL_CHILD = {
    "pelvis": "spine1", "left_hip": "left_knee", "right_hip": "right_knee", "spine1": "spine2",
    "left_knee": "left_ankle", "right_knee": "right_ankle", "spine2": "spine3", "left_ankle": "left_foot",
    "right_ankle": "right_foot", "spine3": "neck", "neck": "head", "left_collar": "left_shoulder",
    "right_collar": "right_shoulder", "left_shoulder": "left_elbow", "right_shoulder": "right_elbow",
    "left_elbow": "left_wrist", "right_elbow": "right_wrist",
}
# Joints whose rest direction is aligned (see module docstring); leaves reuse their parent's alignment.
ALIGN_JOINTS = ("left_hip", "right_hip", "left_knee", "right_knee", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow")
REUSE_PARENT_ALIGNMENT = ("left_wrist", "right_wrist", "head", "left_foot", "right_foot")
# GVHMR static_conf channels for the foot joints: [l_ankle, l_foot, r_ankle, r_foot, l_wrist, r_wrist].
FOOT_CHANNELS = {"left_ankle": 0, "left_foot": 1, "right_ankle": 2, "right_foot": 3}
CONTACT_THRESHOLD = 0.5

# Bone-name vocabulary (Meshy, Mixamo, Rigify DEF-, Unreal, Biped, generic). Matched on the normalized
# part of the name once the side token is removed; the spine is assigned along the hierarchy instead.
SIDE_PARTS = {
    "collar": ("shoulder", "clavicle", "collar", "collarbone"),
    "shoulder": ("arm", "upperarm", "uparm", "humerus"),
    "elbow": ("forearm", "lowerarm", "elbow", "lowarm"),
    "wrist": ("hand", "wrist"),
    "hip": ("upleg", "thigh", "upperleg", "hip", "femur"),
    "knee": ("leg", "lowerleg", "calf", "shin", "knee", "lowleg"),
    "ankle": ("foot", "ankle"),
    "foot": ("toebase", "toe", "toes", "ball"),
}
CENTER_PARTS = {"pelvis": ("hips", "pelvis", "hip"), "neck": ("neck",), "head": ("head",)}
NOISE_TOKENS = {"def", "org", "mch", "bip", "bip01", "bip001", "bn", "jnt", "b", "j", "bone"}

# Racket grip on the right hand (eastern forehand: palm flat on the back bevel, parallel to the strings).
# GVHMR's wrists stay near neutral (no ulnar deviation), so the shaft leaves the fist close to the hand axis.
GRIP_ANGLE_DEG = 25.0   # racket axis, from the hand's long axis towards the thumb
PALM_OFFSET_M = 0.075   # wrist -> palm centre along the hand
PALM_DEPTH_M = 0.015    # palm centre -> handle axis, along the palm normal
GRIP_FROM_BUTT_M = 0.07  # butt cap -> where the palm centre holds the handle

# GVHMR's root path is noisy along the camera ray (single-frame pops of 10-60 m/s on this footage, ~90% of
# the root's jerk). That component gets a running median (pops) and a longer Savitzky-Golay window; faster
# steps are clamped, and planted feet (static_conf) pin the root so they neither slide nor float.
DEPTH_MEDIAN, DEPTH_SMOOTH_WINDOW = 5, 13   # frames
MAX_ROOT_SPEED = 8.0     # m/s
FOOT_LOCK = True         # default when the job's settings do not say (rybakina3d.assemble always does)
FLOOR_TOLERANCE_M = 0.03
# Highest the lowest foot may rise above the floor: a serve's jump; on groundstrokes GVHMR's root height
# is mostly depth drift (it cannot tell a hop from it), so one foot stays within a split-step of the court.
MAX_FLOAT_M = {"serve": 0.5, "overhead": 0.5, "default": 0.05}
SIDE_ELEVATION_DEG = 12.0  # check renders: pitch of the side camera


# --------------------------------------------------------------------------------------------- rotations
def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """(...,3) axis-angle -> (...,3,3) rotation matrices (Rodrigues)."""
    aa = np.asarray(aa, np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-12)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    o = np.zeros_like(x)
    K = np.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(aa.shape[:-1] + (3, 3))
    s, c = np.sin(theta)[..., None], np.cos(theta)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,4) unit quaternions (w, x, y, z) with w >= 0."""
    R = np.asarray(R, np.float64)
    m = {(i, j): R[..., i, j] for i in range(3) for j in range(3)}
    tr = np.stack([1 + m[0, 0] + m[1, 1] + m[2, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2],
                   1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2]], -1)
    case = np.argmax(tr, -1)
    r = 0.5 * np.sqrt(np.maximum(np.take_along_axis(tr, case[..., None], -1)[..., 0], 1e-12))
    f = 0.25 / r
    cands = [
        (r, (m[2, 1] - m[1, 2]) * f, (m[0, 2] - m[2, 0]) * f, (m[1, 0] - m[0, 1]) * f),
        ((m[2, 1] - m[1, 2]) * f, r, (m[0, 1] + m[1, 0]) * f, (m[0, 2] + m[2, 0]) * f),
        ((m[0, 2] - m[2, 0]) * f, (m[0, 1] + m[1, 0]) * f, r, (m[1, 2] + m[2, 1]) * f),
        ((m[1, 0] - m[0, 1]) * f, (m[0, 2] + m[2, 0]) * f, (m[1, 2] + m[2, 1]) * f, r),
    ]
    q = np.zeros(R.shape[:-2] + (4,))
    for i, cand in enumerate(cands):
        sel = case == i
        q[sel] = np.stack(cand, -1)[sel]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Minimal rotation taking direction u onto direction v."""
    u = np.asarray(u, np.float64) / np.linalg.norm(u)
    v = np.asarray(v, np.float64) / np.linalg.norm(v)
    axis = np.cross(u, v)
    s, c = np.linalg.norm(axis), float(np.dot(u, v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(u, [1.0, 0, 0] if abs(u[0]) < 0.9 else [0, 1.0, 0])
        return axis_angle_to_matrix(perp / np.linalg.norm(perp) * math.pi)
    return axis_angle_to_matrix(axis / s * math.atan2(s, c))


def rotation_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def smpl_global_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """(L,3) root + (L,63) local axis-angles -> (L,22,3,3) joint rotations in the (Y-up) world."""
    n = len(global_orient)
    local = np.concatenate([np.asarray(global_orient)[:, None], np.asarray(body_pose).reshape(n, 21, 3)], 1)
    R = axis_angle_to_matrix(local)
    G = np.empty_like(R)
    for j, p in enumerate(SMPL_PARENTS):
        G[:, j] = R[:, j] if p < 0 else G[:, p] @ R[:, j]
    return G


def savgol(x: np.ndarray, window: int, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0: removes jitter but, unlike a box filter, keeps peaks."""
    x = np.asarray(x, np.float64)
    n = len(x)
    window = int(window) | 1
    window = min(window, n if n % 2 else n - 1)
    if window <= order + 1:
        return x.copy()
    h = window // 2
    V = np.vander(np.arange(-h, h + 1, dtype=np.float64), order + 1, increasing=True)
    P = np.linalg.pinv(V)
    flat = x.reshape(n, -1)
    out = np.empty_like(flat)
    out[h:n - h] = np.lib.stride_tricks.sliding_window_view(flat, window, axis=0) @ (V[h] @ P)
    out[:h] = (V[:h] @ P) @ flat[:window]  # edges: the polynomial fitted to the first / last window
    out[n - h:] = (V[h + 1:] @ P) @ flat[n - window:]
    return out.reshape(x.shape)


def smooth_quats(q: np.ndarray, window: int) -> np.ndarray:
    """(L,...,4) quaternion tracks: hemisphere-continuous, then Savitzky-Golay smoothed and renormalized
    (window >= 3). Continuity matters even unsmoothed: Blender interpolates quaternion keys per component."""
    q = np.array(q, np.float64)
    if len(q) < 2:
        return q
    dots = np.sum(q[1:] * q[:-1], -1)
    sign = np.concatenate([np.ones((1,) + dots.shape[1:]), np.cumprod(np.where(dots < 0, -1.0, 1.0), 0)])
    q = q * sign[..., None]
    if window < 3 or len(q) < 3:
        return q
    q = savgol(q, window)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ------------------------------------------------------------------------------------------ bone mapping
def split_side(name: str) -> tuple[str | None, str]:
    """'mixamorig:LeftForeArm' -> ('left', 'forearm');  'upperarm_r' -> ('right', 'upperarm')."""
    base = re.split(r"[:|]", name)[-1]
    tokens = [t for t in re.split(r"[\s_.\-]+", base.lower()) if t]
    side = None
    keep = []
    for t in tokens:
        if t in ("l", "left") and side is None:
            side = "left"
        elif t in ("r", "right") and side is None:
            side = "right"
        elif t not in NOISE_TOKENS:
            keep.append(t)
    part = "".join(keep)
    if side is None:
        m = re.match(r"^(left|right)(.+)$", part) or re.match(r"^(.+?)(left|right)$", part)
        if m:
            side = m.group(1) if m.group(1) in ("left", "right") else m.group(2)
            part = m.group(2) if m.group(1) in ("left", "right") else m.group(1)
        elif re.match(r"^[LR][A-Z]", base):
            side = "left" if base[0] == "L" else "right"
            part = part[1:]
    return side, re.sub(r"\d+$", "", part)


def _depths(parents: list[int]) -> list[int]:
    depth = []
    for p in parents:
        depth.append(0 if p < 0 else depth[p] + 1)
    return depth


def topological_order(parents: list[int]) -> list[int]:
    order, seen = [], set()

    def visit(b: int) -> None:
        if b in seen:
            return
        if parents[b] >= 0:
            visit(parents[b])
        seen.add(b)
        order.append(b)

    for b in range(len(parents)):
        visit(b)
    return order


def _is_ancestor(parents: list[int], a: int, b: int) -> bool:
    p = parents[b]
    while p >= 0:
        if p == a:
            return True
        p = parents[p]
    return False


def map_bones(names: list[str], parents: list[int]) -> tuple[dict[str, str], dict]:
    """SMPL joint -> rig bone by name patterns; the spine chain is split along the hierarchy."""
    depth = _depths(parents)
    order = topological_order(parents)
    parts = {i: split_side(n) for i, n in enumerate(names)}
    found: dict[str, int] = {}

    def pick(cands: list[int]) -> int | None:
        return min(cands, key=lambda i: (depth[i], i)) if cands else None

    for joint, accepted in CENTER_PARTS.items():
        for part in accepted:
            b = pick([i for i, (s, p) in parts.items() if s is None and p == part])
            if b is not None:
                found[joint] = b
                break
    for side in ("left", "right"):
        for joint, accepted in SIDE_PARTS.items():
            for part in accepted:
                b = pick([i for i, (s, p) in parts.items() if s == side and p == part])
                if b is not None:
                    found[f"{side}_{joint}"] = b
                    break

    # spine1..3: the bones strictly between the pelvis and the neck (or the collars' common parent)
    if "pelvis" in found:
        top = found.get("neck")
        top = parents[top] if top is not None else (parents[found["left_collar"]] if "left_collar" in found
                                                    else None)
        chain: list[int] = []
        while top is not None and top >= 0 and top != found["pelvis"]:
            chain.append(top)
            top = parents[top]
        if top == found["pelvis"] and chain:
            chain.reverse()
            picks = {"spine1": chain[0], "spine3": chain[-1]}
            if len(chain) >= 3:
                picks["spine2"] = chain[len(chain) // 2]
            found.update(picks)

    warnings = []
    for j, b in list(found.items()):
        p = SMPL_PARENTS[J[j]]
        while p >= 0 and SMPL_JOINTS[p] not in found:
            p = SMPL_PARENTS[p]
        if p >= 0 and not _is_ancestor(parents, found[SMPL_JOINTS[p]], b):
            warnings.append(f"{j} -> {names[b]} is not below {SMPL_JOINTS[p]} -> "
                            f"{names[found[SMPL_JOINTS[p]]]}; dropped")
            del found[j]
    mapping = {j: names[found[j]] for j in SMPL_JOINTS if j in found}
    used = set(mapping.values())
    info = {
        "unmapped_joints": [j for j in SMPL_JOINTS if j not in mapping],
        "unmapped_bones": [names[i] for i in order if names[i] not in used],
        "warnings": warnings,
    }
    return mapping, info


# ------------------------------------------------------------------------------------------- retargeting
@dataclass
class Rig:
    names: list[str]
    parents: list[int]
    rest: np.ndarray        # (B,4,4) bone rest matrices in armature space (Blender bone.matrix_local)
    obj: np.ndarray         # (4,4) armature object world matrix
    order: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rest = np.asarray(self.rest, np.float64)
        self.obj = np.asarray(self.obj, np.float64)
        self.order = self.order or topological_order(self.parents)
        self.index = {n: i for i, n in enumerate(self.names)}
        lin = self.obj[:3, :3]
        self.obj_scale = float(np.cbrt(abs(np.linalg.det(lin))))
        self.obj_rot = lin / self.obj_scale

    def world(self, p_arm: np.ndarray) -> np.ndarray:
        return p_arm @ self.obj[:3, :3].T + self.obj[:3, 3]

    def to_armature(self, p_world: np.ndarray) -> np.ndarray:
        return (p_world - self.obj[:3, 3]) @ np.linalg.inv(self.obj[:3, :3]).T

    def rest_heads_world(self) -> np.ndarray:
        return self.world(self.rest[:, :3, 3])

    def rest_direction(self, bone: str, child: str) -> np.ndarray:
        h = self.rest_heads_world()
        d = h[self.index[child]] - h[self.index[bone]]
        return d / np.linalg.norm(d)


def rest_alignment(rig: Rig, mapping: dict[str, str], rest_z: np.ndarray,
                   align: tuple[str, ...] = ALIGN_JOINTS) -> tuple[np.ndarray, dict[str, float]]:
    """A_j (22,3,3) in world axes; also the rig-vs-SMPL rest angle of every mapped bone (diagnostic)."""
    A = np.tile(np.eye(3), (22, 1, 1))
    angles: dict[str, float] = {}
    for j in SMPL_JOINTS:
        child = SMPL_CHILD.get(j)
        if j not in mapping or child not in mapping:
            continue
        d = rest_z[J[child]] - rest_z[J[j]]
        R = rotation_between(rig.rest_direction(mapping[j], mapping[child]), d)
        angles[j] = round(rotation_angle_deg(R), 1)
        if j in align:
            A[J[j]] = R
    for j in REUSE_PARENT_ALIGNMENT:
        A[J[j]] = A[SMPL_PARENTS[J[j]]]
    return A, angles


def leg_scale(rig: Rig, mapping: dict[str, str], rest_z: np.ndarray) -> float:
    """Rig hip height over SMPL hip height, both measured above the ankle joints."""
    if not all(k in mapping for k in ("pelvis", "left_ankle", "right_ankle")):
        return 1.0
    h = rig.rest_heads_world()
    rig_h = h[rig.index[mapping["pelvis"]], 2] - np.mean([h[rig.index[mapping[k]], 2]
                                                          for k in ("left_ankle", "right_ankle")])
    smpl_h = rest_z[0, 2] - rest_z[[J["left_ankle"], J["right_ankle"]], 2].mean()
    return float(rig_h / smpl_h)


def forward_kinematics(rig: Rig, basis_rot: np.ndarray, root_bone: int, root_loc: np.ndarray) -> np.ndarray:
    """Armature-space pose matrices (L,B,4,4) from pose-bone basis rotations + the root bone's location."""
    L, B = basis_rot.shape[:2]
    P = np.zeros((L, B, 4, 4))
    inv_rest = np.linalg.inv(rig.rest)
    for b in rig.order:
        p = rig.parents[b]
        basis = np.zeros((L, 4, 4))
        basis[:, :3, :3] = basis_rot[:, b]
        basis[:, 3, 3] = 1
        if b == root_bone:
            basis[:, :3, 3] = root_loc
        rel = inv_rest[p] @ rig.rest[b] if p >= 0 else rig.rest[b]
        P[:, b] = (P[:, p] if p >= 0 else np.eye(4)) @ rel @ basis
    return P


def clamp_speed(path: np.ndarray, max_step: float) -> np.ndarray:
    """Rebuild a (L,3) path from its horizontal steps, each shortened to at most max_step."""
    steps = np.diff(path, axis=0)
    n = np.linalg.norm(steps[:, :2], axis=1, keepdims=True)
    steps[:, :2] *= np.minimum(1.0, max_step / np.maximum(n, 1e-12))
    return np.concatenate([path[:1], path[:1] + np.cumsum(steps, 0)])


def median_filter(x: np.ndarray, window: int) -> np.ndarray:
    """Running median along axis 0, edges padded with the end values: removes isolated pops."""
    h = window // 2
    xp = np.concatenate([np.repeat(x[:1], h, 0), x, np.repeat(x[-1:], h, 0)])
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, window, axis=0), axis=-1)


def smooth_along(path: np.ndarray, ray: np.ndarray, window: int) -> np.ndarray:
    """Smooth only the component of a (L,3) path along the unit vector `ray` (median, then Savitzky-Golay)."""
    along = path @ ray
    if len(path) < DEPTH_MEDIAN:
        return path.copy()
    return path + np.outer(savgol(median_filter(along, DEPTH_MEDIAN), window) - along, ray)


def camera_ray(motion: dict) -> np.ndarray | None:
    """Horizontal unit direction (Z-up) from the broadcast camera to the player, median over the clip."""
    if "joints_camera" not in motion:
        return None
    jw, jc = np.asarray(motion["joints_world"], np.float64), np.asarray(motion["joints_camera"], np.float64)
    d = []
    for t in range(min(len(jw), len(jc))):
        R, tr = camera_from_joints(jw[t, :22], jc[t, :22])
        d.append(Y_UP_TO_Z_UP @ (jw[t, 0] + R.T @ tr))   # pelvis - camera centre
    d = np.median(d, 0) * [1.0, 1.0, 0.0]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def planted_skate(feet_xy: np.ndarray, contact: np.ndarray) -> float | None:
    """Mean horizontal motion (cm/frame) of feet planted at both ends of a step (QA's definition)."""
    both = contact[1:] & contact[:-1]
    if not both.any():
        return None
    return round(float(np.linalg.norm(np.diff(feet_xy, axis=0), axis=-1)[both].mean() * 100), 2)


def plant_feet(root: np.ndarray, feet_rel: np.ndarray, contact: np.ndarray, rest_z: np.ndarray,
               lock: bool = True, per_frame: bool = True, smooth_window: int = 0,
               max_float: float | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Root path (L,3) that keeps planted feet still and at their rest height.

    feet_rel (L,K,3) foot joints relative to the root (rotations only), contact (L,K), rest_z (K,).
    Horizontal: while a foot is planted on two consecutive frames the root moves so that foot does not
    (averaged over the planted feet); otherwise the root keeps the input path's own step.
    Vertical: planted feet sit at their rest height, the correction is interpolated through the flight
    phases between contacts; without contacts (or per_frame off) one offset from the lowest foot. Then
    the lowest foot joint is kept between FLOOR_TOLERANCE_M below and max_float above its rest height.
    Returns (path, vertical correction (L,), method).
    """
    L = len(root)
    path = root.copy()
    if lock:
        steps = np.diff(root[:, :2], axis=0)
        for t in range(1, L):
            k = contact[t] & contact[t - 1]
            if k.any():
                path[t, :2] = path[t - 1, :2] - (feet_rel[t, k, :2] - feet_rel[t - 1, k, :2]).mean(0)
            else:
                path[t, :2] = path[t - 1, :2] + steps[t - 1]
    lift = root[:, None, 2] + feet_rel[..., 2] - rest_z
    planted = contact.any(1)
    if per_frame and planted.any():
        g = -(lift * contact).sum(1)[planted] / contact.sum(1)[planted]
        g = np.interp(np.arange(L), np.flatnonzero(planted), g)
        method = "contacts"
    else:
        g, method = np.full(L, -lift.min(1).mean()), "lowest_foot"
    # Floor: no foot joint deeper than FLOOR_TOLERANCE_M under its rest height (pose noise of the other
    # foot, flight phases). Ceiling: GVHMR's root height drifts with its depth noise (the broadcast camera
    # looks down the court), and away from the sparse static frames that drift would leave the character
    # floating, so the lowest foot may rise at most max_float (a hop; a serve's jump needs more).
    g_lowest = -lift.min(1)  # the correction that puts the lowest foot joint at its rest height
    lo = g_lowest - FLOOR_TOLERANCE_M
    hi = g_lowest + (np.inf if max_float is None else max_float)
    g = np.clip(g, lo, hi)
    if smooth_window and smooth_window >= 3:
        g = np.clip(savgol(g, smooth_window), lo, hi)
    path[:, 2] += g
    return path, g, method


def retarget(rig: Rig, mapping: dict[str, str], motion: dict, smooth_window: int = 0,
             ground_from_contacts: bool = True, foot_lock: bool = FOOT_LOCK,
             align: tuple[str, ...] = ALIGN_JOINTS, max_float: float | None = None) -> dict:
    """One motion.npz (as a dict) -> pose-bone keys for the rig, world positions, grounding info."""
    C = Y_UP_TO_Z_UP
    G = C @ smpl_global_rotations(motion["global_orient"], motion["body_pose"]) @ C.T
    L = len(G)
    rest_z = np.asarray(motion["rest_joints"], np.float64)[:22] @ C.T
    A, angles = rest_alignment(rig, mapping, rest_z, align)
    Ro = rig.obj_rot
    restR = rig.rest[:, :3, :3]
    B = len(rig.names)
    joint_of = {rig.index[b]: J[j] for j, b in mapping.items()}

    pose = np.empty((L, B, 3, 3))
    basis = np.empty((L, B, 3, 3))
    for b in rig.order:
        p = rig.parents[b]
        rel = restR[p].T @ restR[b] if p >= 0 else restR[b]
        parent = pose[:, p] if p >= 0 else np.broadcast_to(np.eye(3), (L, 3, 3))
        if b in joint_of:
            j = joint_of[b]
            pose[:, b] = Ro.T @ G[:, j] @ A[j] @ Ro @ restR[b]
        else:
            pose[:, b] = parent @ rel
        basis[:, b] = rel.T @ np.swapaxes(parent, -1, -2) @ pose[:, b]

    root = rig.index[mapping["pelvis"]]
    scale = leg_scale(rig, mapping, rest_z)
    fps = float(motion.get("fps", 25.0))
    hips = np.asarray(motion["joints_world"], np.float64)[:, 0] @ C.T
    origin = hips[0] * np.array([1.0, 1.0, 0.0])
    gvhmr_path = scale * (hips - origin)
    ray = camera_ray(motion)
    hips_world = gvhmr_path if ray is None else smooth_along(gvhmr_path, ray, DEPTH_SMOOTH_WINDOW)
    hips_world = clamp_speed(hips_world, MAX_ROOT_SPEED / fps)

    def root_loc(target_world: np.ndarray) -> np.ndarray:
        # the bones above the pelvis bone are never animated, so its parent pose is the rest pose
        return (rig.to_armature(target_world) - rig.rest[root, :3, 3]) @ restR[root]

    animated = sorted(joint_of, key=lambda b: rig.order.index(b))
    quats = smooth_quats(matrix_to_quat(basis[:, animated]), smooth_window)
    if smooth_window and smooth_window >= 3:
        hips_world = savgol(hips_world, smooth_window)
    hips_world[:, :2] -= hips_world[0, :2]  # smoothing moves the first frame: start at the origin again
    basis_s = np.broadcast_to(np.eye(3), (L, B, 3, 3)).copy()
    basis_s[:, animated] = quat_to_matrix(quats)

    # feet relative to the hips depend on the rotations only; the root path is then fitted to them
    P = forward_kinematics(rig, basis_s, root, root_loc(hips_world))
    heads = rig.world(P[..., :3, 3])
    rest_heads = rig.rest_heads_world()
    feet = [(rig.index[mapping[k]], ch) for k, ch in FOOT_CHANNELS.items() if k in mapping]
    fb = [b for b, _ in feet]
    static = np.asarray(motion.get("static_conf", np.zeros((L, 6))))[:, [ch for _, ch in feet]]
    contact = static > CONTACT_THRESHOLD
    skate_before = planted_skate(heads[:, fb, :2], contact)
    path, ground, method = plant_feet(hips_world, heads[:, fb] - heads[:, root, None], contact,
                                      rest_heads[fb, 2], lock=foot_lock, per_frame=ground_from_contacts,
                                      smooth_window=smooth_window, max_float=max_float)
    warnings = []
    if method != "contacts":
        warnings.append("no static foot frames: grounded on the lowest foot")
    loc = root_loc(path)
    P = forward_kinematics(rig, basis_s, root, loc)
    heads = rig.world(P[..., :3, 3])

    lowest = (heads[:, fb, 2] - rest_heads[fb, 2]).min(1)
    if lowest.min() < -0.05:
        warnings.append(f"feet {-lowest.min() * 100:.0f} cm below ground at frame {int(lowest.argmin())}")
    capped = int((lowest > max_float - 1e-4).sum()) if max_float is not None else 0
    if capped:
        warnings.append(f"root held down on {capped} frames (float cap {max_float * 100:.0f} cm)")
    pops = np.linalg.norm(np.diff(gvhmr_path[:, :2], axis=0), axis=1) * fps
    if len(pops) and pops.max() > MAX_ROOT_SPEED:
        n = int((pops > MAX_ROOT_SPEED).sum())
        warnings.append(f"GVHMR root pops: {n} steps over {MAX_ROOT_SPEED:g} m/s (max {pops.max():.0f} m/s "
                        f"at frame {int(pops.argmax())}), clamped")

    mapped = [j for j in SMPL_JOINTS if j in mapping]
    return {
        "frames": L,
        "animated": [rig.names[b] for b in animated],
        "quats": quats.astype(np.float32),
        "root_bone": rig.names[root],
        "root_loc": loc.astype(np.float32),
        "pose": P,
        "heads": heads,
        "joint_names": mapped,
        "joint_bones": [mapping[j] for j in mapped],
        "positions": heads[:, [rig.index[mapping[j]] for j in mapped]].astype(np.float32),
        "scale": scale,
        "origin": origin,
        "root_shift": path - gvhmr_path,
        "ground": float(ground.mean()),
        "ground_method": method,
        "contact_frames": int(contact.any(1).sum()),
        "skate_cm": {"before": skate_before, "after": planted_skate(heads[:, fb, :2], contact)},
        "lowest_foot_cm": {k: round(float(np.percentile(lowest, q)) * 100, 1)
                           for k, q in (("p5", 5), ("p50", 50), ("p95", 95))},
        "align_deg": angles,
        "warnings": warnings,
    }


def camera_from_joints(joints_world: np.ndarray, joints_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid (R, t) with joints_camera = R @ joints_world + t (Kabsch); GVHMR's two outputs are rigid."""
    w, c = np.asarray(joints_world, np.float64), np.asarray(joints_camera, np.float64)
    mw, mc = w.mean(0), c.mean(0)
    U, _, Vt = np.linalg.svd((w - mw).T @ (c - mc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mc - R @ mw


def broadcast_camera(motion: dict, frame: int, scale: float, origin: np.ndarray, shift: np.ndarray) -> dict:
    """The clip's camera at `frame` in the retarget world: Blender camera matrix (Z-up) + intrinsics.

    `shift` is how far the retargeted root was moved off GVHMR's path on that frame (foot planting); the
    camera moves with it so it keeps its view of the body."""
    C = Y_UP_TO_Z_UP
    R, t = camera_from_joints(motion["joints_world"][frame, :22], motion["joints_camera"][frame, :22])
    centre = C @ (-R.T @ t)
    rot = C @ np.stack([R[0], -R[1], -R[2]], 1)  # OpenCV (x right, y down, z ahead) -> Blender camera
    M = np.eye(4)
    M[:3, :3] = rot
    M[:3, 3] = scale * (centre - origin) + shift
    K = np.asarray(motion["K"][frame], np.float64)
    return {"matrix": M, "fx": float(K[0, 0]), "cx": float(K[0, 2]), "cy": float(K[1, 2])}


# ============================================================================== Blender side (bpy below)
def _log(msg: str) -> None:
    print(f"[r3d] {msg}", flush=True)


def _reset_scene(fps: float) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 next to the saved moveset
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = scene.render.fps / fps


def _armature(path: str):
    arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise RuntimeError(f"{path}: no armature (is the character rigged?)")
    return arm


def _import_rig(path: str):
    bpy.ops.import_scene.gltf(filepath=path)
    arm = _armature(path)
    # Meshy ships a zero-length preview action and icosphere bone shapes: drop both
    arm.animation_data_clear()
    for a in list(bpy.data.actions):
        bpy.data.actions.remove(a)
    for pb in arm.pose.bones:
        pb.custom_shape = None
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and o.parent is None and not o.modifiers and o.name.startswith("Icosphere"):
            bpy.data.objects.remove(o)
    # Meshy's armature is in centimetres under a 0.01 object scale. Bake that scale into the bones and the
    # skinned mesh: Blender's glTF importer misplaces bone-parented children (the racket) of a scaled
    # armature by bone length / scale, and engines expect a unit-scale skeleton anyway.
    bpy.ops.object.select_all(action="DESELECT")
    for o in (arm, *arm.children_recursive):
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()
    return arm


def rig_from_blender(arm) -> Rig:
    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    parents = [names.index(b.parent.name) if b.parent else -1 for b in bones]
    rest = np.array([np.array(b.matrix_local) for b in bones])
    return Rig(names, parents, rest, np.array(arm.matrix_world))


def _write_action(arm, name: str, res: dict, fps: float):
    from bpy_extras.anim_utils import action_ensure_channelbag_for_slot

    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type="OBJECT", name=arm.name)
    bag = action_ensure_channelbag_for_slot(action, slot)
    frames = np.arange(res["frames"], dtype=np.float64)

    def curve(path: str, index: int, values: np.ndarray, group: str) -> None:
        fc = bag.fcurves.new(path, index=index, group_name=group)
        fc.keyframe_points.add(len(values))
        fc.keyframe_points.foreach_set("co", np.stack([frames, values], 1).ravel())
        fc.keyframe_points.foreach_set("interpolation", np.full(len(values), 1, np.int32))  # LINEAR
        fc.update()

    for k, bone in enumerate(res["animated"]):
        path = f'pose.bones["{bone}"].rotation_quaternion'
        for i in range(4):
            curve(path, i, res["quats"][:, k, i], bone)
    for i in range(3):
        curve(f'pose.bones["{res["root_bone"]}"].location', i, res["root_loc"][:, i], res["root_bone"])
    action.use_fake_user = True
    return action, slot


def _assign(arm, action, slot) -> None:
    ad = arm.animation_data or arm.animation_data_create()
    ad.action = action
    if action is not None:
        ad.action_slot = slot


def _fk_error_mm(arm, rig: Rig, res: dict, frame: int) -> float:
    """Blender's evaluated pose vs our numpy FK (guards the basis maths and the keying)."""
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    err = 0.0
    for b, name in enumerate(rig.names):
        h = np.array(arm.pose.bones[name].head)
        err = max(err, float(np.linalg.norm(h - res["pose"][frame, b, :3, 3])) * rig.obj_scale)
    return err * 1000


def attach_racket(arm, rig: Rig, mapping: dict[str, str]) -> dict:
    """Build the racket and parent it to the right-hand bone in an eastern forehand grip (rest pose)."""
    import racket

    hand, forearm = mapping.get("right_wrist"), mapping.get("right_elbow")
    if hand is None or forearm is None:
        return {"error": "no right hand / forearm bone"}
    heads = rig.rest_heads_world()
    wrist = heads[rig.index[hand]]
    a = rig.obj_rot @ rig.rest[rig.index[hand], :3, 1]  # bone Y = the hand's long axis
    to_body = heads[rig.index[mapping["pelvis"]]] - wrist
    to_body[2] = 0
    n = to_body - np.dot(to_body, a) * a                 # palm normal: towards the body at rest
    n /= np.linalg.norm(n)
    thumb = np.cross(a, n)                               # right hand: long axis x palm normal = thumb
    th = math.radians(GRIP_ANGLE_DEG)
    r = math.cos(th) * a + math.sin(th) * thumb
    palm = wrist + a * PALM_OFFSET_M + n * PALM_DEPTH_M
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = np.cross(r, n), r, n  # racket +Y along the shaft, +Z = string normal
    M[:3, 3] = palm - r * GRIP_FROM_BUTT_M

    obj = racket.build("Racket", arm.users_collection[0])
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = hand
    bpy.context.view_layer.update()
    pb = arm.pose.bones[hand]
    # a bone-parented object hangs off the bone's tail
    tail = np.array(Matrix.Translation((0, pb.length, 0)))
    parent = np.array(arm.matrix_world) @ np.array(pb.matrix) @ tail
    obj.matrix_parent_inverse = Matrix.Identity(4)
    obj.matrix_basis = Matrix(np.linalg.inv(parent) @ M)
    bpy.context.view_layer.update()
    placed = np.array(obj.matrix_world)
    return {"bone": hand, "grip_angle_deg": GRIP_ANGLE_DEG, "length_m": racket.LENGTH,
            "placement_error_mm": float(np.abs(placed[:3, 3] - M[:3, 3]).max() * 1000),
            "butt_world": M[:3, 3].round(4).tolist(), "axis_world": r.round(3).tolist(),
            "string_normal_world": n.round(3).tolist(), "triangles": racket.triangle_count(obj)}


class CheckRenderer:
    """Workbench stills: the clip's own broadcast camera (for overlaying on preview.mp4) + a side view."""

    def __init__(self, width: int, height: int, out_dir: Path):
        scene = bpy.context.scene
        self.scene, self.width, self.height, self.out_dir = scene, width, height, out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        scene.render.engine = "BLENDER_WORKBENCH"
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "TEXTURE"
        shading.show_backface_culling = False
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.resolution_percentage = 100
        self.cam = bpy.data.objects.new("CheckCamera", bpy.data.cameras.new("CheckCamera"))
        scene.collection.objects.link(self.cam)
        scene.camera = self.cam
        me = bpy.data.meshes.new("CheckGround")
        me.from_pydata([(-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0)], [], [(0, 1, 2, 3)])
        mat = bpy.data.materials.new("CheckGround")
        mat.diffuse_color = (0.35, 0.37, 0.4, 1.0)
        me.materials.append(mat)
        self.ground = bpy.data.objects.new("CheckGround", me)
        scene.collection.objects.link(self.ground)

    def render(self, name: str, cam: dict, clip_size: tuple[int, int], pelvis: np.ndarray) -> dict:
        scene, data = self.scene, self.cam.data
        scene.render.resolution_x, scene.render.resolution_y = self.width, self.height
        data.type = "PERSP"
        data.sensor_fit = "HORIZONTAL"
        data.sensor_width = 36.0
        w, h = clip_size
        data.lens = cam["fx"] / w * 36.0
        data.shift_x = -(cam["cx"] - w / 2) / max(w, h)
        data.shift_y = (cam["cy"] - h / 2) / max(w, h)
        data.clip_start, data.clip_end = 0.1, 200.0
        self.cam.matrix_world = Matrix(cam["matrix"])
        self.ground.hide_render = True
        broadcast = self._still(f"{name}_cam.png")

        right = cam["matrix"][:3, 0].copy()  # the broadcast camera's right, levelled
        right[2] = 0
        right /= max(np.linalg.norm(right), 1e-9)
        e = math.radians(SIDE_ELEVATION_DEG)  # looking slightly down so the ground plane shows
        back = right * math.cos(e) + np.array([0, 0, math.sin(e)])
        up = np.array([0, 0, 1.0]) - math.sin(e) * back
        up /= np.linalg.norm(up)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2] = np.cross(up, back), up, back
        M[:3, 3] = [pelvis[0], pelvis[1], 0.9] + back * 8.0
        scene.render.resolution_x = scene.render.resolution_y = self.height
        data.type, data.ortho_scale = "ORTHO", 2.8
        data.shift_x = data.shift_y = 0.0
        self.cam.matrix_world = Matrix(M)
        self.ground.location = (pelvis[0], pelvis[1], 0.0)
        self.ground.hide_render = False
        side = self._still(f"{name}_side.png")
        self.ground.hide_render = True
        return {"broadcast": broadcast, "side": side}

    def _still(self, filename: str) -> str:
        path = self.out_dir / filename
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return str(path)

    def remove(self) -> None:
        for o in (self.cam, self.ground):
            bpy.data.objects.remove(o)


def _export(job: dict) -> None:
    out = job["out"]
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.export_scene.gltf(
        filepath=out["glb"], export_format="GLB", export_animations=True, export_animation_mode="ACTIONS",
        export_force_sampling=True, export_frame_step=1, export_anim_single_armature=True,
        export_reset_pose_bones=True, export_rest_position_armature=True, export_skins=True,
        export_def_bones=False, export_image_format="AUTO", export_yup=True, export_apply=False,
        export_cameras=False, export_lights=False,
    )
    bpy.ops.export_scene.fbx(
        filepath=out["fbx"], use_selection=False, object_types={"ARMATURE", "MESH"}, add_leaf_bones=False,
        bake_anim=True, bake_anim_use_all_bones=True, bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=True, bake_anim_force_startend_keying=True, bake_anim_simplify_factor=0.0,
        path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y",
    )


def build(job: dict) -> dict:
    t0 = time.time()
    settings = job["settings"]
    fps = float(job.get("fps", 25.0))
    _reset_scene(fps)
    arm = _import_rig(job["rig"])
    rig = rig_from_blender(arm)
    mapping, map_info = map_bones(rig.names, rig.parents)
    missing = [j for j in ("pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_shoulder",
                           "right_shoulder", "left_elbow", "right_elbow") if j not in mapping]
    if missing:
        raise RuntimeError(f"rig {job['rig']}: cannot map {missing} (bones: {rig.names})")
    _log(f"rig: {len(rig.names)} bones, mapped {len(mapping)}/22, "
         f"unmapped bones {map_info['unmapped_bones']}")
    racket_info = attach_racket(arm, rig, mapping) if settings.get("racket", True) else None

    out = job["out"]
    joints_dir = Path(out["joints_dir"])
    joints_dir.mkdir(parents=True, exist_ok=True)
    renderer = CheckRenderer(job["render"]["width"], job["render"]["height"], Path(out["checks_raw"]))
    align = tuple(settings.get("align_joints", ALIGN_JOINTS))
    clips, renders, angles = {}, [], {}
    for clip in job["clips"]:
        cid = clip["id"]
        m = dict(np.load(clip["motion"]))
        max_float = MAX_FLOAT_M.get(clip.get("stroke"), MAX_FLOAT_M["default"])
        res = retarget(rig, mapping, m, smooth_window=int(settings.get("smooth_window", 0)),
                       ground_from_contacts=bool(settings.get("ground_from_contacts", True)),
                       foot_lock=bool(settings.get("foot_lock", FOOT_LOCK)), align=align, max_float=max_float)
        angles = res["align_deg"]
        clip_fps = float(m.get("fps", fps))
        if abs(clip_fps - fps) > 1e-3:
            res["warnings"].append(f"motion fps {clip_fps} differs from the export rate {fps}")
        if clip.get("n_frames") and clip["n_frames"] != res["frames"]:
            res["warnings"].append(f"{res['frames']} motion frames vs {clip['n_frames']} clip frames")
        action, slot = _write_action(arm, cid, res, fps)
        _assign(arm, action, slot)
        contact = int(np.clip(clip.get("contact_index", res["frames"] // 2), 0, res["frames"] - 1))
        fk_mm = _fk_error_mm(arm, rig, res, contact)
        for f in clip.get("check_frames", []):
            f = int(np.clip(f, 0, res["frames"] - 1))
            bpy.context.scene.frame_set(f)
            cam = broadcast_camera(m, f, res["scale"], res["origin"], res["root_shift"][f])
            pelvis = res["heads"][f, rig.index[res["root_bone"]]]
            files = renderer.render(f"{cid}_{f:03d}", cam, tuple(clip["clip_size"]), pelvis)
            renders.append({"clip": cid, "frame": f, **files})
        cam_c = broadcast_camera(m, contact, res["scale"], res["origin"], res["root_shift"][contact])
        _assign(arm, None, None)
        track = arm.animation_data.nla_tracks.new()
        track.name = cid
        strip = track.strips.new(cid, 0, action)
        strip.action_slot = slot
        track.mute = True  # exporters assign each action themselves; muted tracks keep the rest pose clean

        np.savez_compressed(joints_dir / f"{cid}.npz", names=np.array(res["joint_bones"]),
                            smpl_joints=np.array(res["joint_names"]), positions=res["positions"],
                            fps=np.float64(fps))
        cam_pos = cam_c["matrix"][:3, 3]
        clips[cid] = {
            "frames": res["frames"], "fps": fps, "scale": round(res["scale"], 4),
            "ground_offset_m": round(res["ground"], 4), "ground_method": res["ground_method"],
            "contact_frames": res["contact_frames"], "contact_index": contact,
            "skate_cm_per_frame": res["skate_cm"], "lowest_foot_cm": res["lowest_foot_cm"],
            "max_float_m": max_float,
            "start_xy": res["origin"][:2].round(3).tolist(), "fk_error_mm": round(fk_mm, 3),
            "camera_at_contact": {"position_zup": cam_pos.round(3).tolist(),
                                  "matrix_zup": cam_c["matrix"].round(5).tolist(),
                                  "hfov_deg": round(math.degrees(2 * math.atan(clip["clip_size"][0] / 2
                                                                                / cam_c["fx"])), 2)},
            "warnings": res["warnings"],
        }
        _log(f"{cid}: {res['frames']} frames, scale {res['scale']:.3f}, ground {res['ground'] * 100:+.1f} cm "
             f"({res['ground_method']}), fk {fk_mm:.2f} mm" + (f", {len(res['warnings'])} warnings"
                                                               if res["warnings"] else ""))
    renderer.remove()
    bpy.context.scene.frame_set(0)
    _export(job)
    if out.get("blend"):
        bpy.ops.wm.save_as_mainfile(filepath=out["blend"], compress=True)
    report = {
        "rig": {"path": job["rig"], "bones": rig.names,
                "hierarchy": {n: (rig.names[p] if p >= 0 else None) for n, p in zip(rig.names, rig.parents,
                                                                                   strict=True)},
                "armature_scale": round(rig.obj_scale, 6)},
        "mapping": mapping, **map_info,
        "alignment": {"aligned_joints": list(align), "rest_angle_deg": angles},
        "racket": racket_info, "clips": clips, "renders": renders,
        "outputs": {k: out[k] for k in ("glb", "fbx")},
        "blender": bpy.app.version_string, "seconds": round(time.time() - t0, 1),
    }
    Path(out["report"]).write_text(json.dumps(report, indent=2))
    _log(f"exported {len(clips)} animations -> {out['glb']} ({time.time() - t0:.0f}s)")
    return report


def _unit_frame(M) -> np.ndarray:
    """A 4x4 world matrix without its scale (rotation columns normalized)."""
    F = np.array(M, np.float64)
    F[:3, :3] /= np.linalg.norm(F[:3, :3], axis=0)
    return F


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    c = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def _rest_pose(arm) -> None:
    ad = arm.animation_data
    if ad is not None:
        ad.action = None
        for track in ad.nla_tracks:
            track.mute = True
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def _racket_rest_error(racket_obj, placed: dict) -> dict:
    """The re-imported racket at the rest pose vs where the build put it (butt mm, shaft / normal deg)."""
    F = _unit_frame(racket_obj.matrix_world)
    return {"butt_mm": round(float(np.linalg.norm(F[:3, 3] - placed["butt_world"])) * 1000, 1),
            "axis_deg": round(_angle_deg(F[:3, 1], np.array(placed["axis_world"])), 2),
            "normal_deg": round(_angle_deg(F[:3, 2], np.array(placed["string_normal_world"])), 2)}


def verify(job: dict) -> dict:
    """Fresh Blender: re-import the GLB (and the FBX). Checks the animation names and lengths, that every
    clip replays the joint tracks `qa` reads (anim/joints), and that the racket sits in the right hand where
    the build put it and stays there."""
    out = job["out"]
    clips = {c["id"]: c for c in job["clips"]}
    expected = sorted(clips)
    report = json.loads(Path(out["report"]).read_text())
    hand, placed = report["mapping"].get("right_wrist"), report.get("racket") or {}
    _reset_scene(float(job.get("fps", 25.0)))  # the imported keys then land on whole frames
    bpy.ops.import_scene.gltf(filepath=out["glb"])
    arm = _armature(out["glb"])
    racket_obj = next((o for o in bpy.data.objects if o.name.startswith("Racket")), None)
    actions = sorted(a.name for a in bpy.data.actions)
    res: dict = {"glb_actions": actions, "glb_names_match": actions == expected, "glb_racket_parent": None,
                 "glb_racket_rest": None, "glb_clips": {}}
    if racket_obj is not None:
        parent = racket_obj.parent_bone or (racket_obj.parent.name if racket_obj.parent else None)
        res["glb_racket_parent"] = f"{racket_obj.parent_type}:{parent}"
        if "butt_world" in placed:
            _rest_pose(arm)
            res["glb_racket_rest"] = _racket_rest_error(racket_obj, placed)
    W = np.array(arm.matrix_world)
    for cid in expected:
        action = bpy.data.actions.get(cid)
        if action is None:
            continue
        ad = arm.animation_data or arm.animation_data_create()
        ad.action = action
        if action.slots:
            ad.action_slot = action.slots[0]
        tracks = np.load(Path(out["joints_dir"]) / f"{cid}.npz")
        names = [str(n) for n in tracks["names"]]
        n = len(tracks["positions"])
        contact = int(np.clip(clips[cid].get("contact_index", n // 2), 0, n - 1))
        joint_err, rel = 0.0, []
        for f in sorted({0, contact, n - 1}):
            bpy.context.scene.frame_set(f)
            bpy.context.view_layer.update()
            heads = np.array([(W @ np.append(np.array(arm.pose.bones[b].head), 1.0))[:3] for b in names])
            joint_err = max(joint_err, float(np.linalg.norm(heads - tracks["positions"][f], axis=1).max()))
            if racket_obj is not None and hand in arm.pose.bones:
                H = _unit_frame(W @ np.array(arm.pose.bones[hand].matrix))
                rel.append(np.linalg.inv(H) @ _unit_frame(racket_obj.matrix_world))
        clip_res = {"frames": int(round(action.frame_range[1] - action.frame_range[0])) + 1,
                    "joint_error_mm": round(joint_err * 1000, 3)}
        if rel:  # the racket's pose in the hand frame must not change while the clip plays
            clip_res["racket_drift_mm"] = round(max(float(np.linalg.norm(r[:3, 3] - rel[0][:3, 3]))
                                                    for r in rel) * 1000, 3)
            clip_res["racket_drift_deg"] = round(max(_angle_deg(r[:3, 1], rel[0][:3, 1]) for r in rel), 3)
        res["glb_clips"][cid] = clip_res
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=out["fbx"])
        res["fbx_actions"] = sorted(a.name for a in bpy.data.actions)
        racket_obj = next((o for o in bpy.data.objects if o.name.startswith("Racket")), None)
        if racket_obj is not None and "butt_world" in placed:
            _rest_pose(_armature(out["fbx"]))
            res["fbx_racket_rest"] = _racket_rest_error(racket_obj, placed)
    except Exception as e:  # noqa: BLE001 - the FBX check is informative only
        res["fbx_error"] = str(e)
    return res


def main(argv: list[str]) -> None:
    job = json.loads(Path(argv[0]).read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if "--verify" in argv:
        res = verify(job)
        Path(job["out"]["verify"]).write_text(json.dumps(res, indent=2))
        clips = res["glb_clips"].values()
        _log(f"verify: GLB {len(res['glb_actions'])} actions (names match: {res['glb_names_match']}), "
             f"joint replay <= {max((c['joint_error_mm'] for c in clips), default=0):.3f} mm, "
             f"racket at rest {res['glb_racket_rest']}, FBX {len(res.get('fbx_actions', []))} actions, "
             f"FBX racket {res.get('fbx_racket_rest') or res.get('fbx_error')}")
    else:
        build(job)


if __name__ == "__main__" and bpy is not None:
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
