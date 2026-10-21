"""SMPL / SMPL-X body skeleton (the 22 body joints GVHMR recovers), joint indices and rotation helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

JOINTS = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

# Index of each joint in JOINTS (motion.npz joints_world / joints_camera carry the two SMPL hand joints after
# these 22, so the same indices apply to their 24 joints).
(PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE, SPINE2, L_ANKLE, R_ANKLE, SPINE3, L_FOOT, R_FOOT, NECK,
 L_COLLAR, R_COLLAR, HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST) = range(len(JOINTS))

# GVHMR predicts per-frame "static" probabilities for these joints, in this channel order.
STATIC_JOINTS = ["left_ankle", "left_foot", "right_ankle", "right_foot", "left_wrist", "right_wrist"]

# SMPL world frame is Y-up; Blender / the retargeter work Z-up with the character facing -Y.
Y_UP_TO_Z_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)


def global_rotations(global_orient: np.ndarray, body_pose: np.ndarray) -> np.ndarray:
    """(L,3) root + (L,63) local axis-angles → (L,22,3,3) joint rotations in the world frame."""
    n = global_orient.shape[0]
    local = np.concatenate([global_orient[:, None], body_pose.reshape(n, 21, 3)], axis=1)
    R_local = Rotation.from_rotvec(local.reshape(-1, 3)).as_matrix().reshape(n, 22, 3, 3)
    G = np.empty_like(R_local)
    for j, p in enumerate(PARENTS):
        G[:, j] = R_local[:, j] if p < 0 else G[:, p] @ R_local[:, j]
    return G
