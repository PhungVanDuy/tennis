"""Unit tests for the shared helpers and the pure parts of the footage → clip stages."""

from __future__ import annotations

import re

import numpy as np

from rybakina3d.index.labeler import _chunks, _fix_time
from rybakina3d.motion import smpl
from rybakina3d.motion.smpl import JOINTS, PARENTS, global_rotations
from rybakina3d.selection import size_score
from rybakina3d.tracking import _fill_and_smooth, iou
from rybakina3d.util import read_npz


def test_iou():
    a = np.array([0, 0, 10, 10.0])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30.0])) == 0.0
    assert abs(iou(a, np.array([5, 0, 15, 10.0])) - 50 / 150) < 1e-9


def test_fill_and_smooth_interpolates_gaps():
    boxes = {0: np.array([0, 0, 10, 10.0]), 4: np.array([4, 0, 14, 10.0])}
    out, valid = _fill_and_smooth(boxes, 5, win=1)
    assert valid.tolist() == [True, False, False, False, True]
    assert np.allclose(out[:, 0], [0, 1, 2, 3, 4])


def test_chunks_cover_all_shots_in_order():
    shots = [{"id": i, "start": i * 10.0, "end": i * 10.0 + 10} for i in range(10)]
    chunks = _chunks(shots, 35)
    assert [s["id"] for c in chunks for s in c] == list(range(10))
    assert all(c[-1]["end"] - c[0]["start"] >= 35 for c in chunks[:-1])


def test_fix_time_maps_window_relative_times():
    shot = {"start": 300.0, "end": 305.0}
    assert _fix_time(302.0, shot, 240.0) == 302.0
    assert _fix_time(62.0, shot, 240.0) == 302.0
    assert _fix_time(10.0, shot, 240.0) is None


def test_global_rotations_identity_and_chain():
    n = 2
    go = np.zeros((n, 3))
    bp = np.zeros((n, 63))
    G = global_rotations(go, bp)
    assert np.allclose(G, np.eye(3))
    # rotating the pelvis rotates every descendant by the same amount
    go[:, 1] = np.pi / 2
    G = global_rotations(go, bp)
    assert np.allclose(G[:, 21], G[:, 0])
    assert len(PARENTS) == 22


def test_joint_index_constants_name_their_joints():
    consts = {k: v for k, v in vars(smpl).items() if re.fullmatch(r"[LR]?_?[A-Z]+\d?", k) and isinstance(v, int)}
    assert len(consts) == len(JOINTS)
    for name, i in consts.items():
        assert JOINTS[i] == re.sub(r"^R_", "right_", re.sub(r"^L_", "left_", name)).lower()


def test_read_npz_loads_every_array(tmp_path):
    np.savez(tmp_path / "m.npz", x=np.arange(3), stroke="serve")
    m = read_npz(tmp_path / "m.npz")
    assert m["x"].tolist() == [0, 1, 2] and str(m["stroke"]) == "serve"


def test_size_score_is_relative_to_a_near_court_player_and_capped():
    assert size_score(200.0) == 0.5 and size_score(400.0) == 1.0 and size_score(2000.0) == 1.5
