"""The stroke checks in `curate` on synthetic joint trajectories."""

from __future__ import annotations

import numpy as np

from rybakina3d.curate import check
from rybakina3d.motion.smpl import HEAD, L_WRIST, R_WRIST

FPS = 25.0


def _body(n: int) -> np.ndarray:
    J = np.zeros((n, 24, 3))
    J[:, HEAD, 1] = 1.6
    J[:, L_WRIST] = [-0.3, 0.9, 0.0]
    J[:, R_WRIST] = [0.3, 0.9, 0.0]
    return J


def test_serve_contact_is_the_wrist_apex():
    J = _body(60)
    J[:, R_WRIST, 1] = 0.9 + 1.2 * np.exp(-((np.arange(60) - 40) ** 2) / 20)  # apex at frame 40, 2.1 m
    res = check("serve", J, contact=34, fps=FPS)
    assert res["ok"] and res["contact_index"] == 40


def test_serve_label_rejected_when_the_arm_never_goes_up():
    res = check("serve", _body(60), contact=30, fps=FPS)
    assert not res["ok"] and "above the head" in res["reason"]


def test_forehand_contact_snaps_to_peak_hand_speed():
    J = _body(50)
    x = np.cumsum(np.where(np.abs(np.arange(50) - 27) < 3, 0.5, 0.0))  # fast swing around frame 27
    J[:, R_WRIST, 0] += x
    res = check("forehand", J, contact=25, fps=FPS)
    assert res["ok"] and abs(res["contact_index"] - 27) <= 2


def test_still_hand_is_not_a_groundstroke():
    res = check("backhand", _body(50), contact=25, fps=FPS)
    assert not res["ok"]


def test_contact_refinement_is_idempotent(tmp_path):
    from types import SimpleNamespace

    from rybakina3d.curate import _write_contact
    from rybakina3d.util import read_json, write_json

    clip = SimpleNamespace(meta=tmp_path / "clip.json")
    write_json(clip.meta, {"contact_index": 55, "contact_source": "audio+motion+motion", "contact_index_selected": 50})
    for _ in range(3):  # re-running curate must not drift
        meta = read_json(clip.meta)
        _write_contact(clip, meta, int(meta["contact_index_selected"]), 52)
    meta = read_json(clip.meta)
    assert meta["contact_index"] == 52 and meta["contact_source"] == "audio+motion"
    _write_contact(clip, meta, 50, 50)  # a failed check falls back to the selected contact
    assert read_json(clip.meta)["contact_source"] == "audio"
