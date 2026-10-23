"""Unit tests for character/hands.py's failure paths: a hand it cannot read, or a failed mesh export, gives a
hand_rig.json with "ok": false (retarget then keeps the fixed grip), never a crash."""

from __future__ import annotations

import json

import numpy as np

from rybakina3d.blender import fingers as fg
from rybakina3d.character import hands

BONES = ["Hips", "RightHand", "LeftHand", "RightForeArm", "LeftForeArm"]
MAPPING = {"pelvis": "Hips", "right_wrist": "RightHand", "left_wrist": "LeftHand", "right_elbow": "RightForeArm",
           "left_elbow": "LeftForeArm"}


def _tiny_mesh(per_hand: int = 300) -> dict:
    """An exported hand mesh (fingers.py export's arrays) with a few hundred vertices per hand: a coarse hand."""
    rest = np.tile(np.eye(4), (len(BONES), 1, 1))
    for i, x in enumerate([0.0, -0.5, 0.5, -0.3, 0.3]):
        rest[i, :3, 3] = [x, 0.0, 1.0 if i < 3 else 1.1]
    rng = np.random.default_rng(0)
    co = np.r_[rng.normal([-0.55, 0.0, 1.0], 0.02, (per_hand, 3)), rng.normal([0.55, 0.0, 1.0], 0.02, (per_hand, 3))]
    n = len(co) - len(co) % 3
    weights = np.zeros((len(co), 4))
    weights[:per_hand, 0] = weights[per_hand:, 1] = 1.0
    return {"co": co, "tri": np.arange(n).reshape(-1, 3), "weights": weights,
            "groups": np.array(["RightHand", "LeftHand", "RightForeArm", "LeftForeArm"]), "bones": np.array(BONES),
            "parents": np.array([-1, 0, 0, 0, 0]), "rest_world": rest, "mapping": json.dumps(MAPPING),
            "vertices": np.int64(len(co)), "blender": "test"}


def test_a_coarse_hand_is_not_ok_and_never_raises():
    data, arrays = hands.build_hand_rig(_tiny_mesh(), "character/rigged.glb", 1e-5)
    assert data["ok"] is False and arrays == {} and data["version"] == fg.HAND_RIG_VERSION
    for side in fg.SIDES:
        assert data["sides"][side]["ok"] is False and data["sides"][side]["reason"].startswith("coarse hand")
    assert "Right: coarse hand" in data["reason"] and "Left: coarse hand" in data["reason"]


def test_a_failed_export_writes_a_not_ok_hand_rig(tmp_path, monkeypatch):
    def fail(rig, weld_m, out, blender=None):
        raise RuntimeError("blender fingers.py export failed:\n...\nRuntimeError: no bone for ['right_wrist']")

    monkeypatch.setattr(hands, "export_mesh", fail)
    out = tmp_path / "character" / "hand_rig.json"
    out.parent.mkdir()
    out.with_suffix(".npz").write_bytes(b"an earlier run's arrays")
    data = hands.make(tmp_path / "rigged.glb", out, 1e-5, label="character/rigged.glb")
    written = json.loads(out.read_text())
    assert written == data and data["ok"] is False and data["rig"] == "character/rigged.glb"
    assert data["reason"] == "hand mesh export failed: RuntimeError: no bone for ['right_wrist']"
    assert not out.with_suffix(".npz").exists()                            # no stale arrays next to a failed rig
