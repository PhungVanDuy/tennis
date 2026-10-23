"""Unit tests for the hand settings (config.Retarget grip / fingers / left_hand_ik / grip_metrics): off in code and in
rybakina.yaml, rybakina_5s.yaml and naomi_5s.yaml, so those jobs are the ones from before them; switched on by
configs/naomi.yaml and configs/naomi_5s_grip.yaml; hand_rig.json remade only when it is missing or stale, and passed to
the job only when fresh."""

from __future__ import annotations

import json
import os

import pytest
from pydantic import ValidationError

from rybakina3d import assemble
from rybakina3d.blender import fingers
from rybakina3d.config import PROJECT_ROOT, Config, Retarget

SHIPPED = ("rybakina.yaml", "rybakina_5s.yaml", "naomi_5s.yaml")
HAND_KEYS = {"grip", "fingers", "left_hand_ik", "grip_metrics", "hand_rig", "hand_rig_label"}
CLIP = {"id": "forehand_00m07s", "motion": "motion.npz", "stroke": "forehand", "view": "behind_low",
        "contact_index": 39, "n_frames": 127, "clip_size": [1920, 1080], "check_frames": [29, 39, 45]}


def _grip_config(run_dir) -> Config:
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi_5s_grip.yaml")
    cfg.run_dir = run_dir
    return cfg


def test_hand_settings_leave_the_shipped_jobs_unchanged():
    for name in SHIPPED:
        cfg = Config.load(PROJECT_ROOT / "configs" / name)
        r = cfg.retarget
        assert (r.grip, r.fingers, r.left_hand_ik, r.grip_metrics, r.hands) == ("fixed", False, False, False, False)
        assert assemble.hand_settings(cfg) == {}
        assert assemble.ensure_hand_rig(cfg) is None      # nothing made, scipy never imported for these
        settings = assemble.make_job(cfg, [CLIP], 25.0)["settings"]   # the whole job, not only hand_settings
        assert not HAND_KEYS & set(settings)
        assert set(settings) - {"racket_colors", "leg_weights", "knee_band_cm"} == {
            "smooth_window", "ground_from_contacts", "foot_lock", "racket", "weld_m", "smooth_limb_weights"}


def test_fingers_need_the_palm_seat():
    with pytest.raises(ValidationError):
        Retarget(fingers=True)
    assert Retarget(grip="palm", fingers=True).hands
    assert Retarget(left_hand_ik=True).hands and Retarget(grip_metrics=True).hands and not Retarget().hands


@pytest.mark.parametrize("name, run_dir", [("naomi.yaml", "rybakina"), ("naomi_5s_grip.yaml", "rybakina_5s_grip")])
def test_grip_configs_switch_the_hands_on(tmp_path, name, run_dir):
    cfg = Config.load(PROJECT_ROOT / "configs" / name)
    r = cfg.retarget
    assert (r.grip, r.fingers, r.left_hand_ik) == ("palm", True, True)
    assert cfg.run_dir == PROJECT_ROOT / "runs" / run_dir and cfg.character.variant == "naomi"
    assert (r.weld_m, r.smooth_limb_weights, r.leg_weights) == (1e-5, 20, "band")   # naomi's skin fixes still apply
    cfg.run_dir = tmp_path
    on = {"grip": "palm", "fingers": True, "left_hand_ik": True, "grip_metrics": True}
    assert assemble.hand_settings(cfg) == on           # no hand_rig.json yet: retarget then keeps the fixed grip
    cfg.run.rig.parent.mkdir(parents=True)
    cfg.run.rig.write_bytes(b"glb")
    _write_hand_rig(cfg)
    label = str(cfg.run.hand_rig.relative_to(cfg.run.root))
    assert assemble.hand_settings(cfg) == {**on, "hand_rig": str(cfg.run.hand_rig), "hand_rig_label": label}
    assert not label.startswith("/") and label.endswith("character/hand_rig.json")
    assert HAND_KEYS <= set(assemble.make_job(cfg, [CLIP], 25.0)["settings"])
    _write_hand_rig(cfg, version=fingers.HAND_RIG_VERSION + 1)
    assert assemble.hand_settings(cfg) == on           # another format: not handed to Blender


def _write_hand_rig(cfg: Config, **fields) -> None:
    data = {"version": fingers.HAND_RIG_VERSION, "ok": True, "reason": None, "weld_m": cfg.retarget.weld_m} | fields
    cfg.run.hand_rig.write_text(json.dumps(data))


def test_hand_rig_is_made_when_missing_or_stale(tmp_path, monkeypatch):
    from rybakina3d.character import hands

    made = []
    monkeypatch.setattr(hands, "make", lambda rig, out, weld_m, blender=None, label=None: made.append((out, weld_m)))
    cfg = _grip_config(tmp_path)
    cfg.run.rig.parent.mkdir(parents=True)
    cfg.run.rig.write_bytes(b"glb")
    assert assemble.ensure_hand_rig(cfg) == cfg.run.hand_rig and made == [(cfg.run.hand_rig, 1e-5)]   # missing
    _write_hand_rig(cfg)
    assert assemble.ensure_hand_rig(cfg) and len(made) == 1                                         # fresh
    os.utime(cfg.run.rig, (cfg.run.hand_rig.stat().st_mtime + 10,) * 2)
    assert assemble.ensure_hand_rig(cfg) and len(made) == 2                                         # older than the rig
    os.utime(cfg.run.rig, (cfg.run.hand_rig.stat().st_mtime - 10,) * 2)
    cfg.retarget.weld_m = 0.0
    assert assemble.ensure_hand_rig(cfg) and len(made) == 3                                         # another weld
    _write_hand_rig(cfg, version=fingers.HAND_RIG_VERSION - 1)
    assert assemble.ensure_hand_rig(cfg) and len(made) == 4                                         # another format
    _write_hand_rig(cfg, ok=False, reason="hand mesh export failed: no Blender")
    assert assemble.ensure_hand_rig(cfg) and len(made) == 5                                         # retried
    _write_hand_rig(cfg, ok=False, reason="Right: coarse hand: 420 vertices (< 1500)")
    assert assemble.ensure_hand_rig(cfg) and len(made) == 5             # a hand that cannot be rigged stays as it is
    cfg.retarget.grip, cfg.retarget.fingers = "fixed", False
    _write_hand_rig(cfg, version=fingers.HAND_RIG_VERSION - 1)
    assert assemble.ensure_hand_rig(cfg) and len(made) == 6             # the IK alone uses it too: checked as well
    cfg.retarget.left_hand_ik = False
    assert assemble.ensure_hand_rig(cfg) is None and len(made) == 6     # every hand setting off
