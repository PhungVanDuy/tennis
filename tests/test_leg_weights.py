"""Unit tests for the leg skin weights: the numpy half of blender/retarget.py's leg_weights and the job settings
rybakina3d.assemble makes of it."""

from __future__ import annotations

from typing import get_args

import numpy as np

from rybakina3d import assemble
from rybakina3d.blender import retarget as rt
from rybakina3d.config import PROJECT_ROOT, Config, Retarget


# ------------------------------------------------------------------------------------- retarget.py (numpy)
def test_knee_plane_is_the_bisector_through_the_knee():
    hip, knee, ankle = np.array([0, 0, 1.0]), np.array([0, 0, 0.5]), np.array([0, 0.5, 0.5])   # a 90 deg bend
    point, n = rt.knee_plane(hip, knee, ankle)
    assert np.allclose(point, knee)
    assert np.allclose(n, np.array([0, 1.0, -1.0]) / np.sqrt(2))
    straight = rt.knee_plane([0, 0, 1.0], [0, 0, 0.5], [0, 0, 0.0])[1]
    assert np.allclose(straight, [0, 0, -1.0])


def test_knee_band_splits_on_the_plane_and_keeps_the_leg_total():
    point, n, half = np.zeros(3), np.array([0, 0, -1.0]), 0.05
    co = np.array([[0, 0, 0.2], [0, 0, 0.05], [0, 0.03, 0.0], [0, 0, -0.025], [0, 0, -0.05], [0, 0, -0.3]])
    w_thigh = np.array([1.0, 0.9, 0.4, 0.1, 0.0, 0.0])
    w_shin = np.array([0.0, 0.0, 0.4, 0.8, 0.9, 1.0])
    t, s = rt.knee_band(co, w_thigh, w_shin, point, n, half)
    assert np.allclose(t + s, w_thigh + w_shin)                     # hip and ankle blends untouched
    assert s[0] == 0 and s[1] == 0 and t[4] == 0 and t[5] == 0      # outside the band: one bone only
    assert np.isclose(s[2], 0.4) and np.isclose(t[2], 0.4)          # on the plane: half and half
    assert 0.5 < s[3] / (t[3] + s[3]) < 1                            # below the joint: mostly shin
    # a vertex with no leg weight keeps none
    assert rt.knee_band(co[:1], [0.0], [0.0], point, n, half)[1][0] == 0


def test_leg_modes_are_known_and_default_to_the_shipped_rig():
    assert rt.LEG_WEIGHTS == get_args(Retarget.model_fields["leg_weights"].annotation)
    assert Retarget().leg_weights == "meshy" and rt.KNEE_BAND_CM == Retarget().knee_band_cm


# ------------------------------------------------------------------------------------------- assemble
def test_leg_settings_leave_the_default_job_unchanged():
    cfg = Config.load()
    assert cfg.retarget.leg_weights == "meshy"
    assert assemble.leg_settings(cfg) == {}


def test_naomi_ships_the_knee_band():
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    assert assemble.leg_settings(cfg) == {"leg_weights": "band", "knee_band_cm": 5.0}


def test_leg_settings_per_mode():
    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    cfg.retarget.knee_band_cm = 6.0
    assert assemble.leg_settings(cfg) == {"leg_weights": "band", "knee_band_cm": 6.0}
    cfg.retarget.leg_weights = "meshy"
    assert assemble.leg_settings(cfg) == {}
