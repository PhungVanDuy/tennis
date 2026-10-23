"""Contact refinement on the soundtrack: a sharp strike near Gemini's estimate wins."""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from rybakina3d.config import Config
from rybakina3d.selection import Onsets, _candidates


def test_onset_refine_finds_the_strike(tmp_path):
    sr = 16000
    rng = np.random.default_rng(0)
    y = 0.01 * rng.standard_normal(sr * 6)
    t_hit = 3.27
    i = int(t_hit * sr)
    y[i : i + 160] += 0.8 * np.sin(np.linspace(0, 400 * np.pi, 160))  # 10 ms broadband click
    path = tmp_path / "a.wav"
    wavfile.write(path, sr, (y * 32767).astype(np.int16))

    onsets = Onsets(path)
    t, z = onsets.refine(3.1, radius=0.5)
    assert abs(t - t_hit) < 0.03
    assert z > 2.5


def test_source_span_keeps_only_the_strokes_inside_it():
    cfg = Config.load()
    shot = {"shot_id": 0, "view": "behind_low", "replay": False, "slow_motion": False, "rybakina_side": "near",
            "full_body": True, "quality": 4, "outfit": "", "start_frame": 0, "end_frame": 460,
            "events": [{"hitter": "rybakina", "type": t, "t": s, "confidence": 0.9}
                       for t, s in (("forehand", 7.3), ("backhand", 9.6))]}
    assert [c.stroke for c in _candidates(cfg, {"shots": [shot]})] == ["forehand", "backhand"]
    cfg.select.source_span_s = (6.5, 8.5)
    assert [c.stroke for c in _candidates(cfg, {"shots": [shot]})] == ["forehand"]
