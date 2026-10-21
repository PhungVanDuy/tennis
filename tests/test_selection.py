"""Contact refinement on the soundtrack: a sharp strike near Gemini's estimate wins."""

from __future__ import annotations

import numpy as np
from scipy.io import wavfile

from rybakina3d.selection import Onsets


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
