"""Stage `signature`: what makes the strokes *hers*, measured on the recovered motion.

Per clip (SMPL joints, world frame, Y-up) around the refined contact:
  serve        trophy knee flex, toss-hand apex above the head, contact height, toss→contact time
  groundstroke stance width, knee flex, hip–shoulder separation (coil), contact height, hand speed
Heights are normalised by her standing height and widths by leg length, so numbers compare across clips
and can be compared with another player's. Per stroke: mean ± sd over the moveset clips whose motion passed
`curate`'s stroke check (a flagged clip, e.g. a return of serve labelled as a serve, would skew the stroke).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import medfilt

from .config import Config
from .motion.smpl import HEAD, L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER, R_WRIST
from .util import log, read_json, read_npz, write_json


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) of the chain a-b-c, per frame."""
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _yaw(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = q - p
    return np.degrees(np.arctan2(d[:, 2], d[:, 0]))


def features(stroke: str, J: np.ndarray, rest: np.ndarray, k: int, fps: float, handed: str = "right") -> dict:
    hit, toss = (R_WRIST, L_WRIST) if handed == "right" else (L_WRIST, R_WRIST)
    leg = float(np.linalg.norm(rest[L_HIP] - rest[L_KNEE]) + np.linalg.norm(rest[L_KNEE] - rest[L_ANKLE]))
    height = float(rest[HEAD, 1] - min(rest[L_ANKLE, 1], rest[R_ANKLE, 1])) + 0.12  # crown above the head joint
    ground = np.minimum(J[:, L_ANKLE, 1], J[:, R_ANKLE, 1]).min() - 0.08          # ankle sits ~8 cm up
    pre = slice(max(0, k - round(1.2 * fps)), k + 1)
    knee = np.minimum(_angle(J[:, L_HIP], J[:, L_KNEE], J[:, L_ANKLE]),
                      _angle(J[:, R_HIP], J[:, R_KNEE], J[:, R_ANKLE]))
    speed = medfilt(np.linalg.norm(np.gradient(J[:, hit], axis=0), axis=-1) * fps, 5)
    out = {
        "contact_height": round(float((J[k, hit, 1] - ground) / height), 3),
        "knee_flex_deg": round(float(180 - knee[pre].min()), 1),
        "hand_speed_ms": round(float(speed[max(0, k - 3): k + 4].max()), 1),
    }
    if stroke in ("serve", "overhead"):
        win = slice(max(0, k - round(2.0 * fps)), k + 1)
        apex = win.start + int(np.argmax(J[win, toss, 1]))
        out["toss_above_head_m"] = round(float(J[apex, toss, 1] - J[apex, HEAD, 1]), 2)
        out["toss_to_contact_s"] = round((k - apex) / fps, 2)
    else:
        sep = _yaw(J[:, L_SHOULDER], J[:, R_SHOULDER]) - _yaw(J[:, L_HIP], J[:, R_HIP])
        sep = (sep + 180) % 360 - 180
        out["stance_width"] = round(float(np.linalg.norm((J[k, L_ANKLE] - J[k, R_ANKLE])[[0, 2]]) / leg), 2)
        out["hip_shoulder_sep_deg"] = round(float(np.abs(sep[pre]).max()), 1)
    return out


def run(cfg: Config) -> dict:
    run_ = cfg.run
    per_clip, by_stroke = {}, {}
    rows = {r["id"]: r for r in read_json(run_.curation)["clips"]} if run_.curation.exists() else {}
    for cid in run_.kept_clip_ids():
        if cid in rows and not rows[cid].get("label_ok", rows[cid]["ok"]):
            continue
        clip = run_.clip(cid)
        if not clip.motion.exists():
            continue
        meta = read_json(clip.meta)
        m = read_npz(clip.motion)
        f = features(meta["stroke"], m["joints_world"], m["rest_joints"], int(meta["contact_index"]), float(m["fps"]),
                     cfg.athlete.handedness)
        per_clip[cid] = {"stroke": meta["stroke"], **f}
        by_stroke.setdefault(meta["stroke"], []).append(f)
    summary = {}
    for stroke, rows in by_stroke.items():
        keys = rows[0].keys()
        summary[stroke] = {k: {"mean": round(float(np.mean([r[k] for r in rows])), 3),
                               "sd": round(float(np.std([r[k] for r in rows])), 3)} for k in keys}
        summary[stroke]["n"] = len(rows)
        log(f"{stroke:9s} n={len(rows)}  " + "  ".join(f"{k}={v['mean']:.2f}±{v['sd']:.2f}"
                                                      for k, v in summary[stroke].items() if k != "n"))
    out = {"per_clip": per_clip, "by_stroke": summary}
    write_json(run_.signature, out)
    return out
