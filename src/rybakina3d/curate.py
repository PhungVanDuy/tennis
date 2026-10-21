"""Stage `curate`: check every recovered clip against what the stroke must look like, decide the moveset.

Selection over-generates candidates; the motion itself is the stricter judge of the labels:
  * serve / overhead — the hitting wrist must rise well above the head; contact = its highest point;
  * forehand / backhand / slice / volley — contact = peak hitting-hand speed near the audio contact
    (a two-handed backhand uses both wrists), and that speed must be a real swing.
The refined contact index is written back to clip.json, always re-derived from the contact `select` chose,
so re-running is idempotent; a clip that fails its check keeps that original contact.
Clips are ranked per stroke (size of the player, QA gate, tracking; valid ones first). With
`curate.keep_all` every clip goes into the moveset and failed checks are only flagged; otherwise the best
`select.strokes[type]` valid clips per stroke are kept and the rest dropped.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.signal import medfilt

from .config import Config
from .motion.smpl import HEAD, L_WRIST, R_WRIST
from .selection import size_score
from .util import log, read_json, read_npz, write_json

OVERHEAD_MIN_RISE_M = 0.25   # wrist above the head at contact on a serve / smash
SWING_MIN_SPEED_MS = 4.0     # a groundstroke's hitting hand moves at least this fast near contact
SEARCH_OVERHEAD = 25         # frames around the labelled contact to look for the wrist apex (the selected
                             # contact can sit ~20 frames early on a serve: the toss bounce or a late strike)
SEARCH_GROUND = 14           # frames around the audio contact to look for the peak hand speed (audio can
                             # lock onto the opponent's or a nearby strike: offsets of 9-14 frames were seen)
QA_PENALTY = {"PASS": 0.0, "WARN": 0.3, "FAIL": 1.0}


def _speed(p: np.ndarray, fps: float) -> np.ndarray:
    v = np.linalg.norm(np.diff(p, axis=0), axis=-1) * fps
    v = np.concatenate([v[:1], v])
    return medfilt(v, 5)  # single-frame pose glitches are not swings


def check(stroke: str, J: np.ndarray, contact: int, fps: float, handed: str = "right") -> dict:
    """Validate one clip's motion against its stroke label; returns the refined contact and reasons."""
    hit = R_WRIST if handed == "right" else L_WRIST
    n = len(J)
    if stroke in ("serve", "overhead"):
        lo, hi = max(0, contact - SEARCH_OVERHEAD), min(n, contact + SEARCH_OVERHEAD + 1)
        rise = J[lo:hi, hit, 1] - J[lo:hi, HEAD, 1]
        k = lo + int(np.argmax(rise))
        ok = float(rise.max()) >= OVERHEAD_MIN_RISE_M
        return {"ok": ok, "contact_index": k, "measure": round(float(rise.max()), 3),
                "reason": None if ok else f"hitting wrist only {rise.max():+.2f} m above the head"}
    hands = [L_WRIST, R_WRIST] if stroke == "backhand" else [hit]
    local = J - J[:, :1]  # pelvis-relative: the world root carries GVHMR's depth noise, not the swing
    sp = np.mean([_speed(local[:, h], fps) for h in hands], axis=0)
    lo, hi = max(0, contact - SEARCH_GROUND), min(n, contact + SEARCH_GROUND + 1)
    k = lo + int(np.argmax(sp[lo:hi]))
    ok = float(sp[k]) >= SWING_MIN_SPEED_MS
    return {"ok": ok, "contact_index": k, "measure": round(float(sp[k]), 2),
            "reason": None if ok else f"hand speed near contact only {sp[k]:.1f} m/s"}


def run(cfg: Config) -> dict:
    run_ = cfg.run
    rows = []
    for cid in run_.clip_ids():
        clip = run_.clip(cid)
        if not clip.motion.exists():
            continue
        meta = read_json(clip.meta)
        m = read_npz(clip.motion)
        selected = int(meta.get("contact_index_selected", meta["contact_index"]))
        res = check(meta["stroke"], m["joints_world"], selected, float(m["fps"]), cfg.athlete.handedness)
        qa = read_json(clip.qa) if clip.qa.exists() else {}
        gate = qa.get("gate", "PASS")
        size = size_score(meta.get("hitter_height_px", 0))
        score = meta.get("score", 0.0) + 0.5 * size - QA_PENALTY.get(gate, 0.5)
        contact = res["contact_index"] if res["ok"] else selected  # a failed check proves nothing about contact
        rows.append({"id": cid, "stroke": meta["stroke"], "ok": res["ok"] and gate != "FAIL",
                     "label_ok": res["ok"], "score": round(score, 3), "qa": gate, "contact_index": contact,
                     "measure": res["measure"], "reason": res["reason"] or (f"QA {gate}" if gate == "FAIL" else None)})
        _write_contact(clip, meta, selected, contact)

    kept: list[str] = []
    for stroke, quota in cfg.select.strokes.items():
        ranked = sorted((r for r in rows if r["stroke"] == stroke), key=lambda r: (not r["ok"], -r["score"]))
        valid = [r for r in ranked if r["ok"]]
        flagged = [r for r in ranked if not r["ok"]]
        kept += [r["id"] for r in (ranked if cfg.curate.keep_all else valid[:quota])]
        if ranked:
            what = "flagged" if cfg.curate.keep_all else "rejected"
            log(f"{stroke:10s} keep {len(ranked) if cfg.curate.keep_all else min(quota, len(valid))}"
                f"{'' if cfg.curate.keep_all else f'/{quota}'}  (valid {len(valid)}, {what} {len(flagged)}"
                + (": " + ", ".join(f"{r['id']} — {r['reason']}" for r in flagged) if flagged else "") + ")")
    for r in rows:
        r["kept"] = r["id"] in kept
    out = {"keep_all": cfg.curate.keep_all, "kept": kept, "clips": rows}
    write_json(run_.curation, out)
    return out


def _write_contact(clip, meta: dict, selected: int, contact: int) -> None:
    """Store the contact in clip.json, keeping the one `select` chose (and its source) for re-runs."""
    meta.setdefault("contact_index_selected", selected)
    base = meta.setdefault("contact_source_selected", re.sub(r"(\+motion)+$", "", meta["contact_source"]))
    new = {"contact_index": contact, "contact_source": base + ("+motion" if contact != selected else "")}
    if any(meta.get(k) != v for k, v in new.items()):
        write_json(clip.meta, {**meta, **new})
