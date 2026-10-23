"""Stage `site`: the local review website, a static single-page app over the run folder.

build() copies web/ into <run>/site/, aggregates what every stage wrote into data/site.json (plus one small
skeleton file per recovered clip under data/motion/) and exposes every media file the page shows (previews,
overlays, track / camera checks, character images, meshes, the moveset, retarget checks, the artwork sheets) as a
*relative symlink* under media/. Nothing heavy is copied (the workstation disk is small) and the run folder stays
relocatable. A missing output is written as null and the page shows it as pending, so the site can be
rebuilt at any point while stages are still running. Every character variant of the run (variants/<name>/: another
character on the same motion) is listed under `variants` with its moveset in motion at the Quest 3 render profile
(its reel, every clip's full-resolution video and light tile) or, until those exist, the profile's draft.

serve() is a threaded static server for that folder: it follows the symlinks, sends the right MIME types
(model/gltf-binary for .glb) and answers Range requests with 206 so Chrome can seek inside <video>.
"""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from datetime import datetime
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .camera import CALIBRATED_VIEWS
from .config import PROJECT_ROOT, ClipLayout, Config, RunLayout
from .curate import OVERHEAD_MIN_RISE_M, SWING_MIN_SPEED_MS
from .gltf import glb_stats
from .media import _bin
from .motion.smpl import PARENTS
from .util import log, read_json, step, write_json

WEB_DIR = PROJECT_ROOT / "web"

# Optional outputs of other stages, looked up by name (not part of RunLayout yet).
TIMINGS_FILE = "timings.json"          # {stage: seconds}, any stage may add to it
CHECK_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm"}

# GVHMR joints_world carries the 24 SMPL joints: the 22 body joints plus the two hands.
SMPL24_PARENTS = [*PARENTS, 20, 21]
PELVIS, L_HIP, R_HIP, L_WRIST, R_WRIST = 0, 1, 2, 20, 21
STROKES = ("serve", "forehand", "backhand", "slice", "volley", "overhead")
CONTACT_P = 0.5                        # static_conf above this = planted (same as `qa`)


# --- media links ----------------------------------------------------------------------------------------
class _Linker:
    """Relative symlinks site/media/<rel> → run files; returns the page URL, None for a missing file.

    A rebuild swaps each link in place (rename over the old one) and prunes the links it did not write only at
    the end, so a running `r3d serve` never answers 404 for a file that exists while the site is rebuilt.
    """

    def __init__(self, site: Path):
        self.media = site / "media"
        self.count = 0
        self.written: set[Path] = set()

    def prune(self) -> None:
        """Remove links (then empty folders) left by a previous build that this one did not write."""
        if not self.media.exists():
            return
        for p in sorted(self.media.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if p.is_symlink():
                if p not in self.written:
                    p.unlink()
            elif p.is_dir() and not any(p.iterdir()):
                p.rmdir()

    def __call__(self, src: Path | None, rel: str) -> str | None:
        if src is None or not src.exists():
            return None
        dst = self.media / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(os.path.relpath(src, dst.parent))
        os.replace(tmp, dst)  # atomic: the old link stays valid until the new one takes its place
        self.written.add(dst)
        self.count += 1
        return f"media/{rel}"


def _file(link: _Linker, src: Path | None, rel: str) -> dict | None:
    url = link(src, rel)
    if url is None:
        return None
    # Tripo's FBX is gzip-encoded on disk; the page sniffs and inflates it itself
    return {"url": url, "size_mb": round(src.stat().st_size / 1e6, 2), "kind": src.suffix.lstrip(".").lower()}


def _video(link: _Linker, src: Path, rel: str) -> dict | None:
    """URL + where the first frame sits on the media timeline (previews with audio start one frame late)."""
    url = link(src, rel)
    if url is None:
        return None
    info: dict[str, Any] = {"url": url, "start": 0.0, "frames": None}
    try:
        out = subprocess.run(
            [_bin("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=start_time,nb_frames,width,height", "-of", "default=noprint_wrappers=1", str(src)],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        kv = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
        start = kv.get("start_time", "N/A")
        info.update(start=float(start) if start != "N/A" else 0.0, frames=_int(kv.get("nb_frames")),
                    width=_int(kv.get("width")), height=_int(kv.get("height")))
    except (RuntimeError, subprocess.SubprocessError, ValueError) as e:
        log(f"[yellow]ffprobe {src.name}: {e}[/]")
    return info


def _int(v: str | None) -> int | None:
    try:
        return int(v) if v not in (None, "N/A") else None
    except ValueError:
        return None


def _mtime(*paths: Path) -> str | None:
    times = [p.stat().st_mtime for p in paths if p.exists()]
    return datetime.fromtimestamp(max(times)).isoformat(timespec="seconds") if times else None


def _maybe_json(path: Path) -> Any:
    try:
        return read_json(path) if path.exists() else None
    except ValueError as e:
        log(f"[yellow]{path.name}: unreadable JSON ({e})[/]")
        return None


def _finite(o: Any) -> Any:
    """NaN / inf → None everywhere: json.dumps writes them as bare NaN, which the browser refuses to parse."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _round(x: Any, nd: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# --- footage -------------------------------------------------------------------------------------------
def _shot_usable(cfg: Config, s: dict) -> str | None:
    """Why `select` would skip this shot (None = usable); mirrors selection._candidates."""
    sel = cfg.select
    if s["view"] not in sel.allowed_views:
        return f"view {s['view']}"
    if s["replay"]:
        return "replay"
    if s["slow_motion"] and not sel.allow_slow_motion:
        return "slow motion"
    if s["rybakina_side"] not in ("near", "solo"):
        return f"Rybakina {s['rybakina_side']}"
    if not s["full_body"]:
        return "body cut off"
    if s["quality"] < sel.min_quality:
        return f"quality {s['quality']} < {sel.min_quality}"
    return None


def _footage(cfg: Config, clip_metas: dict[str, dict]) -> dict:
    run = cfg.run
    probe = _maybe_json(run.probe) or {}
    shots = (_maybe_json(run.shots) or {}).get("shots", [])
    index = _maybe_json(run.index) or {}
    labels = {s["shot_id"]: s for s in index.get("shots", [])}
    clips_by_shot: dict[int, list[str]] = {}
    for cid, m in clip_metas.items():
        clips_by_shot.setdefault(m.get("shot_id"), []).append(cid)

    rows, by_view, strokes, opp = [], {}, dict.fromkeys(STROKES, 0), 0
    replays = slow = full = usable = 0
    quality = dict.fromkeys(range(1, 6), 0)
    for s in shots:
        lab = labels.get(s["id"])
        row = {"id": s["id"], "start": s["start"], "end": s["end"], "start_frame": s["start_frame"],
               "end_frame": s["end_frame"], "clips": sorted(clips_by_shot.get(s["id"], [])), "label": None}
        if lab:
            events = sorted(lab["events"], key=lambda e: e["t"])
            skip = _shot_usable(cfg, lab)
            row["label"] = {k: lab[k] for k in ("view", "replay", "slow_motion", "rybakina_side", "full_body",
                                                 "quality", "outfit")}
            row["label"]["events"] = events
            row["label"]["skip"] = skip
            v = by_view.setdefault(lab["view"], {"shots": 0, "seconds": 0.0})
            v["shots"] += 1
            v["seconds"] = round(v["seconds"] + s["end"] - s["start"], 2)
            replays += lab["replay"]
            slow += lab["slow_motion"]
            full += lab["full_body"]
            usable += skip is None
            quality[int(lab["quality"])] = quality.get(int(lab["quality"]), 0) + 1
            for e in events:
                if e["hitter"] == "rybakina" and e["type"] in strokes:
                    strokes[e["type"]] += 1
                elif e["hitter"] == "opponent" and e["type"] in STROKES:
                    opp += 1
        rows.append(row)

    return {
        "video": cfg.source.video.name,
        "youtube": cfg.source.youtube,
        "probe": probe,
        "minutes": _round(probe.get("duration", 0) / 60, 1) if probe else None,
        "n_shots": len(shots),
        "n_labelled": len(labels),
        "index_model": index.get("model"),
        "dropped_events": index.get("dropped_events"),
        "missing_labels": index.get("missing", []),
        "by_view": by_view,
        "strokes": strokes,
        "n_strokes": sum(strokes.values()),
        "opponent_strokes": opp,
        "replays": replays, "slow_motion": slow, "full_body": full, "usable": usable,
        "quality": quality,
        "shots": rows,
    }


# --- clips ---------------------------------------------------------------------------------------------
def _body_frame(j: np.ndarray) -> np.ndarray:
    """Rows right, up, forward of a body from its (24,3) SMPL joints (Y-up; left hip is +X when facing +Z)."""
    lateral = j[L_HIP] - j[R_HIP]
    lateral[1] = 0.0
    lateral /= max(np.linalg.norm(lateral), 1e-9)
    up = np.array([0.0, 1.0, 0.0])
    return np.stack([-lateral, up, np.cross(lateral, up)])


def _camera_rel(jw: np.ndarray, jc: np.ndarray, K: np.ndarray, size: list[int]) -> dict | None:
    """Where the broadcast camera stood on one frame, in the player's body frame (pelvis origin).

    joints_camera and joints_world are the same body under two rigid transforms, so Kabsch recovers the
    world → camera pose; the page re-applies it around the retargeted character for its 'broadcast' view.
    """
    ca, cb = jw.mean(0), jc.mean(0)
    U, _, Vt = np.linalg.svd((jw - ca).T @ (jc - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cb - R @ ca
    resid = float(np.linalg.norm(jw @ R.T + t - jc, axis=1).mean())
    if resid > 0.05:  # not one rigid body: do not pretend we know the camera
        return None
    centre = -R.T @ t
    basis = _body_frame(jw)
    vfov = 2 * math.degrees(math.atan(size[1] / 2 / float(K[1, 1])))
    return {"pos": np.round(basis @ (centre - jw[PELVIS]), 3).tolist(),
            "distance_m": _round(np.linalg.norm(centre - jw[PELVIS]), 2), "vfov_deg": _round(vfov, 2),
            "fit_m": _round(resid, 4)}


def _load_npz(path: Path, *keys: str) -> dict | None:
    """All arrays of an npz, or None (with a warning) when it is unreadable or lacks one of `keys`."""
    if not path.exists():
        return None
    try:
        with np.load(path) as npz:
            data = {k: npz[k] for k in npz.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as e:
        log(f"[yellow]{path.parent.name}/{path.name}: unreadable ({e}), shown as pending[/]")
        return None
    missing = [k for k in keys if k not in data]
    if missing:
        log(f"[yellow]{path.parent.name}/{path.name}: no {', '.join(missing)}, shown as pending[/]")
        return None
    return data


def _motion(clip: ClipLayout, meta: dict, handed: str) -> tuple[dict | None, dict | None]:
    """Review numbers of one motion.npz + the world joints the page draws as the GVHMR skeleton."""
    m = _load_npz(clip.motion, "fps", "joints_world", "joints_camera", "K", "static_conf", "contact_index")
    if m is None or len(m["joints_world"]) == 0:
        return None, None
    J = np.asarray(m["joints_world"], np.float64)
    if not np.isfinite(J).all():
        log(f"[yellow]{clip.id}: non-finite joints in motion.npz, shown as pending[/]")
        return None, None
    n, fps = len(J), float(m["fps"])
    k = int(np.clip(meta.get("contact_index", m["contact_index"]), 0, n - 1))
    hand = R_WRIST if handed == "right" else L_WRIST
    speed = np.linalg.norm(np.gradient(J[:, hand], axis=0), axis=-1) * fps
    pelvis = J[:, PELVIS][:, [0, 2]]
    K = np.asarray(m["K"], np.float64)
    size = meta.get("size", [1920, 1080])
    planted = np.asarray(m["static_conf"])[:, :4] > CONTACT_P
    summary = {
        "frames": n,
        "fps": fps,
        "duration_s": _round(n / fps, 2),
        "contact_index": k,
        "static_camera": bool(m["static_camera"]) if "static_camera" in m else None,
        "camera_motion_px": _round(m["camera_motion_px"], 2) if "camera_motion_px" in m else None,
        "calibrated": bool(m.get("calibrated", False)),
        "f_px": _round(K[k, 0, 0], 1),
        "hfov_deg": _round(2 * math.degrees(math.atan(size[0] / 2 / K[k, 0, 0])), 1),
        "pelvis_travel_m": _round(np.linalg.norm(np.diff(pelvis, axis=0), axis=1).sum(), 2),
        "pelvis_displacement_m": _round(np.linalg.norm(pelvis[-1] - pelvis[0]), 2),
        "hand_speed_contact_ms": _round(speed[max(0, k - 3): k + 4].max(), 1),
        "hand_speed_peak_ms": _round(speed.max(), 1),
        "planted_frac": _round(planted.any(axis=1).mean(), 2),
        "camera_rel": _camera_rel(J[k], np.asarray(m["joints_camera"], np.float64)[k], K[k], size),
    }
    skeleton = {"fps": fps, "n": n, "contact_index": k, "parents": SMPL24_PARENTS,
                "joints": np.round(J, 3).reshape(-1).tolist()}
    return summary, skeleton


def _camera_calib(clip: ClipLayout) -> dict | None:
    c = _load_npz(clip.camera)
    if c is None:
        return None
    out: dict[str, Any] = {"method": str(c["method"]) if "method" in c else None}
    if "ok" in c:
        out["ok_frac"] = _round(np.asarray(c["ok"]).mean(), 3)
    if "reproj_px" in c:
        err = np.asarray(c["reproj_px"], np.float64)
        err = err[err >= 0]
        out["reproj_px"] = _round(np.median(err), 2) if err.size else None
    if "K" in c:
        f = np.asarray(c["K"])[:, 0, 0]
        out["f_px"] = [_round(f.min(), 0), _round(f.max(), 0)]
    return out


def _qa(clip: ClipLayout) -> dict | None:
    q = _maybe_json(clip.qa)
    if not isinstance(q, dict):
        return None
    status = str(q.get("status") or q.get("gate") or "").upper() or None
    return {"status": status, "reasons": q.get("reasons", []), "notes": q.get("notes", []),
            "metrics": {k: v for k, v in q.items() if k not in ("id", "status", "gate", "reasons", "notes")}}


def _find_anim(anims: list[str], cid: str) -> str | None:
    """The moveset animation of a clip: named by its id, or 'Armature|…|<id>' as FBX round-trips write it."""
    if cid in anims:
        return cid
    return next((a for a in anims if a.split("|")[-1] == cid), None)


def _clips(cfg: Config, link: _Linker, site: Path, anims: list[str], retarget: Any,
           checks: list[dict], curation: dict[str, dict], signature: dict[str, dict]) -> list[dict]:
    run = cfg.run
    sel = _maybe_json(run.selection) or {}
    order = [c["id"] for c in sel.get("clips", [])]
    ids = sorted(set(order) | set(run.clip_ids()), key=lambda c: (order.index(c) if c in order else 999, c))
    per_clip_rt = _per_clip(retarget)
    motion_dir = site / "data" / "motion"
    written: set[str] = set()
    out = []
    for cid in ids:
        clip = run.clip(cid)
        meta = _maybe_json(clip.meta) or next((c for c in sel.get("clips", []) if c["id"] == cid), {})
        track = None
        if (t := _load_npz(clip.track, "coverage", "track_id", "bbx_xyxy")) is not None:
            box_h = t["bbx_xyxy"][:, 3] - t["bbx_xyxy"][:, 1] if len(t["bbx_xyxy"]) else np.array([np.nan])
            track = {"coverage": _round(t["coverage"], 3), "track_id": int(t["track_id"]),
                     "box_h_median_px": _round(np.median(box_h), 1)}
        motion, skeleton = _motion(clip, meta, cfg.athlete.handedness)
        skel_url = None
        if skeleton is not None:
            write_json(motion_dir / f"{cid}.json", skeleton)
            written.add(f"{cid}.json")
            skel_url = f"data/motion/{cid}.json"
        out.append({
            "id": cid,
            "meta": meta,
            "media": {
                "preview": _video(link, clip.preview, f"clips/{cid}/preview.mp4"),
                "overlay": _video(link, clip.overlay, f"clips/{cid}/overlay.mp4"),
                "track": link(clip.track_preview, f"clips/{cid}/track.jpg"),
                "camera": link(clip.camera_preview, f"clips/{cid}/camera.jpg"),
            },
            "track": track,
            "motion": motion,
            "skeleton": skel_url,
            "camera_calib": _camera_calib(clip),
            "qa": _qa(clip),
            "curation": curation.get(cid),
            "signature": signature.get(cid),
            "anim": _find_anim(anims, cid),
            "retarget": per_clip_rt.get(cid),
            "checks": [c for c in checks if c.get("clip") == cid],
        })
    # drop skeletons of clips that are gone only now, after the new ones are in place
    for p in motion_dir.glob("*.json") if motion_dir.exists() else []:
        if p.name not in written:
            p.unlink()
    return out


def _per_clip(report: Any) -> dict[str, Any]:
    """retarget.json → {clip id: entry}, whether it keeps clips as a list of rows or a mapping."""
    if not isinstance(report, dict):
        return {}
    clips = report.get("clips")
    if isinstance(clips, dict):
        return dict(clips)
    if isinstance(clips, list):
        return {r["id"]: r for r in clips if isinstance(r, dict) and "id" in r}
    return {}


# --- curation ------------------------------------------------------------------------------------------
OVERHEAD_STROKES = ("serve", "overhead")


def _why(r: dict, rank: int, n_valid: int, quota: int) -> tuple[str, str, str]:
    """(category, short reason for the clip card, full sentence) of one curation.json row."""
    stroke, m, score = r.get("stroke") or "clip", r.get("measure"), r.get("score")
    if r.get("kept") and r.get("ok"):
        return "kept", f"kept #{rank} of {quota}", f"Kept: #{rank} of the {quota} {stroke} clips by curate's score " \
            f"({_round(score, 2)}); its motion matches the label."
    if r.get("kept"):  # curate.keep_all: animated anyway, the failed check is a flag
        cat, short, why = _why({**r, "kept": False}, rank, n_valid, quota)
        return "flagged", f"flagged: {short}", f"In the moveset (every clip is animated), but flagged: {why}"
    if r.get("ok"):
        return "quota", f"over quota: #{rank} of {n_valid} valid", f"Valid {stroke}, but ranked #{rank} of " \
            f"{n_valid} by curate's score ({_round(score, 2)}); the moveset keeps the best {quota}."
    reason = str(r.get("reason") or "")
    also = " Also failed the QA gate." if r.get("qa") == "FAIL" else ""
    if reason.startswith("QA") or m is None:
        return "qa", reason or "rejected", f"Failed the QA gate ({reason or r.get('qa')})."
    if stroke in OVERHEAD_STROKES:
        a = "an" if stroke[:1] in "aeiou" else "a"
        return "label", f"not {a} {stroke}: wrist {m:+.2f} m over head", \
            f"The motion contradicts the '{stroke}' label: her hitting wrist peaks {m:+.2f} m above her head near " \
            f"the contact, {a} {stroke} reaches at least +{OVERHEAD_MIN_RISE_M:.2f} m.{also}"
    return "label", f"no real swing: hand {m:.1f} m/s", \
        f"The motion contradicts the '{stroke}' label: the hitting hand moves only {m:.1f} m/s near the contact, " \
        f"a real {stroke} swing is at least {SWING_MIN_SPEED_MS:.1f} m/s.{also}"


def _curation(cfg: Config) -> tuple[dict | None, dict[str, dict]]:
    """curation.json → (summary for the overview, {clip id: row + why it was kept or rejected})."""
    cur = _maybe_json(cfg.run.curation)
    if not isinstance(cur, dict):
        return None, {}
    rows = []
    for r in cur.get("clips", []):
        if isinstance(r, dict) and "id" in r:
            # the clip's QA gate as it stands now (QA re-runs after the retarget), the one the page shows
            now = (_qa(cfg.run.clip(r["id"])) or {}).get("status") or r.get("qa")
            rows.append({**r, "qa_at_curate": r.get("qa"), "qa": now})
    kept = [k for k in cur.get("kept", []) if any(r["id"] == k for r in rows)]
    stroke_of = {r["id"]: r.get("stroke") for r in rows}
    out: dict[str, dict] = {}
    by_stroke: dict[str, dict] = {}
    for r in rows:
        stroke = r.get("stroke")
        quota = (sum(stroke_of[k] == stroke for k in kept) if cur.get("keep_all")
                 else cfg.select.strokes.get(stroke, 0))
        valid = sorted((x for x in rows if x.get("stroke") == stroke and x.get("ok")), key=lambda x: -x["score"])
        if r.get("kept"):
            rank = [k for k in kept if stroke_of[k] == stroke].index(r["id"]) + 1
        elif r.get("ok"):
            rank = [x["id"] for x in valid].index(r["id"]) + 1
        else:
            rank = 0
        cat, short, why = _why(r, rank, len(valid), quota)
        out[r["id"]] = {**r, "category": cat, "rank": rank or None, "quota": quota, "short": short, "why": why}
        g = by_stroke.setdefault(stroke, {"quota": quota, "kept": 0, "flagged": 0, "label": 0, "qa": 0,
                                          "quota_over": 0})
        g[{"kept": "kept", "flagged": "flagged", "label": "label", "qa": "qa", "quota": "quota_over"}[cat]] += 1
    cats = [v["category"] for v in out.values()]
    label = [v for v in out.values() if v["category"] == "label"]
    flagged = [v for v in out.values() if v["category"] == "flagged"]
    summary = {
        "n": len(rows), "kept": len(kept), "kept_ids": kept, "keep_all": bool(cur.get("keep_all")),
        "flagged": len(flagged), "flagged_qa_fail": sum(v.get("qa") == "FAIL" for v in flagged),
        "label": len(label), "label_qa_fail": sum(v.get("qa") == "FAIL" for v in label),
        "label_overhead": {s: sum(v["stroke"] == s for v in label) for s in OVERHEAD_STROKES},
        "qa": cats.count("qa"), "quota": cats.count("quota"),
        "by_stroke": by_stroke,
        "thresholds": {"overhead_min_rise_m": OVERHEAD_MIN_RISE_M, "swing_min_speed_ms": SWING_MIN_SPEED_MS},
    }
    return summary, out


# --- character + moveset -------------------------------------------------------------------------------
def _stats(path: Path) -> dict | None:
    if not path.exists() or path.suffix.lower() != ".glb":
        return None
    try:
        return glb_stats(path)
    except (ValueError, OSError, struct.error) as e:  # struct.error: a GLB cut short (still being written)
        log(f"[yellow]{path.name}: {e}[/]")
        return None


def _character(cfg: Config, link: _Linker) -> dict:
    run = cfg.run
    refs = []
    for r in _maybe_json(run.refs) or []:
        refs.append({**r, "url": link(run.refs_dir / r["path"], f"character/refs/{r['path']}")})
    stylize = _maybe_json(run.sheet_dir / "stylize.json") or {}
    mesh = _maybe_json(run.mesh_dir / "mesh.json") or {}
    # mesh.json holds the views as they were fed to image-to-3D (side views re-named by what they show)
    views_src = mesh.get("views") or stylize.get("views") or {}
    views = {name: link(run.root / rel, f"character/views/{name}.png") for name, rel in views_src.items()}
    sheet = {
        "hero": link(run.sheet_dir / "hero.png", "character/sheet/hero.png"),
        "turnaround": link(run.sheet_dir / "turnaround.png", "character/sheet/turnaround.png"),
        "views": views,
        # key = the side of her the image shows; the file is where the image model drew it
        "view_files": {name: Path(rel).name for name, rel in views_src.items()},
        "prompts": {k: {"model": v.get("model"), "prompt": v.get("prompt")}
                    for k, v in stylize.items() if isinstance(v, dict) and "prompt" in v},
    }
    generators = {}
    for name, info in (mesh.get("generators") or {}).items():
        gdir = run.mesh_dir / name
        files = {key: _file(link, gdir / fname, f"character/mesh/{name}/{fname}")
                 for key, fname in (info.get("files") or {}).items()}
        stats = dict(info.get("stats") or {})
        for key, fname in (info.get("files") or {}).items():
            if key not in stats and (s := _stats(gdir / fname)):
                stats[key] = s
        generators[name] = {"endpoint": info.get("endpoint"), "error": info.get("error"),
                            "files": {k: v for k, v in files.items() if v}, "stats": stats}
    rig_stats = _stats(run.rig)
    return {
        "refs": refs,
        "sheet": sheet,
        "mesh": {"primary": mesh.get("primary"), "generators": generators,
                 "updated": _mtime(run.mesh_dir / "mesh.json")},
        "rig": {"url": link(run.rig, "character/rigged.glb"), "stats": rig_stats,
                "size_mb": _round(run.rig.stat().st_size / 1e6, 2) if run.rig.exists() else None},
        "report": _maybe_json(run.character_report),
        "height_m": cfg.character.height_m,
    }


def _checks(cfg: Config, link: _Linker, clip_ids: list[str]) -> list[dict]:
    d = cfg.run.checks_dir
    if not d.exists():
        return []
    out = []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in CHECK_EXTS or not p.is_file():
            continue
        rel = p.relative_to(d).as_posix()
        # `retarget` names its renders <clip id>_<frame>.<ext> (or <clip id>.<ext> for a whole clip)
        clip = next((c for c in sorted(clip_ids, key=len, reverse=True)
                     if p.stem == c or p.stem.startswith(f"{c}_")), None)
        tail = p.stem[len(clip) + 1:] if clip else ""
        out.append({"name": rel, "url": link(p, f"anim/checks/{rel}"), "clip": clip,
                    "frame": int(tail) if tail.isdigit() else None,
                    "kind": "video" if p.suffix.lower() in (".mp4", ".webm") else "image"})
    return out


def _moveset(cfg: Config, link: _Linker, checks: list[dict]) -> dict:
    run = cfg.run
    stats = _stats(run.moveset_glb)
    return {
        "url": link(run.moveset_glb, "anim/rybakina_moveset.glb"),
        "fbx": link(run.moveset_fbx, "anim/rybakina_moveset.fbx"),
        "stats": stats,
        "animations": (stats or {}).get("animations", []),
        "retarget": _maybe_json(run.retarget_report),
        "checks": checks,
        "updated": _mtime(run.moveset_glb),
    }


# --- artwork -------------------------------------------------------------------------------------------
def _artwork(cfg: Config, link: _Linker) -> dict | None:
    """The character sheets + moveset atlas from `artwork compose` (full PNG + its JPEG preview), the paint
    report and the game-ready downloads; None until the artwork exists."""
    run = cfg.run
    rep = _maybe_json(run.artwork_report)
    motion = _artwork_motion(cfg, link)
    if not isinstance(rep, dict):
        return {"motion": motion} if motion else None
    images = {}
    for key, path in (("sheet", run.character_sheet), ("sheet_render", run.character_sheet_render),
                      ("atlas", run.moveset_atlas)):
        info = rep.get({"sheet": "character_sheet", "sheet_render": "character_sheet_render",
                        "atlas": "moveset_atlas"}[key]) or {}
        full = _file(link, path, f"artwork/{path.name}")
        preview = run.artwork_dir / str(info.get("preview") or f"{path.stem}_preview.jpg")
        images[key] = {**full, "preview": link(preview, f"artwork/{preview.name}") or full["url"],
                       "size": info.get("size")} if full else None
    figures = rep.get("figures") or {}
    slots = {n: e["paint"] for n, e in figures.items() if isinstance(e.get("paint"), dict)}
    poses = _maybe_json(run.artwork_poses)
    sources = [{k: s.get(k) for k in ("slot", "clip", "frame", "at", "offset", "contact_index", "azimuth")}
               for s in (poses.get("slots") or [] if isinstance(poses, dict) else []) if s.get("clip")]
    return {
        **images,
        "painted": rep.get("painted") or [],
        "rendered": rep.get("rendered") or [],
        "silhouettes": sorted(n for n, e in figures.items() if e.get("source") == "silhouette"),
        "missing": rep.get("missing") or [],
        "sources": sources,
        "n_atlas": len(rep.get("atlas_figures") or {}),
        "n_flagged": sum(bool(e.get("flagged")) for e in (rep.get("atlas_figures") or {}).values()),
        "paint": {"model": (rep.get("inputs") or {}).get("paint_model") or cfg.models.image_paint,
                  "accept_iou": (rep.get("inputs") or {}).get("accept_iou"), "slots": slots} if slots else None,
        "downloads": {"glb": _file(link, run.moveset_glb, "anim/rybakina_moveset.glb"),
                      "fbx": _file(link, run.moveset_fbx, "anim/rybakina_moveset.fbx")},
        "motion": motion,
        "updated": _mtime(run.artwork_report),
    }


def _fresh(derived: Path, *sources: Path | None) -> bool:
    """A file made from others (a site copy of a clip video) exists and is not older than any of them."""
    return derived.exists() and all(s is None or not s.exists() or derived.stat().st_mtime >= s.stat().st_mtime
                                    for s in sources)


def _artwork_motion(cfg: Config, link: _Linker) -> dict | None:
    """The artwork in motion: the moveset reels from `artwork motion-compose` (painted + all-render, with their
    posters), the turntable, and every moveset clip's render video (`motion-render`) and accepted painted video
    (`motion-paint`) with its pose-check numbers. The grid plays the light site copies motion-compose writes
    (360 px); a clip rendered or painted again since falls back to the full video. None until any exists."""
    from .artwork.motion_compose import motion_clips, poster_path, resolve_path, tiles_dir

    run = cfg.run
    groups = motion_clips(cfg)
    renders = _maybe_json(run.motion_renders / "renders.json") or {}
    rep = _maybe_json(run.motion_report) or {}
    tt = renders.get("turntable") if isinstance(renders.get("turntable"), dict) else {}
    turntable_mp4 = resolve_path(run.root, run.motion_renders, tt.get("video"), run.motion_renders / "turntable.mp4")
    if not (any(c.render for _, cs in groups for c in cs) or run.moveset_reel_render.exists()
            or turntable_mp4.exists()):
        return None
    web = tiles_dir(cfg)
    base = "artwork/motion"

    def media(src: Path | None, rel: str) -> str | None:
        return link(src, f"{base}/{rel}") if src else None

    clips, sources = [], []
    for stroke, cs in groups:
        for c in cs:
            render = painted = None
            if c.render:
                sources.append(c.render)
                url = media(c.render, f"render/{c.id}.mp4")
                tile, tile_jpg = web / f"{c.id}.mp4", web / f"{c.id}.jpg"
                fresh = _fresh(tile, c.render) and tile_jpg.exists()
                render = {"url": url, "tile": media(tile, f"tiles/{c.id}.mp4") if fresh else url,
                          "poster": (media(tile_jpg, f"tiles/{c.id}.jpg") if fresh else None)
                          or media(c.poster, f"render/{c.id}.jpg"),
                          "poster_full": media(c.poster, f"render/{c.id}.jpg"),
                          "n_frames": c.n_frames, "contact_frame": c.contact}
            if c.painted:
                sources.append(c.painted)
                url = media(c.painted, f"painted/{c.id}.mp4")
                tile, tile_jpg = web / f"{c.id}_painted.mp4", web / f"{c.id}_painted.jpg"
                fresh = _fresh(tile, c.painted) and tile_jpg.exists()
                p = c.paint or {}
                painted = {"url": url, "tile": media(tile, f"tiles/{c.id}_painted.mp4") if fresh else url,
                           "poster": media(tile_jpg, f"tiles/{c.id}_painted.jpg") if fresh else None,
                           "iou_median": _round(p.get("iou_median")), "iou_p10": _round(p.get("iou_p10")),
                           "iou_min": _round(p.get("iou_min")), "model": p.get("model")}
            clips.append({"id": c.id, "stroke": stroke, "flagged": c.flagged, "render": render, "painted": painted,
                          "paint": c.paint if c.paint and not c.paint["accepted"] else None})

    def reel(path: Path, key: str) -> dict | None:
        f = _file(link, path, f"{base}/{path.name}")
        if not f:
            return None
        info = rep.get(key) or {}
        return {**f, "poster": link(poster_path(path), f"{base}/{poster_path(path).name}"),
                "size": info.get("size"), "frames": info.get("frames"), "duration_s": info.get("duration_s")}

    # motion-render's own poster, else the one motion-compose takes from the video's first frame
    tt_poster = resolve_path(run.root, run.motion_renders, tt.get("poster"), run.motion_renders / "turntable.jpg")
    tt_poster_url = (media(tt_poster, f"render/{tt_poster.name}") if tt_poster.exists()
                     else media(web / "turntable.jpg", "tiles/turntable.jpg")
                     if _fresh(web / "turntable.jpg", turntable_mp4) else None)
    paint_rep = _maybe_json(run.motion_painted / "paint.json") or {}
    reel_paths = [run.moveset_reel, run.moveset_reel_render]
    return {
        "reel": reel(run.moveset_reel, "reel"),
        "reel_render": reel(run.moveset_reel_render, "reel_render"),
        "turntable": {"url": media(turntable_mp4, "render/turntable.mp4"),
                      "poster": tt_poster_url,
                      "seconds": tt.get("seconds") or tt.get("duration_s")} if turntable_mp4.exists() else None,
        "clips": clips,
        "n_clips": len(clips),
        "n_rendered": sum(bool(c["render"]) for c in clips),
        "n_painted": sum(bool(c["painted"]) for c in clips),
        "n_flagged": sum(c["flagged"] for c in clips),
        "fps": rep.get("fps") or renders.get("fps"),
        "timing": rep.get("timing"),
        "in_reel": {"painted": rep.get("painted") or [], "rendered": rep.get("rendered") or [],
                    "missing": rep.get("missing") or []} if rep else None,
        # the reel shows what existed when it was composed; renders or repaints made since are not in it yet
        "reel_stale": bool(rep) and any(not _fresh(p, *sources) for p in reel_paths if p.exists()),
        "models": {"render": f"Blender {renders['blender']} Cycles" if renders.get("blender") else "Blender Cycles",
                   "image_paint": paint_rep.get("image_model") or cfg.models.image_paint,
                   "video_paint": paint_rep.get("model") or cfg.models.video_paint},
        "updated": _mtime(run.motion_report, run.motion_renders / "renders.json", run.motion_painted / "paint.json"),
    }


# --- character variants --------------------------------------------------------------------------------
# A variant is another character on the same captured motion (run.root/variants/<name>/, see RunLayout). The page
# shows its moveset in motion from the best render profile (artwork/layout.py PROFILES) that has anything: the
# Quest 3 deliverable, else the quarter-size draft of the same framing. A reel that holds the whole moveset beats a
# partial one, any reel report beats bare renders, and a clip's quest3 video replaces its draft as soon as it exists.
VARIANT_PROFILES = ("quest3", "draft")


def _first(*paths: Path | None) -> Path | None:
    return next((p for p in paths if p is not None and p.exists()), None)


def _report_path(root: Path, folder: Path, value: Any) -> Path | None:
    """A path from a report: run-root relative when it has a folder, else a name inside the stage's folder."""
    if not isinstance(value, str) or not value:
        return None
    return root / value if "/" in value else folder / value


def _rows(report: Any) -> dict[str, dict]:
    """renders.json / motion_<profile>.json clip rows as {clip id: row}, kept as a mapping or a list of rows."""
    clips = report.get("clips") if isinstance(report, dict) else None
    if isinstance(clips, list):
        return {r["id"]: r for r in clips if isinstance(r, dict) and isinstance(r.get("id"), str)}
    return {k: v for k, v in clips.items() if isinstance(v, dict)} if isinstance(clips, dict) else {}


def _variant_config(cfg: Config, variant: str) -> tuple[str | None, str | None]:
    """(display name, project-relative config path) of the configs/*.yaml that builds this variant, if any: one on
    this run folder first, else any with that variant name."""
    found = []
    for i, p in enumerate(sorted((PROJECT_ROOT / "configs").glob("*.yaml"))):
        try:
            c = Config.load(p)
        except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError):  # not a config (or a broken one)
            continue
        if c.character.variant == variant:
            found.append((c.run_dir.resolve() != cfg.run_dir.resolve(), i, c.display_name,
                          p.relative_to(PROJECT_ROOT).as_posix()))
    return min(found)[2:] if found else (None, None)


def _comparisons(cfg: Config, link: _Linker) -> list[dict]:
    """Side-by-side videos in <run>/compare/ (one clip rendered with two characters or settings), newest first; a
    <name>.json beside a video may give its "title" and "note"."""
    d = cfg.run.root / "compare"
    out = []
    for mp4 in sorted(d.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if d.is_dir() else []:
        meta = m if isinstance(m := _maybe_json(mp4.with_suffix(".json")), dict) else {}
        poster = mp4.with_suffix(".jpg")
        out.append({"id": mp4.stem, "title": meta.get("title") or mp4.stem.replace("_", " "),
                    "note": meta.get("note"), "video": _video(link, mp4, f"compare/{mp4.name}"),
                    "poster": link(poster, f"compare/{poster.name}") if poster.exists() else None})
    return out


def _variants(cfg: Config, link: _Linker) -> list[dict]:
    """Every character variant in the run folder with its moveset or anything in motion yet (a variant whose
    character is still being built has nothing to show here)."""
    d = cfg.run.root / "variants"
    names = sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith(".")) if d.is_dir() else []
    return [v for name in names if (v := _variant(cfg, link, name))]


def _variant(cfg: Config, link: _Linker, name: str) -> dict | None:
    """One variant's moveset in motion (`artwork motion-render` / `motion-compose --profile quest3|draft`).

    The live profile is the first of VARIANT_PROFILES whose reel report (motion_<profile>.json) holds every moveset
    clip, else the first with a reel report, else the first with renders: while quest3 renders batch by batch, the
    complete draft reel stays up. Its clips come from the report merged with renders.json, so clips rendered after
    the last compose show up too; each is linked as the full-resolution video + poster and, when motion-compose
    made one, the light tile + its poster (the grid only ever plays the tiles). A clip whose quest3 video already
    exists links that one instead of the live profile's (`profile` says which). Results arrive progressively:
    whatever exists is linked, the rest is None, and `progress` counts the renders of every profile."""
    run = RunLayout(cfg.run.root, name)
    root = run.root
    if not any(p.exists() for p in (run.moveset_glb, run.moveset_fbx, run.motion_dir)):
        return None
    label, config = _variant_config(cfg, name)

    def rel(src: Path) -> str:
        try:
            return src.relative_to(root).as_posix()
        except ValueError:  # a report pointing outside the run folder: still linked, under the variant
            return f"variants/{name}/{src.name}"

    def media(src: Path | None) -> str | None:
        return link(src, rel(src)) if src else None

    reports, renders, progress = {}, {}, {}
    for p in VARIANT_PROFILES:
        rdir = run.motion_renders_for(p)
        reports[p] = rep if isinstance(rep := _maybe_json(run.motion_dir / f"motion_{p}.json"), dict) else None
        renders[p] = r if isinstance(r := _maybe_json(rdir / "renders.json"), dict) else {}
        done = {cid for cid, row in _rows(renders[p]).items()
                if _first(_report_path(root, rdir, row.get("video") or row.get("mp4")), rdir / f"{cid}.mp4")}
        # a clip's video exists before renders.json lists it (the report is written when the whole run ends)
        done |= {f.stem for f in rdir.glob("*.mp4") if not f.stem.endswith("_matte")} if rdir.is_dir() else set()
        progress[p] = {"rendered": len(done), "composed": reports[p] is not None,
                       "size": renders[p].get("size") or (reports[p] or {}).get("size")}
    moveset = set(run.kept_clip_ids())
    whole = {p for p in VARIANT_PROFILES if reports[p] and moveset
             and moveset <= {cid for cid, r in _rows(reports[p]).items() if r.get("mp4")}}
    live = (next((p for p in VARIANT_PROFILES if p in whole), None)
            or next((p for p in VARIANT_PROFILES if reports[p]), None)
            or next((p for p in VARIANT_PROFILES if progress[p]["rendered"]), None))
    rep = reports.get(live) or {}
    rend = renders.get(live) or {}
    downloads = {"glb": _file(link, run.moveset_glb, rel(run.moveset_glb)),
                 "fbx": _file(link, run.moveset_fbx, rel(run.moveset_fbx)),
                 "animations": len(stats.get("animations") or []) if (stats := _stats(run.moveset_glb)) else None}
    out: dict[str, Any] = {
        "variant": name, "name": rep.get("name") or label or name.replace("_", " ").title(),
        "motion_from": rep.get("motion_from") or cfg.athlete.name, "config": config,
        # the clips' render size: the report's `size` is its reel's, the fixed 2064 x 2208 canvas even for the draft
        "profile": live, "composed": bool(rep), "size": rend.get("size") or rep.get("size"),
        "fps": rend.get("fps") or rep.get("fps"), "reel": None, "reel_stale": False, "groups": [], "clips": [],
        "n_clips": 0, "n_rendered": 0, "n_tiles": 0, "n_tiles_stale": 0, "n_outdated": 0, "n_flagged": 0,
        "n_in_reel": None,
        "progress": progress, "downloads": downloads, "made": rep.get("made"),
        "updated": _mtime(*(run.motion_dir / f"motion_{p}.json" for p in VARIANT_PROFILES),
                          *(run.motion_renders_for(p) / "renders.json" for p in VARIANT_PROFILES)),
    }
    if live is None:
        return out

    from .artwork.compose import _atlas_groups
    from .artwork.layout import PROFILES

    rdir = run.motion_renders_for(live)
    tdir = run.motion_dir / f"{run.stem}_moveset_tiles_{live}"
    rep_rows, rend_rows = _rows(rep), _rows(rend)
    # every moveset clip in the atlas order (strokes in ATLAS.stroke_order, clips in curation order), which is the
    # reel's: motion-compose lays out the same groups minus the clips it had no render for, so a clip rendered since
    # the last compose takes its place among them instead of trailing its group (or its stroke trailing the others).
    # The reel's stroke for a clip wins if it differs (curation changed since).
    atlas = _atlas_groups(cfg, dict.fromkeys([*run.kept_clip_ids(), *rend_rows, *rep_rows]))
    flagged = {c["id"]: c["flagged"] for _, cs in atlas for c in cs}
    stroke_of = {c["id"]: s for s, cs in atlas for c in cs}
    for g in rep.get("groups") or []:
        if isinstance(g, dict) and isinstance(g.get("clips"), list) and isinstance(g.get("stroke"), str):
            stroke_of.update({c: g["stroke"] for c in g["clips"] if isinstance(c, str)})
    rank = {s: i for i, (s, _) in enumerate(atlas)}
    groups: dict[str, list[str]] = {}
    for cid in stroke_of:
        groups.setdefault(stroke_of[cid], []).append(cid)
    groups = dict(sorted(groups.items(), key=lambda g: rank.get(g[0], len(rank))))

    size, fps = out["size"], out["fps"]
    best = VARIANT_PROFILES[0]
    best_dir, best_rows = run.motion_renders_for(best), _rows(renders.get(best) or {})
    clips, sources = [], []
    for stroke, ids in groups.items():
        for cid in ids:
            r, row = rep_rows.get(cid) or {}, rend_rows.get(cid) or {}
            mp4 = _first(_report_path(root, rdir, r.get("mp4")),
                         _report_path(root, rdir, row.get("video") or row.get("mp4")), rdir / f"{cid}.mp4")
            poster = _first(_report_path(root, rdir, r.get("poster")), _report_path(root, rdir, row.get("poster")),
                            rdir / f"{cid}.jpg")
            source = live
            hi = (_first(_report_path(root, best_dir, (best_rows.get(cid) or {}).get("video")), best_dir / f"{cid}.mp4")
                  if live != best else None)
            if hi:  # the deliverable is already there for this clip: link it, with its own facts
                source, mp4, row, r = best, hi, best_rows.get(cid) or {"size": progress[best]["size"]}, {}
                poster = _first(_report_path(root, best_dir, row.get("poster")), hi.with_suffix(".jpg")) or poster
            # a video older than the moveset shows an earlier rig (it is re-rendered from the current one): pending
            outdated = bool(mp4 and run.moveset_glb.exists() and not _fresh(mp4, run.moveset_glb))
            tile = _first(_report_path(root, tdir, r.get("tile")), tdir / f"{cid}.mp4")
            tile_poster = _first(_report_path(root, tdir, r.get("tile_poster")), tdir / f"{cid}.jpg")
            if outdated:
                mp4 = poster = tile = tile_poster = None
            f = _file(link, mp4, rel(mp4)) if mp4 else None
            sources += [mp4] if f else []
            clips.append({
                "id": cid, "stroke": r.get("stroke") or stroke, "profile": source if mp4 else None,
                "outdated": outdated,
                "flagged": bool(r["flagged"]) if "flagged" in r else flagged.get(cid, False),
                "mp4": f["url"] if f else None, "mb": f["size_mb"] if f else None,
                "poster": media(poster), "tile": media(tile), "tile_poster": media(tile_poster),
                # a tile older than its render shows an earlier render of the same clip: still played, but noted
                "tile_stale": bool(tile and mp4 and not _fresh(tile, mp4)),
                # the full-resolution video's own facts (renders.json describes the file linked as mp4, also after a
                # re-render the report has not seen), else the report's; frames at the profile's rate (60 fps)
                "size": row.get("size") or r.get("size") or (list(PROFILES[source].size) if source != live else size),
                "fps": row.get("fps") or r.get("fps") or (PROFILES[source].fps if source != live else fps),
                "frames": row.get("n_frames") or row.get("frames") or r.get("frames"),
                "contact_frame": next((x for x in (row.get("contact_frame"), r.get("contact_frame")) if x is not None),
                                      None),
            })
    reel_info = rep.get("reel") if isinstance(rep.get("reel"), dict) else {}
    reel_mp4 = _first(_report_path(root, run.motion_dir, reel_info.get("mp4")),
                      run.motion_dir / f"{run.stem}_moveset_reel_{live}.mp4")
    if reel_mp4:
        v = _video(link, reel_mp4, rel(reel_mp4)) or {}
        frames = v.get("frames") or reel_info.get("frames")
        reel_fps = reel_info.get("fps") or rep.get("fps") or fps
        loop = reel_info.get("loop_s") or (round(frames / reel_fps, 3) if frames and reel_fps else None)
        out["reel"] = {
            "url": v.get("url"), "poster": media(_first(_report_path(root, run.motion_dir, reel_info.get("poster")),
                                                         reel_mp4.with_suffix(".jpg"))),
            "size_mb": round(reel_mp4.stat().st_size / 1e6, 2),
            "size": [v["width"], v["height"]] if v.get("width") and v.get("height") else rep.get("size") or size,
            "frames": frames, "fps": reel_fps, "loop_s": loop,
        }
        # the reel holds what was rendered when it was composed
        in_reel = [cid for cid, r in rep_rows.items() if r.get("mp4")]
        out["n_in_reel"] = len(in_reel) if rep_rows else None
        out["reel_stale"] = (not _fresh(reel_mp4, *sources)
                             or (out["n_in_reel"] is not None and len(sources) > len(in_reel)))
    out.update(
        groups=[{"stroke": s, "clips": ids} for s, ids in groups.items()],
        clips=clips, n_clips=len(clips), n_rendered=sum(bool(c["mp4"]) for c in clips),
        n_hires=sum(c["profile"] == best for c in clips),
        n_tiles=sum(bool(c["tile"]) for c in clips), n_tiles_stale=sum(c["tile_stale"] for c in clips),
        n_outdated=sum(c["outdated"] for c in clips),
        n_flagged=sum(bool(c["flagged"]) for c in clips),
    )
    return out


# --- pipeline ------------------------------------------------------------------------------------------
def _gpu_pins() -> dict[str, str]:
    import tomllib

    try:
        project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]
        deps = project["optional-dependencies"]
    except (OSError, KeyError, ValueError):
        return {}
    pins = {}
    for spec in deps.get("gpu", []):
        name, _, ver = spec.partition("==")
        if ver:
            pins[name.split("[")[0]] = ver
    return pins


def _status(done: int, total: int) -> str:
    return "pending" if done == 0 else "done" if done >= total else "partial"


def _pipeline(cfg: Config, d: dict) -> list[dict]:
    run, m = cfg.run, cfg.models
    timings = _maybe_json(run.root / TIMINGS_FILE) or {}
    fo, clips, ch, cu, sig = d["footage"], d["clips"], d["character"], d["curation"], d["signature"]
    n = max(len(clips), 1)
    kept = [c for c in clips if c["curation"] and c["curation"]["kept"]] if cu else clips
    calib_views = [c for c in clips if (c["meta"] or {}).get("view") in CALIBRATED_VIEWS]
    count = {
        "preview": sum(bool(c["media"]["preview"]) for c in clips),
        "track": sum(bool(c["track"]) for c in clips),
        "motion": sum(bool(c["motion"]) for c in clips),
        "qa": sum(bool(c["qa"]) for c in clips),
        "anim": sum(bool(c["anim"]) for c in kept),
        "calib": sum(bool(c["camera_calib"]) for c in clips),
        "calib_used": sum(bool(c["motion"] and c["motion"]["calibrated"]) for c in clips),
    }
    # a court calibration only changes the motion once GVHMR is re-run on it
    calib_note = "".join([
        f" · {count['calib_used']} on court calibration" if count["calib_used"] else "",
        f" · {count['calib'] - count['calib_used']} calibrations not used yet"
        if count["calib"] > count["calib_used"] else "",
    ])
    skipped = len(clips) - len(calib_views)
    camera_breakdown = [f"{skipped} {'side' if skipped == 1 else 'other'}-view clip{'s' * (skipped != 1)} "
                        "not calibrated (by design)"] if skipped else []
    if len(calib_views) > count["calib"]:
        camera_breakdown.append(f"{len(calib_views) - count['calib']} below the fit bar")
    sel = _maybe_json(run.selection) or {}
    rig = ch["rig"]["stats"] or {}
    gvhmr = _gpu_pins().get("gvhmr")
    qa_sum = (_maybe_json(run.qa_summary) or {}).get("overall") or {}
    rt = d["moveset"]["retarget"] if isinstance(d["moveset"]["retarget"], dict) else {}
    timings.setdefault("retarget", rt.get("seconds"))
    roles = [r.get("role") for r in ch["refs"]]
    clip_files = [run.clip(c) for c in run.clip_ids()]
    if cu and cu["keep_all"]:
        cu_detail = f"all {cu['n']} clips kept"
        cu_breakdown = [f"{cu['flagged']} flagged: motion ≠ stroke label"
                        + (f", {cu['flagged_qa_fail']} of them also QA FAIL" if cu["flagged_qa_fail"] else ""),
                        "none dropped (curate.keep_all: the full range of her movement)"]
    elif cu:
        cu_detail = f"{cu['kept']} kept of {cu['n']}"
        cu_breakdown = [
            f"{cu['label']} rejected: label mismatch (motion ≠ stroke)"
            + (f", {cu['label_qa_fail']} of them also QA FAIL" if cu["label_qa_fail"] else ""),
            f"{cu['qa']} rejected on QA alone",
            f"{cu['quota']} valid, over the per-stroke quota",
        ]
    else:
        cu_detail, cu_breakdown = "pending", []
    if d["moveset"]["url"]:
        rt_detail = (f"{count['anim']} animations (all kept clips)" if cu and count["anim"] >= len(kept)
                     else f"{count['anim']}/{len(kept)} kept clips animated")
    else:
        rt_detail = "moveset pending"
    by_stroke = (sig or {}).get("by_stroke") or {}
    art = d["artwork"] or {}
    renders = _maybe_json(run.artwork_renders / "renders.json") or {}
    art_paint = art.get("paint") or {}
    if art.get("sheet"):
        art_detail = (f"{len(art['painted'])} painted figures · {art['n_atlas']} clips in the atlas"
                      + (f" · {len(art['rendered'])} left as renders" if art["rendered"] else ""))
    else:
        art_detail = "rendered, not composed yet" if renders else "pending"
    art_motion = art.get("motion") or {}
    art_breakdown = []
    if art_motion:
        art_breakdown.append(f"{art_motion['n_rendered']} clips in motion · {art_motion['n_painted']} painted")
        art_breakdown.append("moveset reel composed" if art_motion.get("reel") else "moveset reel pending")
    stages = [
        ("footage", "motion", "Footage", ["ffmpeg", "PySceneDetect adaptive"],
         _status(int(run.probe.exists()) + int(run.shots.exists()), 2),
         f"{fo['minutes'] or '?'} min · {fo['n_shots']} shots", [], [run.probe, run.shots],
         "probe, 16 kHz audio, 360p proxy, hard cuts"),
        ("index", "motion", "Gemini index", [m.index], _status(fo["n_labelled"], max(fo["n_shots"], 1)),
         f"{fo['n_labelled']} shots · {fo['n_strokes']} of her strokes", [], [run.index],
         "view, replay, court side, quality, strokes + contact times"),
        ("select", "motion", "Select", ["audio onsets", m.locate], _status(len(sel.get("clips", [])), 1),
         f"{len(sel.get('clips', []))} of {sel.get('n_candidates', '?')} candidates", [], [run.selection],
         "audio contact + hitter box, over-generated ×2 per stroke"),
        ("clips", "motion", "Clips", ["ffmpeg"], _status(count["preview"], n),
         f"{count['preview']}/{len(clips)} cut", [], [c.meta for c in clip_files],
         f"frame-exact, {cfg.clips.model_fps} fps model clip + 720p preview"),
        ("track", "motion", "Track", [cfg.track.detector, "ByteTrack"], _status(count["track"], n),
         f"{count['track']}/{len(clips)} tracked", [], [c.track for c in clip_files],
         "person tracks locked to the hitter box"),
        ("camera", "motion", "Camera", ["court-line calibration (ITF court model)"],
         _status(count["calib"], max(len(calib_views), 1)), f"{count['calib']}/{len(clips)} calibrated",
         camera_breakdown, [c.camera for c in clip_files],
         "per-frame focal length + camera pose from the painted court lines"),
        ("motion", "motion", "GVHMR", [f"GVHMR {gvhmr}" if gvhmr else "GVHMR", "ViTPose", "SimpleVO"],
         _status(count["motion"], n), f"{count['motion']}/{len(clips)} recovered{calib_note}", [],
         [c.motion for c in clip_files],
         "world-grounded SMPL-X on GVHMR's own world path; court camera, VO only when uncalibrated"),
        ("qa", "motion", "QA", ["skate · jitter · reprojection · timing"], _status(count["qa"], n),
         (f"{qa_sum.get('PASS', 0)} pass · {qa_sum.get('WARN', 0)} warn · {qa_sum.get('FAIL', 0)} fail"
          if qa_sum else f"{count['qa']}/{len(clips)} measured"), [],
         [run.qa_summary], "gated PASS / WARN / FAIL per clip, re-measured on the retarget"),
        ("curate", "motion", "Curate",
         ["motion vs label check", "QA gate", "keep all · flag" if cu and cu["keep_all"] else "per-stroke quota"],
         "done" if cu else "pending", cu_detail, cu_breakdown, [run.curation],
         f"serve: wrist ≥ {OVERHEAD_MIN_RISE_M:.2f} m over the head; groundstroke: hand ≥ "
         f"{SWING_MIN_SPEED_MS:.0f} m/s; "
         + ("every clip kept, failures flagged" if cu and cu["keep_all"] else "best per stroke kept")),
        ("retarget", "motion", "Retarget", [f"Blender {rt['blender']}" if rt.get("blender") else "Blender"],
         _status(count["anim"], max(len(kept), 1)), rt_detail, [],
         [run.moveset_glb, run.retarget_report], "kept clips → character rig, foot planting, racket, GLB + FBX"),
        ("signature", "motion", "Signature", ["SMPL joints · numpy"],
         "done" if by_stroke else "pending",
         (f"{len(by_stroke)} strokes · {sum(r.get('n', 0) for r in by_stroke.values())} label-checked clips"
          if by_stroke else "pending"), [], [run.signature],
         "per-stroke biomechanics: contact height, knee flex, coil, stance, toss"),
        ("refs", "character", "References", ["client deck", "Wikimedia Commons"], _status(len(ch["refs"]), 1),
         f"{roles.count('identity')} identity · {roles.count('style')} style", [], [run.refs],
         "identity photos + style targets"),
        ("stylize", "character", "Stylize", sorted({m.image_hero, m.image_views}),
         _status(sum(bool(v) for v in ch["sheet"]["views"].values()) + bool(ch["sheet"]["hero"]), 5),
         f"hero + {sum(bool(v) for v in ch['sheet']['views'].values())} views", [],
         [run.sheet_dir / "stylize.json"], "hero in the Osaka style → A-pose turnaround"),
        ("mesh", "character", "Mesh + rig", [m.mesh, m.mesh_alt, m.rig], _status(int(bool(rig)), 1),
         f"{rig.get('triangles', 0):,} tris · {len(rig.get('joints', []))} bones" if rig else "pending", [],
         [run.mesh_dir / "mesh.json", run.rig], "multi-view → quad mesh → humanoid rig (Tripo as A/B)"),
        ("artwork", "character", "Artwork",
         [f"Blender {renders['blender']} Cycles" if renders.get("blender") else "Blender Cycles",
          art_paint.get("model") or m.image_paint,
          *([art_motion["models"]["video_paint"]] if art_motion.get("n_painted") else [])],
         "done" if art.get("sheet") else "partial" if renders else "pending", art_detail, art_breakdown,
         [run.artwork_report, run.motion_report], "poses from her motion → Cycles renders → repainted in the "
                                                  "client's style (silhouette-checked) → character sheet + moveset "
                                                  "atlas; every clip in motion → the moveset reel"),
    ]
    return [{"id": sid, "lane": lane, "name": name, "models": models, "status": status, "detail": detail,
             "breakdown": breakdown, "what": what, "updated": _mtime(*outputs), "seconds": timings.get(sid)}
            for sid, lane, name, models, status, detail, breakdown, outputs, what in stages]


# --- notes ---------------------------------------------------------------------------------------------
def _notes(cfg: Config, d: dict) -> dict:
    """Limitations and next steps of the run as it stands, each derived from what the stages wrote."""
    clips, ch, mv, fo, cu = d["clips"], d["character"], d["moveset"], d["footage"], d["curation"]
    lim, nxt = [], []
    kept = [c for c in clips if c["curation"] and c["curation"]["kept"]] if cu else clips

    # what the footage could give: real-time, near-court, full-body shots only
    labels = [s["label"] for s in fo["shots"] if s["label"]]
    if labels:
        far = sum(lab["rybakina_side"] == "far" for lab in labels)
        her_usable = sum(1 for lab in labels if not lab["skip"] for e in lab["events"]
                         if e["hitter"] == "rybakina" and e["type"] in STROKES)
        lim.append(f"Only real-time, near-court, full-body shots are captured: replays ({fo['replays']} shots, "
                   f"{fo['slow_motion']} in slow motion), far-court ({far}) and close-up "
                   f"({fo['by_view'].get('closeup', {}).get('shots', 0)}) shots are excluded, leaving "
                   f"{fo['usable']}/{fo['n_shots']} shots with {her_usable} of her {fo['n_strokes']} strokes.")

    # curation of Gemini's labels on the motion
    if cu:
        rows = [c["curation"] for c in clips if c["curation"]]
        bad = [r for r in rows if r["category"] in ("label", "flagged")]
        serves = [r for r in bad if r["stroke"] == "serve"]
        other = len(bad) - len(serves)
        parts = []
        if serves:
            # the wrist apex is searched near the labelled contact only; the clips seen are her returns of serve
            parts.append(f"{len(serves)} 'serves' never lift her hitting wrist {OVERHEAD_MIN_RISE_M:.2f} m above "
                         "her head near the contact (likely returns of serve)")
        if other:
            parts.append(f"{other} other clip{'s' * (other != 1)} show no overhead reach or real swing")
        if bad:
            # QA FAIL counts the whole set of mislabelled clips, not one of the groups listed after it
            n_fail = sum(r.get("qa") == "FAIL" for r in bad)
            qa_note = f" ({n_fail} of them also QA FAIL)" if n_fail else ""
            found = (f"{len(bad)} clip{'s' * (len(bad) != 1)} contradict their label{qa_note}: "
                     f"{'; '.join(parts)}")
        else:
            found = "No clip contradicted its label"
        if cu["keep_all"]:
            moveset = (f"All {cu['n']} clips are kept in the moveset for the full range of her movement; the "
                       f"{cu['flagged']} flagged ones are marked on the Moveset page and left out of the stroke "
                       "signature.")
        else:
            moveset = (f"{cu['kept']}/{cu['n']} clips make the moveset ({cu['quota']} valid ones over the "
                       "per-stroke quota).")
        lim.append(f"Gemini's stroke labels are model output, so curate checks them on the 3D motion. {found}. "
                   + moveset)

    # camera: court calibration fed to GVHMR
    used = [c for c in clips if c["motion"] and c["motion"]["calibrated"]]
    uncal = [c for c in clips if c["motion"] and not c["motion"]["calibrated"]]
    if used:
        tail = ""
        if uncal:
            one = len(uncal) == 1
            ids = ", ".join(c["id"] for c in uncal[:3]) + ("…" if len(uncal) > 3 else "")
            views = "/".join(sorted({_view_label(c["meta"].get("view")) for c in uncal}))
            out = all(c["curation"] and not c["curation"]["kept"] for c in uncal) if cu else False
            vo = any(c["motion"]["static_camera"] is False for c in uncal)
            tail = (f"; the {len(uncal)} {views}-view clip{'' if one else 's'} ({ids}"
                    + (", rejected by curate" if out else "") + f") keep{'s' if one else ''} an assumed lens"
                    + (" and visual odometry" if vo else ""))
        lim.append(f"Court-line calibration (ITF court model) fixed the lens and camera pose of "
                   f"{len(used)}/{len(clips)} clips and GVHMR ran on it{tail}.")
    else:
        hfov = _median([c["motion"]["hfov_deg"] for c in clips if c["motion"]])
        if hfov is not None:
            lim.append(f"No court calibration in the motion yet: a {hfov:.0f}° lens is assumed per view.")

    # world path: GVHMR's own trajectory; depth is still the weak axis
    qa_all = (d["qa_summary"] or {}).get("overall") or {}
    if qa_all:
        share = qa_all.get("median_skate_depth_share")
        lim.append("The root follows GVHMR's own world path (velocity prior + foot contacts), not the noisy "
                   "in-camera depth; from one broadcast view depth is still the weak axis"
                   + (f" ({share * 100:.0f}% of the remaining foot skate runs along the camera ray)."
                      if isinstance(share, (int, float)) else "."))

    # foot planting at retarget
    rt = mv.get("retarget") if isinstance(mv.get("retarget"), dict) else {}
    settings, racket = rt.get("settings") or {}, rt.get("racket")
    per = _per_clip(rt)
    before = [e["skate_cm_per_frame"]["before"] for e in per.values() if isinstance(e, dict)
              and isinstance(e.get("skate_cm_per_frame"), dict)]
    after = [e["skate_cm_per_frame"]["after"] for e in per.values() if isinstance(e, dict)
             and isinstance(e.get("skate_cm_per_frame"), dict)]
    if settings.get("foot_lock"):
        skate = (f" (median foot skate {np.median(before):.2f} → {np.median(after):.2f} cm/frame)"
                 if before and after else "")
        low = [e["lowest_foot_cm"]["p5"] for e in per.values() if isinstance(e, dict)
               and isinstance((e.get("lowest_foot_cm") or {}).get("p5"), (int, float))]
        dip = (f"; with no heel/toe IK the lowest foot still dips {abs(np.median(low)):.0f} cm into the court "
               "(5th percentile)" if low and np.median(low) < -0.5 else "")
        lim.append(f"Feet are planted at retarget from GVHMR's contact probabilities{skate}{dip}.")
    else:
        nxt.append("Plant the feet at retarget from GVHMR's contact probabilities.")

    # hands and racket (retarget.json's "hands": the moveset's own finger bones and left-hand IK, when switched on)
    joints = (ch["rig"]["stats"] or {}).get("joints", [])
    hands = rt.get("hands") or {}
    finger_bones = int(hands.get("finger_bones") or 0)
    fingers = finger_bones > 0 or any(w in j.lower() for j in joints for w in ("finger", "thumb", "index"))
    left_ik = bool((hands.get("settings") or {}).get("left_hand_ik"))
    if racket:
        grip = (f"seated on the palm, {racket.get('grip_angle_deg', 0):.0f}° across it" if racket.get("grip") == "palm"
                else f"{racket.get('grip_angle_deg', 0):.0f}° grip")
        lim.append(f"The racket is a rigid prop on {racket.get('bone', 'the hand')} "
                   f"({racket.get('length_m', 0) * 100:.1f} cm, {grip}): it "
                   "follows GVHMR's wrist, so the racket face and forearm roll are not measured from the footage.")
    if joints and not fingers:
        lim.append(f"The Meshy auto-rig has {len(joints)} bones and no fingers: the grip is not articulated.")
    elif finger_bones:
        lim.append(f"The {finger_bones} finger bones added to the auto-rig hold one grip solved on the handle (the "
                   "left hand closes it where it holds the racket): the grip is not measured per stroke.")
    if cfg.athlete.backhand == "two_handed" and racket and not left_ik:
        lim.append("Two-handed backhand: the left hand follows GVHMR's left wrist and is not constrained to the "
                   "handle, so it can sit off the racket.")
    elif cfg.athlete.backhand == "two_handed" and racket:
        lim.append("Two-handed backhand: IK puts the left hand on the handle wherever the wrists are close; when it "
                   "takes hold is read from GVHMR's wrist distance, not seen in the footage.")
    lim.append("Hair, ponytail and dress are skinned to the body (no cloth or hair simulation).")

    # QA of what ships
    gates = [c["qa"]["status"] for c in kept if c["qa"]]
    if gates:
        ground = sum(1 for c in kept if c["qa"] and c["qa"]["status"] == "WARN"
                     and all(("ground" in r or "penetration" in r) for r in c["qa"]["reasons"] if r.startswith("WARN")))
        lim.append(f"QA on the {len(gates)} kept clips: {gates.count('PASS')} pass, {gates.count('WARN')} warn, "
                   f"{gates.count('FAIL')} fail"
                   + (f"; {ground} of the warnings are only GVHMR's ground height (offset or foot penetration "
                      "after re-grounding)." if ground else "."))
    if rt.get("problems"):
        lim.append(f"Retarget problems: {'; '.join(map(str, rt['problems']))}.")

    # next steps
    pending = [s["name"] for s in d["pipeline"] if s["status"] != "done"]
    if pending:
        nxt.append(f"Finish {', '.join(pending)}.")
    if not mv["url"]:
        nxt.append("Retarget the kept clips onto the rig and export the moveset (GLB + FBX).")
    unused = [c["id"] for c in clips if c["camera_calib"] and c["motion"] and not c["motion"]["calibrated"]]
    if unused:
        nxt.append(f"Re-run GVHMR on the court calibration of {len(unused)} clips, then retarget and QA.")
    if joints and not fingers:
        nxt.append("Add finger bones (or a hand rig) so the grip can be posed per stroke.")
    if cfg.athlete.backhand == "two_handed" and racket and not left_ik:
        nxt.append("Pin the left hand to the handle with IK on two-handed backhands.")
    if racket:
        nxt.append("Track the racket in the footage to drive the racket face instead of following the wrist.")
    have = {c["meta"].get("stroke") for c in kept}
    missing = [s for s, k in cfg.select.strokes.items() if k and s not in have]
    if missing:
        # count from the index itself: fo["strokes"] only tracks STROKES, not e.g. celebrations
        hers = [(e["type"], s["label"]["skip"] is None) for s in fo["shots"] if s["label"]
                for e in s["label"]["events"] if e["hitter"] == "rybakina" and e["type"] in missing]
        labelled = ", ".join(f"{sum(t == s for t, _ in hers)} {s}" for s in missing)
        usable_n = sum(u for _, u in hers)
        cut = [c for c in clips if c["meta"].get("stroke") in missing]
        tail = (f"; {len(cut)} clip{'s' * (len(cut) != 1)} cut from them, rejected by curate" if cut else "")
        nxt.append(f"No kept clips for {', '.join(missing)}: this compilation labels only {labelled} of hers, "
                   f"{usable_n} in usable shots{tail}. Add footage.")
    nxt.append("Blend the strokes into idle, ready and recovery loops for the game.")
    if d.get("signature"):
        nxt.append("Run another player through the same pipeline and compare stroke signatures.")
    return {"limitations": lim, "next": nxt}


VIEW_LABELS = {"main_broadcast": "main broadcast", "behind_low": "behind, low", "side": "side",
               "closeup": "close-up", "other": "other"}


def _view_label(view: str | None) -> str:
    return VIEW_LABELS.get(view or "", view or "unknown")


def _median(values: list[float]) -> float | None:
    return _round(np.median(values), 0) if values else None


# --- build ---------------------------------------------------------------------------------------------
def _copy_web(site: Path) -> int:
    """web/ → site/, dropping front-end files a previous build copied that web/ no longer has."""
    rels = {p.relative_to(WEB_DIR) for p in WEB_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")}
    for p in site.rglob("*"):
        rel = p.relative_to(site)
        if rel.parts[0] not in ("data", "media") and p.is_file() and not p.is_symlink() and rel not in rels:
            p.unlink()
    for rel in rels:
        dst = site / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(WEB_DIR / rel, dst)
    return len(rels)


def build(cfg: Config) -> Path:
    if cfg.character.variant:  # the shared site would show the variant's character as the run's
        raise ValueError(f"the review site is the run's: build it with the base config, not the variant "
                         f"{cfg.character.variant!r} (its section is read from variants/ either way)")
    run = cfg.run
    site = run.site_dir
    run.ensure(site / "data")
    link = _Linker(site)
    with step("site"):
        n_web = _copy_web(site)
        mv_stats = _stats(run.moveset_glb)
        checks = _checks(cfg, link, run.clip_ids())
        rt = _maybe_json(run.retarget_report)
        cur_summary, cur_rows = _curation(cfg)
        sig = _maybe_json(run.signature)
        sig = sig if isinstance(sig, dict) and sig.get("by_stroke") else None
        clips = _clips(cfg, link, site, (mv_stats or {}).get("animations", []), rt, checks, cur_rows,
                       (sig or {}).get("per_clip") or {})
        metas = {c["id"]: c["meta"] for c in clips if c["meta"]}
        d: dict[str, Any] = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "athlete": cfg.athlete.model_dump(),
            "footage": _footage(cfg, metas),
            "selection": {**{k: v for k, v in (_maybe_json(run.selection) or {}).items() if k != "clips"},
                          "config": cfg.select.model_dump(mode="json")},
            "clips": clips,
            "character": _character(cfg, link),
            "moveset": _moveset(cfg, link, checks),
            "artwork": _artwork(cfg, link),
            "variants": _variants(cfg, link),
            "comparisons": _comparisons(cfg, link),
            "qa_summary": _maybe_json(run.qa_summary),
            "curation": cur_summary,
            "signature": sig,
            "config": {"models": cfg.models.model_dump(), "clips": cfg.clips.model_dump(),
                       "track": cfg.track.model_dump(), "motion": cfg.motion.model_dump(),
                       "character": cfg.character.model_dump(), "retarget": cfg.retarget.model_dump()},
        }
        d["pipeline"] = _pipeline(cfg, d)
        d["notes"] = _notes(cfg, d)
        write_json(site / "data" / "site.json", _finite(d))
        link.prune()
        n_motion = sum(bool(c["motion"]) for c in clips)
        kept = f", {cur_summary['kept']} kept by curate" if cur_summary else ""
        variants = "".join(f", {v['name']}: {v['n_rendered']}/{v['n_clips']} clips ({v['profile'] or 'pending'})"
                           for v in d["variants"])
        log(f"{n_web} web files, {link.count} media links, {len(clips)} clips ({n_motion} with motion{kept}), "
            f"{d['footage']['n_shots']} shots{variants} → {site}")
    return site


# --- serve ---------------------------------------------------------------------------------------------
MIME = {
    ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".fbx": "application/octet-stream",
    ".mp4": "video/mp4", ".webm": "video/webm", ".js": "text/javascript", ".mjs": "text/javascript",
    ".json": "application/json", ".css": "text/css", ".html": "text/html", ".svg": "image/svg+xml",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".gif": "image/gif", ".wasm": "application/wasm",
}


def _parse_range(spec: str, size: int) -> tuple[int, int] | None:
    """First range of a 'bytes=a-b' header as inclusive offsets, None if unsatisfiable."""
    unit, _, ranges = spec.partition("=")
    if unit.strip() != "bytes" or not ranges:
        return None
    first, _, last = ranges.split(",")[0].strip().partition("-")
    try:
        if first == "":  # suffix range: the last N bytes
            length = int(last)
            if length <= 0:
                return None
            return max(0, size - length), size - 1
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    except ValueError:
        return None
    return (start, end) if start <= end and start < size else None


class _Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **MIME}
    protocol_version = "HTTP/1.1"  # keep-alive: Chrome issues many small range requests per video

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")  # revalidate: a rebuild must show up on reload
        super().end_headers()

    def send_head(self):
        self._range = None
        spec = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not spec or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")  # noqa: SIM115 - handed to copyfile, closed by the base class
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        st = os.fstat(f.fileno())
        rng = _parse_range(spec, st.st_size)
        if rng is None:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{st.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        start, end = rng
        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{st.st_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(st.st_mtime))
        self.end_headers()
        self._range = (start, end)
        return f

    def copyfile(self, source, outputfile) -> None:
        if getattr(self, "_range", None) is None:
            return super().copyfile(source, outputfile)
        remaining = self._range[1] - self._range[0] + 1
        while remaining > 0:
            buf = source.read(min(256 * 1024, remaining))
            if not buf:
                break
            outputfile.write(buf)
            remaining -= len(buf)

    def log_request(self, code="-", size="-") -> None:
        if isinstance(code, int) and code >= 400:
            log(f"[yellow]{code}[/] {self.command} {self.path}")

    def log_message(self, format, *args) -> None:  # noqa: A002 - base-class signature
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # a <video> drops its connection whenever it seeks: not an error worth a traceback
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def serve(cfg: Config, port: int) -> None:
    site = cfg.run.site_dir
    if not (site / "data" / "site.json").exists():
        build(cfg)
    httpd = _Server(("127.0.0.1", port), partial(_Handler, directory=str(site)))
    log(f"review site → [bold]http://127.0.0.1:{port}/[/]  ({site}; Ctrl-C to stop)")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        log("review site stopped")
