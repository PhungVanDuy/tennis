"""Stage `retarget`: every recovered motion onto the rigged character, in Blender, as one moveset.

The driver writes a JSON job and runs blender/retarget.py headless (binary: $R3D_BLENDER, else
cfg.remote.blender when it exists, else `blender`). Inside Blender: SMPL -> rig retargeting (world-rotation
transfer with rest-pose alignment), a procedural racket in the right hand, one action per clip named by the
clip id, GLB + FBX export, the character's joint tracks for `qa` (anim/joints/<id>.npz) and check stills
from each clip's own broadcast camera (recovered from GVHMR's world/camera joints) plus a side view.
A second, fresh Blender re-imports the GLB and FBX: every clip must replay its joint track and carry the
racket in the right hand where the build put it. The stills are then put side by side with the matching
preview.mp4 frame: anim/checks/<id>_<frame>.jpg.

The moveset holds every clip `curate` kept that has a usable motion.npz; `--clip` only limits the
check renders.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from .blender import SCRIPTS_DIR
from .config import Config, RunLayout
from .gltf import glb_stats
from .media import read_frame
from .util import log, read_json, step, write_json

# Check stills at these frame offsets from contact for every clip of the stroke (others: contact only).
# Serve: trophy position, racket drop, contact, follow-through.
EXTRA_CHECKS = {"serve": (-20, -8, 0, 8), "forehand": (-10, 0, 6)}
PANEL = 360          # px, side of each panel in the check sheets
OVERLAY_ALPHA = 0.6  # character opacity over the broadcast frame
SAVE_BLEND = True    # anim/rybakina_moveset.blend for inspection (never pulled to the workstation)
FOOT_LOCK = True     # planted feet pin the root path (always on; not a config option)
MOTION_KEYS = ("fps", "global_orient", "body_pose", "joints_world", "rest_joints", "joints_camera", "K")
# re-import tolerances: joint replay, racket placement at rest, racket drift on the hand while playing
JOINT_TOL_MM, RACKET_TOL_MM, RACKET_TOL_DEG, DRIFT_TOL_MM = 1.0, 2.0, 0.5, 1.0


def blender_binary(cfg: Config) -> str:
    if os.environ.get("R3D_BLENDER"):
        return os.environ["R3D_BLENDER"]
    if Path(cfg.remote.blender).exists():
        return cfg.remote.blender
    return shutil.which("blender") or "blender"


def _check_frames(meta: dict, n: int) -> list[int]:
    contact = int(meta.get("contact_index", n // 2))
    offsets = (0, *EXTRA_CHECKS.get(meta.get("stroke"), ()))
    return sorted({int(np.clip(contact + d, 0, n - 1)) for d in offsets})


def motion_problem(path: Path) -> str | None:
    """Why a motion.npz cannot be retargeted (missing keys, too short, NaNs), or None."""
    try:
        m = np.load(path)
        missing = [k for k in MOTION_KEYS if k not in m]
        if missing:
            return f"lacks {', '.join(missing)}"
        n = len(m["global_orient"])
        tracks = [k for k in ("body_pose", "joints_world", "joints_camera", "K", "static_conf") if k in m]
        if n < 2 or any(len(m[k]) != n for k in tracks):
            return f"{n} frames or mismatched track lengths"
        for k in ("global_orient", "body_pose", "joints_world", "rest_joints", "joints_camera", "K"):
            if not np.isfinite(m[k]).all():
                return f"non-finite values in {k}"
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        return f"unreadable ({e})"
    return None


def _clips(cfg: Config, only: list[str] | None) -> list[dict]:
    clips = []
    for cid in cfg.run.kept_clip_ids():
        clip = cfg.run.clip(cid)
        if not clip.motion.exists():
            log(f"[yellow]{cid}: no motion.npz, skipped[/]")
            continue
        problem = motion_problem(clip.motion)
        if problem:
            log(f"[yellow]{cid}: motion.npz {problem}, skipped[/]")
            continue
        meta = read_json(clip.meta)
        n = int(np.load(clip.motion)["global_orient"].shape[0])
        clips.append({
            "id": cid, "motion": str(clip.motion), "stroke": meta.get("stroke", ""), "view": meta.get("view"),
            "contact_index": int(meta.get("contact_index", n // 2)), "n_frames": int(meta.get("n_frames", n)),
            "clip_size": list(meta.get("size", (1920, 1080))),
            "check_frames": _check_frames(meta, n) if not only or cid in only else [],
        })
    return clips


def _blender(binary: str, job: Path, log_path: Path, *extra: str) -> None:
    cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(SCRIPTS_DIR / "retarget.py"), "--", str(job), *extra]
    if not shutil.which(binary):
        raise FileNotFoundError(f"no Blender at {binary!r} (set R3D_BLENDER or remote.blender)")
    with open(log_path, "a") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line)
            if line.startswith("[r3d]"):
                log(line[6:].rstrip())
        code = proc.wait()
    if code:
        tail = log_path.read_text().splitlines()[-25:]
        raise RuntimeError(f"blender failed ({code}); last lines of {log_path}:\n" + "\n".join(tail))


def _label(img: np.ndarray, text: str) -> None:
    import cv2

    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (0, 0), (w + 12, h + 12), (30, 30, 30), -1)
    cv2.putText(img, text, (6, h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _over(rgba: np.ndarray, bg: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float32) / 255 * alpha
    return (rgba[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def _crop_box(box: np.ndarray | None, mask: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Square crop around the tracked player and the rendered character (racket included)."""
    h, w = shape
    ys, xs = np.nonzero(mask)
    boxes = [] if box is None else [box]
    if len(xs):
        rb = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float64)
        if box is None or (rb[3] - rb[1]) < 3 * (box[3] - box[1]):  # ignore a render that is way off
            boxes.append(rb)
    if not boxes:
        return 0, 0, w, h
    b = np.array(boxes)
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = min(max(x1 - x0, y1 - y0) * 1.25 + 20, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return x0, y0, x0 + int(side), y0 + int(side)


def compose_check(cfg: Config, cid: str, frame: int, broadcast: str, side: str, out: Path) -> Path:
    """[broadcast frame | character from the same camera | overlay | side view] as one JPEG."""
    import cv2

    clip = cfg.run.clip(cid)
    meta = read_json(clip.meta)
    prev = cv2.cvtColor(read_frame(clip.preview, frame, float(meta["play_fps"])), cv2.COLOR_RGB2BGR)
    h, w = prev.shape[:2]
    cam = cv2.imread(broadcast, cv2.IMREAD_UNCHANGED)
    if cam.shape[:2] != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_AREA)
    box = None
    if clip.track.exists():
        tr = np.load(clip.track)
        if frame < len(tr["bbx_xyxy"]) and bool(tr["valid"][frame]):
            box = tr["bbx_xyxy"][frame].astype(np.float64) * (h / meta["size"][1])
    x0, y0, x1, y1 = _crop_box(box, cam[..., 3] > 0, (h, w))

    def panel(img: np.ndarray) -> np.ndarray:
        return cv2.resize(img[y0:y1, x0:x1], (PANEL, PANEL), interpolation=cv2.INTER_AREA)

    grey = np.full((h, w, 3), 96, np.uint8)
    sv = cv2.imread(side, cv2.IMREAD_UNCHANGED)
    sv = cv2.resize(_over(sv, np.full(sv.shape[:2] + (3,), 225, np.uint8)), (PANEL, PANEL),
                    interpolation=cv2.INTER_AREA)
    panels = [panel(prev), panel(_over(cam, grey)), panel(_over(cam, prev, OVERLAY_ALPHA)), sv]
    for p, text in zip(panels, (f"{cid}  f{frame}", "retarget", "overlay", "side"), strict=True):
        _label(p, text)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


def _clean(run_: RunLayout, clips: list[dict]) -> None:
    """Drop the joint tracks and check sheets of clips no longer in the run, and the sheets about to be
    re-rendered."""
    ids = {c["id"] for c in clips}
    rerender = {c["id"] for c in clips if c["check_frames"]}
    for f in run_.joints_dir.glob("*.npz"):
        if f.stem not in ids:
            f.unlink()
    for f in run_.checks_dir.glob("*.jpg"):
        cid = f.stem.rsplit("_", 1)[0]
        if cid not in ids or cid in rerender:
            f.unlink()
    shutil.rmtree(run_.checks_dir / "raw", ignore_errors=True)


def _untracked_warnings(cfg: Config, report: dict) -> None:
    """Frames where `track` lost the player: GVHMR still fills them, but with a guess (whip pans)."""
    for cid, c in report["clips"].items():
        track = cfg.run.clip(cid).track
        if not track.exists():
            continue
        valid = np.asarray(np.load(track)["valid"], bool)
        if not valid.all():
            lost = np.flatnonzero(~valid)
            runs = np.split(lost, np.flatnonzero(np.diff(lost) > 1) + 1)
            spans = ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else f"{r[0]}" for r in runs)
            c["warnings"].append(f"player untracked on frames {spans}: the motion there is GVHMR's guess")


def verify_problems(verify: dict, report: dict, expected: list[str]) -> list[str]:
    """What the re-import check found wrong with the exported GLB (empty = all good)."""
    problems = []
    if not verify.get("glb_names_match"):
        problems.append(f"re-imported actions {verify.get('glb_actions')} != clips {expected}")
    hand = report["mapping"].get("right_wrist")
    if report.get("racket") and verify.get("glb_racket_parent") != f"BONE:{hand}":
        problems.append(f"racket parent {verify.get('glb_racket_parent')} != BONE:{hand}")
    rest = verify.get("glb_racket_rest")
    if rest and (rest["butt_mm"] > RACKET_TOL_MM
                 or max(rest["axis_deg"], rest["normal_deg"]) > RACKET_TOL_DEG):
        problems.append(f"re-imported racket is not where the build put it: {rest}")
    for cid, c in verify.get("glb_clips", {}).items():
        frames = report["clips"][cid]["frames"]
        if c["frames"] != frames:
            problems.append(f"{cid}: {c['frames']} frames in the GLB, {frames} retargeted")
        if c["joint_error_mm"] > JOINT_TOL_MM:
            problems.append(f"{cid}: GLB replays its joint track {c['joint_error_mm']} mm off")
        if c.get("racket_drift_mm", 0.0) > DRIFT_TOL_MM:
            problems.append(f"{cid}: racket drifts {c['racket_drift_mm']} mm on the hand")
    return problems


def leg_settings(cfg: Config) -> dict:
    """The job settings of retarget.leg_weights (none for "meshy", so such a job is the one from before it)."""
    r = cfg.retarget
    return {"leg_weights": r.leg_weights, "knee_band_cm": r.knee_band_cm} if r.leg_weights == "band" else {}


def run(cfg: Config, only: list[str] | None = None) -> dict:
    run_ = cfg.run
    if not run_.rig.exists():
        raise FileNotFoundError(f"no rigged character at {run_.rig} (run `r3d character mesh`)")
    clips = _clips(cfg, only)
    if not clips:
        log("[yellow]no clip has a usable motion.npz yet; nothing to retarget[/]")
        return {}
    unknown = sorted(set(only or ()) - {c["id"] for c in clips})
    if unknown:
        log(f"[yellow]--clip {unknown}: no such clip with motion[/]")
    anim = run_.anim_dir
    checks_dir = run_.checks_dir
    run_.ensure(anim, checks_dir)
    _clean(run_, clips)

    preview_h = cfg.clips.preview_height
    size = clips[0]["clip_size"]
    fps = float(np.load(clips[0]["motion"])["fps"])
    job = {
        "rig": str(run_.rig),
        "fps": fps,
        "settings": {"smooth_window": cfg.retarget.smooth_window,
                     "ground_from_contacts": cfg.retarget.ground_from_contacts, "foot_lock": FOOT_LOCK,
                     "racket": True, "weld_m": cfg.retarget.weld_m,
                     "smooth_limb_weights": cfg.retarget.smooth_limb_weights,
                     **({"racket_colors": cfg.character.racket_colors} if cfg.character.racket_colors else {}),
                     **leg_settings(cfg)},
        "render": {"width": int(round(size[0] * preview_h / size[1] / 2) * 2), "height": preview_h},
        "clips": clips,
        "out": {"glb": str(run_.moveset_glb), "fbx": str(run_.moveset_fbx),
                "report": str(run_.retarget_report), "joints_dir": str(run_.joints_dir),
                "checks_raw": str(checks_dir / "raw"), "verify": str(anim / "verify.json"),
                "blend": str(run_.moveset_blend) if SAVE_BLEND else None},
    }
    job_path = write_json(anim / "job.json", job)
    log_path = anim / "blender.log"
    log_path.unlink(missing_ok=True)
    binary = blender_binary(cfg)
    with step(f"blender retarget ({len(clips)} clips, {binary})"):
        _blender(binary, job_path, log_path)
    with step("re-import check"):
        _blender(binary, job_path, log_path, "--verify")

    report = read_json(run_.retarget_report)
    verify = read_json(anim / "verify.json")
    _untracked_warnings(cfg, report)
    checks = []
    for r in report.pop("renders", []):
        out = compose_check(cfg, r["clip"], r["frame"], r["broadcast"], r["side"],
                            checks_dir / f"{r['clip']}_{r['frame']:03d}.jpg")
        checks.append(str(out.relative_to(run_.root)))
    shutil.rmtree(checks_dir / "raw", ignore_errors=True)

    stats = glb_stats(run_.moveset_glb)
    expected = sorted(c["id"] for c in clips)
    problems = verify_problems(verify, report, expected)
    if sorted(stats["animations"]) != expected:
        problems.append(f"GLB animations {stats['animations']} != clips {expected}")
    report.update({
        "glb": {k: stats[k] for k in ("triangles", "meshes", "materials", "textures", "skins", "animations",
                                      "size_mb")},
        "fbx_size_mb": round(run_.moveset_fbx.stat().st_size / 1e6, 2),
        "verify": verify, "checks": sorted(str(p.relative_to(run_.root)) for p in checks_dir.glob("*.jpg")),
        "problems": problems, "settings": job["settings"],
    })
    write_json(run_.retarget_report, report)
    n_warn = sum(bool(c["warnings"]) for c in report["clips"].values())
    log(f"moveset: {len(stats['animations'])} animations, {stats['triangles']} tris, "
        f"{stats['size_mb']} MB GLB, {report['fbx_size_mb']} MB FBX; {len(checks)} check sheets rendered; "
        f"{n_warn} clip(s) with warnings")
    if problems:
        raise RuntimeError("; ".join(problems))
    return report
