"""Stage `qa`: quality metrics of every recovered (and retargeted) motion, gated PASS / WARN / FAIL.

Per clip with a motion.npz (see motion/recover.py for its contract):
  skate         horizontal speed of a foot joint while GVHMR calls it planted (static_conf > 0.5), cm/frame,
                also split along / across the camera ray through that foot (see below);
  ground        sole height = foot joint height minus its rest height above the sole: penetration below
                y=0, floating while planted, and the clip's ground offset (median planted clearance);
  jitter        mean norm of the third finite difference, m/s^3: all world joints, wrists, the root alone
                and the pose relative to the root;
  reprojection  joints_camera projected with K against ViTPose (conf > 0.5): px, / box height, PCK@0.1
                (GVHMR is fitted to these same keypoints: this checks consistency, not accuracy);
  timing        frame of peak hitting-wrist speed (relative to the pelvis) minus the contact frame, and
                whether that wrist swings at all around the contact;
  tracking      hitter-track coverage;  camera: static / background motion / calibrated;
  retarget      skate of the character's feet (anim/joints/<id>.npz) during the same planted frames, and
                the jitter of its root and of its pose relative to the root.

Monocular depth is the weak axis on broadcast footage: a player 20 m from a telephoto camera changes size
by ~0.1 px per 10 cm of depth. When the world root follows the in-camera translation (GVHMR's static-camera
`world_from_incam`), that noise lands in the world path along the camera ray, so the skate and root jerk are
also reported by direction, and the gate fails a clip only on what a root-depth fix (GVHMR's own trajectory,
court calibration, foot locking at retarget) cannot repair: skate across the ray, pose jitter, 2D fit, timing,
tracking. Depth-only problems warn; the retargeted skate, once it exists, is gated in full.

Writes clips/<id>/qa.json ({metrics, status, reasons, notes}) and <run>/qa_summary.json (thresholds, one
row per clip, aggregates per stroke). Pure numpy: runs on the workstation or the GPU host.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .motion.smpl import JOINTS, L_ANKLE, L_FOOT, L_WRIST, PELVIS, R_ANKLE, R_FOOT, R_WRIST, STATIC_JOINTS
from .util import console, log, read_json, read_npz, write_json

# --- measurement conventions -----------------------------------------------------------------------------
CONTACT_P = 0.5   # static_conf above this = planted
KP_CONF = 0.5     # ViTPose keypoints below this are mostly guesses (blur, occlusion by the racket/body)
PCK_ALPHA = 0.1   # PCK radius as a fraction of the hitter's box height

# (name, static_conf channel, SMPL joint) of the four foot joints GVHMR scores
FOOT_CHANNELS = tuple((name, ch, JOINTS.index(name)) for ch, name in enumerate(STATIC_JOINTS[:4]))
# Height of the SMPL ankle / foot joints above the lowest sole vertex in the zero pose, measured with the
# body model on this run's betas (ankle 6.9-7.3 / 6.1-6.4 cm, foot 1.5 / 1.6-1.7 cm). Subtracting it turns
# a joint height into an estimate of where the sole is, so a planted foot reads ~0 instead of ~7 cm.
SOLE_OFFSET_M = {L_ANKLE: 0.071, R_ANKLE: 0.063, L_FOOT: 0.015, R_FOOT: 0.017}
WRISTS = {"left": L_WRIST, "right": R_WRIST}
# ViTPose COCO-17 keypoint -> SMPL joint of the same name (the face keypoints have no SMPL joint)
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle"]
COCO_TO_SMPL = {k: JOINTS.index(name) for k, name in enumerate(COCO17) if name in JOINTS}
PARTS = {"arms": (5, 6, 7, 8, 9, 10), "legs": (11, 12, 13, 14, 15, 16)}   # COCO-17 keypoints per body part
STROKES = {"serve", "forehand", "backhand", "slice", "volley", "overhead"}
OVERHEAD = {"serve", "overhead"}   # contact at the top of the reach: the wrist apex is a second marker
# Retargeted joints in the same channel order as FOOT_CHANNELS, per naming scheme (Meshy rig, SMPL).
RIG_FEET = {
    "meshy": ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"),
    "smpl": ("left_ankle", "left_foot", "right_ankle", "right_foot"),
}
RIG_ROOT = {"meshy": ("Hips",), "smpl": ("pelvis",)}

# --- gate thresholds: (warn, fail); a clip's status is its worst rule ------------------------------------
# Foot skate, cm/frame at 25 fps, mean / p95 over planted steps. GVHMR's own world trajectory (static-joint
# post-processing on) gives 0.2-0.5 on three of this run's static-camera clips; 1.5 (0.4 m/s) is a slide you
# see on the character, 3 is one the retargeter's foot lock has to fight. Gated on the across-the-ray part
# for the raw motion (depth noise warns instead) and in full after retarget.
SKATE_MEAN_CM = (1.5, 3.0)
SKATE_P95_CM = (4.0, 8.0)
# Mean jerk of the pose relative to the pelvis, m/s^3. The strokes are jerky by nature: clean clips of this
# run sit at 135-385, so 400 is visible buzz and 800 means pops (lost limbs, left/right swaps).
POSE_JITTER_MS3 = (400.0, 800.0)
# Mean jerk of the root (pelvis) alone; warn-only, it is depth noise the retarget has to smooth away.
# GVHMR's own trajectory measures 180-230 here, the in-camera one 900-8500.
ROOT_JITTER_WARN_MS3 = 500.0
# Median joint error / box height. SMPL hip and ankle joints sit a few % of the height away from where COCO
# annotates them, so ~3-4 % is the floor even for a perfect fit; 10 % means a limb is in the wrong place.
REPROJ_NORM = (0.06, 0.10)
# Fraction of confident keypoints within 0.1 x box height (higher is better).
PCK_MIN = (0.85, 0.65)
# |peak wrist speed - contact| in frames (25 fps: 3 = 120 ms, 8 = 320 ms). The racket wrist is fastest
# within ~1-2 frames of contact; far off means the swing or the contact label is wrong.
TIMING_FRAMES = (3, 8)
# Peak hitting-wrist speed relative to the pelvis within TIMING_FRAMES[1] of the contact, m/s (higher is
# better). This run's real strokes reach 3.5 (slice) to 8; below 2 the player waits while the opponent hits,
# the contact label is far from her swing, or GVHMR barely moved the arm (a scrambled defensive shot).
SWING_MIN_MS = (3.0, 2.0)
# Tracking coverage below the config's min_coverage warns; below this the boxes are mostly interpolated.
COVERAGE_FAIL = 0.5
# Ground rules only warn: retarget re-grounds the motion from its contacts (retarget.ground_from_contacts),
# so a constant offset is repaired downstream; what is left after re-grounding is reported separately.
GROUND_OFFSET_CM = 8.0
GROUND_PEN_CM = 5.0
# Fewer planted steps than this: no foot lock at retarget, and the skate is not measurable.
MIN_CONTACT_STEPS = 3
# Background motion (px/frame) above which an uncalibrated moving camera makes the world path unreliable.
CAMERA_MOTION_WARN_PX = 8.0

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _num(x, nd: int = 3) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def _stats(v: np.ndarray) -> dict:
    if not v.size:
        return {"mean_cm": None, "p95_cm": None, "max_cm": None}
    return {"mean_cm": _num(v.mean(), 2), "p95_cm": _num(np.percentile(v, 95), 2), "max_cm": _num(v.max(), 2)}


# --- metrics (pure, unit-tested) -------------------------------------------------------------------------
def camera_rotations(joints_world: np.ndarray, joints_camera: np.ndarray) -> np.ndarray:
    """(L,3,3) rotations R with camera = R @ world (+ t), solved per frame by Kabsch.

    joints_world and joints_camera are the same posed body. GVHMR's world heading drifts by several degrees
    against even a static camera over a clip, so one rotation per clip would not do."""
    A = joints_world - joints_world.mean(1, keepdims=True)
    B = joints_camera - joints_camera.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd(np.einsum("lki,lkj->lij", A, B))
    V = Vt.transpose(0, 2, 1)
    D = np.tile(np.eye(3), (len(A), 1, 1))
    D[:, 2, 2] = np.sign(np.linalg.det(V @ U.transpose(0, 2, 1)))
    return V @ D @ U.transpose(0, 2, 1)


def ray_dirs(R: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    """(L,K,3) unit world direction of the camera ray through each (L,K,3) camera-frame point.

    The ray, not the optical axis: a player 8 degrees off-centre would otherwise leak 14 % of her depth noise
    into the lateral skate."""
    ray = points_cam / np.linalg.norm(points_cam, axis=-1, keepdims=True)
    return np.einsum("lji,lkj->lki", R, ray)


def horizontal(dirs: np.ndarray) -> np.ndarray:
    """Unit (x, z) part of Y-up world directions (..., 3)."""
    h = dirs[..., [0, 2]]
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)


def skate(pos_h: np.ndarray, contact: np.ndarray, rays: np.ndarray | None = None, rate: float = 1.0) -> dict:
    """Horizontal speed (cm/frame) of feet while planted: pos_h (L,K,2) metres, contact (L,K) bool.

    A step counts when the foot is planted at both of its ends; rate scales it to another frame rate's
    frames. With rays (L,K,2), the unit horizontal directions of the camera ray through each foot, the steps
    are also split into along (depth) and across (lateral) that ray."""
    steps = np.diff(pos_h, axis=0) * 100 * rate
    both = contact[1:] & contact[:-1]
    size = np.linalg.norm(steps, axis=-1)
    out = {**_stats(size[both]), "n_steps": int(both.sum()),
           "per_joint_cm": [_num(size[both[:, k], k].mean(), 2) if both[:, k].any() else None
                            for k in range(pos_h.shape[1])]}
    if rays is not None:
        v = rays[1:]
        depth = (steps * v).sum(-1)
        lateral = steps[..., 1] * v[..., 0] - steps[..., 0] * v[..., 1]
        energy = float((size[both] ** 2).sum())
        out["lateral"] = _stats(np.abs(lateral[both]))
        out["depth_mean_cm"] = _num(np.abs(depth[both]).mean(), 2) if both.any() else None
        out["depth_share"] = _num((depth[both] ** 2).sum() / energy, 3) if energy > 0 else None
    return out


def ground(heights: np.ndarray, contact: np.ndarray, offsets: np.ndarray, pen_tol_cm: float = 1.0) -> dict:
    """Ground contact from foot-joint heights (L,K) metres (Y of a Y-up world), offsets (K,) rest heights."""
    sole = (heights - offsets) * 100                                 # cm above y=0, per joint
    low = sole.min(axis=1)                                           # min foot height per frame
    pen = np.maximum(0.0, -low)
    out = {
        "min_height_cm": _num(low.min(), 2),
        "penetration_max_cm": _num(pen.max(), 2),
        "penetration_frac": _num((pen > pen_tol_cm).mean(), 3),
        "float_mean_cm": None, "float_max_cm": None, "offset_cm": None, "penetration_regrounded_cm": None,
    }
    planted = sole[contact]
    if planted.size:
        offset = float(np.median(planted))
        out.update(
            float_mean_cm=_num(np.maximum(planted, 0).mean(), 2),
            float_max_cm=_num(planted.max(), 2),
            offset_cm=_num(offset, 2),
            # what is left once the ground is moved to the planted feet (as retarget does)
            penetration_regrounded_cm=_num(max(0.0, -(low - offset).min()), 2),
        )
    return out


def jitter(joints: np.ndarray, fps: float) -> float | None:
    """Mean norm of the third finite difference of (L,J,3) positions, in m/s^3."""
    if len(joints) < 4:
        return None
    return float(np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1).mean() * fps**3)


def depth_share(track: np.ndarray, rays: np.ndarray) -> float | None:
    """Fraction of an (L,3) trajectory's jerk energy along the (L,3) unit camera rays through it."""
    if len(track) < 4:
        return None
    d3 = np.diff(track, n=3, axis=0)
    total = float((d3**2).sum())
    along = (d3 * rays[1:-2]).sum(-1)                                # jerk of frames i..i+3, ray at i+1
    return float((along**2).sum() / total) if total > 0 else None


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(L,J,3) camera-frame points with (L,3,3) intrinsics -> (L,J,2) pixels (NaN behind the camera)."""
    p = np.einsum("lij,lkj->lki", K, points_cam)
    z = p[..., 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[..., :2] / z
    uv[np.broadcast_to(z <= 1e-6, uv.shape)] = np.nan
    return uv


def reprojection(joints_cam: np.ndarray, K: np.ndarray, kp2d: np.ndarray, box_h: np.ndarray) -> dict:
    """SMPL joints vs COCO-17 detections, only keypoints with conf > KP_CONF; box_h (L,) pixels."""
    if K.ndim == 2:
        K = np.broadcast_to(K, (len(joints_cam), 3, 3))
    n = min(len(joints_cam), len(kp2d), len(K), len(box_h))
    coco = np.array(list(COCO_TO_SMPL))
    uv = project(joints_cam[:n, list(COCO_TO_SMPL.values())], K[:n])
    det = kp2d[:n, coco]
    err = np.linalg.norm(uv - det[..., :2], axis=-1)                 # (n, 12) px
    norm = err / np.maximum(box_h[:n, None], 1.0)
    ok = (det[..., 2] > KP_CONF) & np.isfinite(err)

    def stats(mask: np.ndarray) -> dict:
        if not mask.any():
            return {"median_px": None, "median_norm": None, "pck": None, "n": 0}
        return {"median_px": _num(np.median(err[mask]), 1), "median_norm": _num(np.median(norm[mask]), 4),
                "pck": _num((norm[mask] < PCK_ALPHA).mean(), 3), "n": int(mask.sum())}

    out = stats(ok)
    out["parts"] = {part: stats(ok & np.isin(coco, ids)[None]) for part, ids in PARTS.items()}
    out["confident_frac"] = _num(ok.mean(), 3)
    return out


def keypoint_box_h(kp2d: np.ndarray) -> np.ndarray:
    """(L,) person height in px from the confident keypoints, padded for the head top / soles they miss."""
    ok = kp2d[..., 2] > KP_CONF
    y = kp2d[..., 1]
    h = (np.where(ok, y, -np.inf).max(1) - np.where(ok, y, np.inf).min(1)) * 1.15
    valid = ok.sum(1) >= 2
    return np.where(valid, h, np.median(h[valid]) if valid.any() else 1.0)


def speed(track: np.ndarray, fps: float) -> np.ndarray:
    """(L,) speed in m/s of one joint's (L,3) trajectory (central differences)."""
    return np.linalg.norm(np.gradient(track, axis=0), axis=-1) * fps


def rig_indices(names: list[str], schemes: dict[str, tuple[str, ...]]) -> list[int] | None:
    """Indices of one naming scheme's joints among a rig's joint names, or None if no scheme matches."""
    clean = [n.split(":")[-1].split("|")[-1] for n in names]       # mixamorig:LeftFoot, Armature|LeftFoot
    for fold in (str, str.lower):
        folded = [fold(c) for c in clean]
        for scheme in schemes.values():
            want = [fold(s) for s in scheme]
            if all(w in folded for w in want):
                return [folded.index(w) for w in want]
    return None


def feet_indices(names: list[str]) -> list[int] | None:
    """Indices of the four retargeted foot joints (FOOT_CHANNELS order), or None if the rig has no match."""
    return rig_indices(names, RIG_FEET)


def contact_mask(static_conf: np.ndarray) -> np.ndarray:
    return static_conf[:, :4] > CONTACT_P


# --- one clip ---------------------------------------------------------------------------------------------
def clip_metrics(m, meta: dict, track=None, anim=None, handedness: str = "right") -> dict:
    """All metrics of one clip: m = motion.npz mapping, track = track.npz mapping, anim = anim/joints npz."""
    fps = float(m["fps"])
    J = np.asarray(m["joints_world"], np.float64)
    Jc = np.asarray(m["joints_camera"], np.float64)[: len(J)]
    n = len(J)
    rel = J - J[:, PELVIS:PELVIS + 1]
    contact = contact_mask(np.asarray(m["static_conf"])[:n])
    names = [name for name, _, _ in FOOT_CHANNELS]
    feet = [j for _, _, j in FOOT_CHANNELS]
    stroke = meta.get("stroke") or (str(m["stroke"]) if "stroke" in m else "unknown")
    q: dict = {"n_frames": n, "fps": fps, "stroke": stroke}

    q["contact"] = {
        "frames_any": int(contact.any(axis=1).sum()),
        "frac": {name: _num(contact[:, c].mean(), 3) for name, c, _ in FOOT_CHANNELS},
    }
    R = camera_rotations(J[:, :22], Jc[:, :22])
    q["skate"] = skate(J[:, feet][..., [0, 2]], contact, horizontal(ray_dirs(R, Jc[:, feet])))
    q["skate"]["per_joint_cm"] = dict(zip(names, q["skate"]["per_joint_cm"], strict=True))
    q["ground"] = ground(J[:, feet, 1], contact, np.array([SOLE_OFFSET_M[j] for j in feet]))
    q["jitter"] = {
        "all_ms3": _num(jitter(J, fps), 1),
        "wrists_ms3": _num(jitter(J[:, list(WRISTS.values())], fps), 1),
        "pose_ms3": _num(jitter(np.delete(rel, PELVIS, axis=1), fps), 1),
        "root_ms3": _num(jitter(J[:, [PELVIS]], fps), 1),
        "root_depth_share": _num(depth_share(J[:, PELVIS], ray_dirs(R, Jc[:, [PELVIS]])[:, 0]), 3),
    }

    if track is not None:
        bbx = np.asarray(track["bbx_xyxy"], np.float64)
        box_h, box_src = bbx[:, 3] - bbx[:, 1], "track"
    else:
        box_h, box_src = keypoint_box_h(np.asarray(m["kp2d"])), "keypoints"
    rep = reprojection(Jc, np.asarray(m["K"], np.float64), np.asarray(m["kp2d"]), box_h)
    q["reprojection"] = {**rep, "box": box_src, "box_h_median_px": _num(np.median(box_h), 1)}

    # judge the contact downstream uses (clip.json, possibly refined by `curate`); keep the audio one too
    audio_k = int(meta.get("contact_index_selected", m["contact_index"] if "contact_index" in m else -1))
    k = int(meta.get("contact_index", audio_k))
    q["timing"] = {"contact_index": k, "contact_source": meta.get("contact_source"), "peak_frame": None,
                   "offset_frames": None, "offset_s": None, "peak_wrist_speed_ms": None,
                   "swing_speed_ms": None, "apex_offset_frames": None, "audio_contact_index": audio_k,
                   "audio_offset_frames": None}
    if stroke in STROKES and 0 <= k < n:
        # relative to the pelvis: the root's depth noise would otherwise dominate the wrist speed
        w = WRISTS.get(handedness, WRISTS["right"])
        v = speed(rel[:, w], fps)
        f = int(np.argmax(v))
        near = v[max(0, k - TIMING_FRAMES[1]): k + TIMING_FRAMES[1] + 1]
        q["timing"].update(peak_frame=f, offset_frames=f - k, offset_s=_num((f - k) / fps, 3),
                           peak_wrist_speed_ms=_num(v[f], 2), swing_speed_ms=_num(near.max(), 2),
                           audio_offset_frames=f - audio_k)
        if stroke in OVERHEAD:
            q["timing"]["apex_offset_frames"] = int(np.argmax(rel[:, w, 1])) - k

    q["tracking"] = {"coverage": _num(track["coverage"], 3) if track is not None else None}
    q["camera"] = {
        "static": bool(m["static_camera"]) if "static_camera" in m else None,
        "motion_px": _num(m["camera_motion_px"], 2) if "camera_motion_px" in m else None,
        "calibrated": bool(m["calibrated"]) if "calibrated" in m else False,
    }
    q["retarget"] = retarget_metrics(anim, contact, fps) if anim is not None else None
    return q


def retarget_metrics(anim, contact: np.ndarray, fps: float) -> dict:
    missing = [key for key in ("names", "positions") if key not in anim]
    if missing:
        return {"error": f"anim/joints file lacks {', '.join(missing)}"}
    names = [str(s) for s in np.asarray(anim["names"]).tolist()]
    idx = feet_indices(names)
    if idx is None:
        return {"error": "no foot joints recognised", "names": names[:40]}
    P = np.asarray(anim["positions"], np.float64)
    fa = float(anim["fps"]) if "fps" in anim else fps
    # the animation may be resampled: look up each of its frames in the source contacts by time, and
    # express its steps per source frame so the thresholds apply unchanged
    src = np.clip(np.round(np.arange(len(P)) * fps / fa).astype(int), 0, len(contact) - 1)
    out = skate(P[:, idx][..., :2], contact[src], rate=fa / fps)   # Z-up: horizontal = x, y
    out["joints"] = [names[i] for i in idx]
    out["per_joint_cm"] = dict(zip(out["joints"], out["per_joint_cm"], strict=True))
    out["n_frames"] = len(P)
    out["fps"] = fa
    # a foot lock can pin the feet and still leave the root's depth noise on the hips
    root = rig_indices(names, RIG_ROOT)
    out["root_ms3"] = _num(jitter(P[:, root], fa), 1) if root else None
    out["pose_ms3"] = _num(jitter(np.delete(P - P[:, root], root, axis=1), fa), 1) if root else None
    return out


# --- gate ------------------------------------------------------------------------------------------------
def gate(q: dict, min_coverage: float = 0.7, regrounded: bool = True) -> tuple[str, list[str], list[str]]:
    """PASS / WARN / FAIL from the thresholds above; returns (status, reasons, notes)."""
    status, reasons, notes = "PASS", [], []

    def flag(level: str, msg: str) -> None:
        nonlocal status
        reasons.append(f"{level}: {msg}")
        if RANK[level] > RANK[status]:
            status = level

    def above(value, limits, label: str, unit: str = "") -> None:
        if value is None:
            return
        warn, fail = limits
        if value > fail:
            flag("FAIL", f"{label} {value:g}{unit} > {fail:g}")
        elif value > warn:
            flag("WARN", f"{label} {value:g}{unit} > {warn:g}")

    def below(value, limits, label: str, unit: str = "") -> None:
        if value is None:
            return
        warn, fail = limits
        if value < fail:
            flag("FAIL", f"{label} {value:g}{unit} < {fail:g}")
        elif value < warn:
            flag("WARN", f"{label} {value:g}{unit} < {warn:g}")

    sk = q["skate"]
    if sk["n_steps"] < MIN_CONTACT_STEPS:
        flag("WARN", f"only {sk['n_steps']} planted foot steps (no foot lock, skate not measurable)")
    lat = sk.get("lateral")
    if lat is not None:
        above(lat["mean_cm"], SKATE_MEAN_CM, "skate across the camera ray", " cm/f")
        above(lat["p95_cm"], SKATE_P95_CM, "skate p95 across the camera ray", " cm/f")
        if sk["mean_cm"] is not None and sk["mean_cm"] > SKATE_MEAN_CM[0]:
            share = sk.get("depth_share")
            along = f", {share:.0%} of it along the camera ray (depth noise)" if share is not None else ""
            flag("WARN", f"skate {sk['mean_cm']:g} cm/f{along}")
    else:
        above(sk["mean_cm"], SKATE_MEAN_CM, "skate mean", " cm/f")
        above(sk["p95_cm"], SKATE_P95_CM, "skate p95", " cm/f")

    g = q["ground"]
    if regrounded:
        if g["offset_cm"] is not None and abs(g["offset_cm"]) > GROUND_OFFSET_CM:
            flag("WARN", f"ground offset {g['offset_cm']:+g} cm (re-grounded at retarget)")
        if g["penetration_regrounded_cm"] is not None and g["penetration_regrounded_cm"] > GROUND_PEN_CM:
            flag("WARN", f"penetration after re-grounding {g['penetration_regrounded_cm']:g} cm")
    else:
        if g["penetration_max_cm"] is not None and g["penetration_max_cm"] > GROUND_PEN_CM:
            flag("WARN", f"penetration {g['penetration_max_cm']:g} cm")
        if g["float_mean_cm"] is not None and g["float_mean_cm"] > GROUND_OFFSET_CM:
            flag("WARN", f"floating while planted {g['float_mean_cm']:g} cm")

    jit = q["jitter"]
    above(jit["pose_ms3"], POSE_JITTER_MS3, "pose jitter", " m/s3")
    if jit["root_ms3"] is not None and jit["root_ms3"] > ROOT_JITTER_WARN_MS3:
        share = jit.get("root_depth_share")
        along = f", {share:.0%} along the camera ray" if share is not None else ""
        flag("WARN", f"root jitter {jit['root_ms3']:g} m/s3{along}")

    r = q["reprojection"]
    if r["n"] == 0:
        flag("WARN", "no confident 2D keypoints")
    above(r["median_norm"], REPROJ_NORM, "reprojection median", " x box")
    below(r["pck"], PCK_MIN, "PCK@0.1")

    t = q["timing"]
    below(t.get("swing_speed_ms"), SWING_MIN_MS,
          f"no swing: hitting wrist within {TIMING_FRAMES[1]} frames of contact peaks at", " m/s")
    if t["offset_frames"] is not None and abs(t["offset_frames"]) > TIMING_FRAMES[0]:
        apex = f", wrist apex {t['apex_offset_frames']:+d}" if t.get("apex_offset_frames") is not None else ""
        level = "FAIL" if abs(t["offset_frames"]) > TIMING_FRAMES[1] else "WARN"
        flag(level, f"wrist speed peak {t['offset_frames']:+d} frames from contact{apex} "
                    f"(> {TIMING_FRAMES[level == 'FAIL']}: swing or contact label wrong)")

    cov = q["tracking"]["coverage"]
    if cov is None:
        notes.append("no track.npz")
    elif cov < COVERAGE_FAIL:
        flag("FAIL", f"track coverage {cov:.0%} < {COVERAGE_FAIL:.0%}")
    elif cov < min_coverage:
        flag("WARN", f"track coverage {cov:.0%} < {min_coverage:.0%}")

    cam = q["camera"]
    if cam["static"] is False and not cam["calibrated"]:
        if (cam["motion_px"] or 0) > CAMERA_MOTION_WARN_PX:
            flag("WARN", f"uncalibrated moving camera ({cam['motion_px']:g} px/frame)")
        else:
            notes.append(f"moving camera ({cam['motion_px']} px/frame), uncalibrated")

    rt = q.get("retarget")
    if rt is None:
        notes.append("not retargeted yet")
    elif "error" in rt:
        flag("WARN", f"retarget: {rt['error']}")
    else:
        above(rt["mean_cm"], SKATE_MEAN_CM, "retargeted skate", " cm/f")
        above(rt["p95_cm"], SKATE_P95_CM, "retargeted skate p95", " cm/f")
        above(rt.get("pose_ms3"), POSE_JITTER_MS3, "retargeted pose jitter", " m/s3")
        if (rt.get("root_ms3") or 0) > ROOT_JITTER_WARN_MS3:
            flag("WARN", f"retargeted root jitter {rt['root_ms3']:g} m/s3 (depth noise kept on the hips)")
    return status, reasons, notes


def thresholds(min_coverage: float) -> dict:
    return {
        "contact_p": CONTACT_P, "kp_conf": KP_CONF, "pck_alpha": PCK_ALPHA,
        "skate_mean_cm": SKATE_MEAN_CM, "skate_p95_cm": SKATE_P95_CM, "pose_jitter_ms3": POSE_JITTER_MS3,
        "root_jitter_warn_ms3": ROOT_JITTER_WARN_MS3, "reproj_norm": REPROJ_NORM, "pck_min": PCK_MIN,
        "timing_frames": TIMING_FRAMES, "swing_min_ms": SWING_MIN_MS,
        "coverage": (min_coverage, COVERAGE_FAIL),
        "ground_offset_cm": GROUND_OFFSET_CM, "ground_pen_cm": GROUND_PEN_CM,
        "min_contact_steps": MIN_CONTACT_STEPS, "camera_motion_warn_px": CAMERA_MOTION_WARN_PX,
        "sole_offset_m": SOLE_OFFSET_M,
    }


# --- summary ---------------------------------------------------------------------------------------------
ROW_KEYS = {
    "skate_cm": ("skate", "mean_cm"), "skate_p95_cm": ("skate", "p95_cm"),
    "skate_depth_share": ("skate", "depth_share"), "ground_offset_cm": ("ground", "offset_cm"),
    "pen_regrounded_cm": ("ground", "penetration_regrounded_cm"), "jitter_ms3": ("jitter", "all_ms3"),
    "jitter_wrists_ms3": ("jitter", "wrists_ms3"), "jitter_pose_ms3": ("jitter", "pose_ms3"),
    "jitter_root_ms3": ("jitter", "root_ms3"), "reproj_px": ("reprojection", "median_px"),
    "reproj_norm": ("reprojection", "median_norm"), "pck": ("reprojection", "pck"),
    "timing_frames": ("timing", "offset_frames"), "swing_ms": ("timing", "swing_speed_ms"),
    "coverage": ("tracking", "coverage"), "camera_px": ("camera", "motion_px"),
}


def _row(cid: str, q: dict) -> dict:
    row = {"id": cid, "stroke": q["stroke"], "status": q["status"]}
    row.update({k: (q.get(a) or {}).get(b) for k, (a, b) in ROW_KEYS.items()})
    row["skate_lateral_cm"] = (q["skate"].get("lateral") or {}).get("mean_cm")
    rt = q.get("retarget")
    row["retarget_skate_cm"] = rt.get("mean_cm") if rt else None
    row["retarget_root_ms3"] = rt.get("root_ms3") if rt else None
    row["reasons"] = q["reasons"]
    return row


def _aggregate(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), **{s: sum(r["status"] == s for r in rows) for s in RANK}}
    for k in [*ROW_KEYS, "skate_lateral_cm", "retarget_skate_cm", "retarget_root_ms3"]:
        signed = k == "timing_frames"  # aggregate |offset|: early and late are both mistimed
        vals = [abs(r[k]) if signed else r[k] for r in rows if r.get(k) is not None]
        out[f"median_{'abs_' if signed else ''}{k}"] = _num(np.median(vals), 3) if vals else None
    return out


def summarize(rows: list[dict], min_coverage: float, missing: list[str], errors: dict[str, str]) -> dict:
    by_stroke: dict[str, list[dict]] = {}
    for r in rows:
        by_stroke.setdefault(r["stroke"], []).append(r)
    return {
        "thresholds": thresholds(min_coverage),
        "overall": _aggregate(rows),
        "by_stroke": {s: _aggregate(rs) for s, rs in sorted(by_stroke.items())},
        "clips": rows,
        "missing_motion": missing,
        "errors": errors,
    }


def _print_table(rows: list[dict]) -> None:
    from rich.console import Console
    from rich.table import Table

    t = Table(title="motion QA", header_style="bold")
    cols = [("clip", "id", "{}"), ("gate", "status", "{}"), ("skate", "skate_cm", "{:.1f}"),
            ("lat", "skate_lateral_cm", "{:.1f}"), ("depth%", "skate_depth_share", "{:.0%}"),
            ("gnd", "ground_offset_cm", "{:+.0f}"), ("pose j", "jitter_pose_ms3", "{:.0f}"),
            ("root j", "jitter_root_ms3", "{:.0f}"), ("px", "reproj_px", "{:.1f}"),
            ("/box", "reproj_norm", "{:.3f}"), ("PCK", "pck", "{:.2f}"), ("dt", "timing_frames", "{:+d}"),
            ("swing", "swing_ms", "{:.1f}"), ("cov", "coverage", "{:.2f}"), ("cam", "camera_px", "{:.1f}"),
            ("rt", "retarget_skate_cm", "{:.1f}"), ("rt root j", "retarget_root_ms3", "{:.0f}")]
    for head, *_ in cols:
        t.add_column(head, justify="left" if head in ("clip", "gate") else "right", no_wrap=True)
    colour = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
    for r in rows:
        cells = ["-" if r.get(k) is None else fmt.format(r[k]) for _, k, fmt in cols]
        cells[1] = f"[{colour[r['status']]}]{r['status']}[/]"
        t.add_row(*cells)
    # the shared console is 80 columns when stderr is not a terminal (remote runs, logs)
    Console(stderr=True, highlight=False, width=max(console.width, 160)).print(t)


def _npz(path: Path) -> dict | None:
    return read_npz(path) if path.exists() else None


def run(cfg: Config) -> dict:
    run_ = cfg.run
    rows, missing, errors = [], [], {}
    min_cov = cfg.track.min_coverage
    for cid in run_.clip_ids():
        clip = run_.clip(cid)
        if not clip.motion.exists():
            missing.append(cid)
            continue
        try:
            meta = read_json(clip.meta)
            anim = _npz(run_.joints_dir / f"{cid}.npz")
            q = clip_metrics(_npz(clip.motion), meta, _npz(clip.track), anim, cfg.athlete.handedness)
        except Exception as e:  # noqa: BLE001 - one unreadable clip must not stop the others; it fails below
            log(f"[red]{cid}: QA failed: {e!r}[/]")
            errors[cid] = repr(e)
            # an unmeasured clip must not pass silently: `curate` reads the gate
            write_json(clip.qa, {"id": cid, "gate": "FAIL", "status": "FAIL",
                                 "reasons": [f"FAIL: QA could not measure the motion ({e!r})"], "notes": []})
            continue
        q["status"], q["reasons"], q["notes"] = gate(q, min_cov, cfg.retarget.ground_from_contacts)
        # "gate" duplicates "status" for readers that look for that name (curate)
        write_json(clip.qa, {"id": cid, "gate": q["status"], **q})
        rows.append(_row(cid, q))
    if missing:
        log(f"[yellow]no motion yet: {', '.join(missing)}[/]")
    summary = summarize(rows, min_cov, missing, errors)
    write_json(run_.qa_summary, summary)
    if rows:
        _print_table(rows)
        o = summary["overall"]
        log(f"{o['n']} clips: [green]{o['PASS']} pass[/], [yellow]{o['WARN']} warn[/], "
            f"[red]{o['FAIL']} fail[/]" + (f", [red]{len(errors)} errors[/]" if errors else "")
            + f" → {run_.qa_summary}")
    return summary
