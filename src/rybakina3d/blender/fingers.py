"""Finger bones, finger skin weights, grip poses and the racket's seat on the palm, from the character's
hand_rig.json (runs INSIDE Blender; the numpy half imports without it).

    blender -b --factory-startup --python-exit-code 1 --python fingers.py -- export rig.glb hand_mesh.npz weld_m

`rybakina3d.character.hands` runs that export (the rig imported and welded exactly as retarget.build does), finds
the finger joints on the welded mesh and writes character/hand_rig.json (+ .npz) next to rigged.glb. Everything in
it is in the rig's rest world frame as retarget.py sees it after import (metres, Z-up), never vertex indices.
Per hand ("Right", "Left"):
  palm     wrist joint, long axis a (wrist -> fingers), palm normal n (out of the palm, fitted to the palm-side
           skin), thumb side t (right hand a x n, left hand n x a), palm centre, heel pad and index knuckle, and
           where the wrist is narrowest (past a wristband); the palm's skin points (npz) the handle rests on
  bones    15 finger bones, 3 per finger (thumb, index, middle, ring, pinky): head, tail, Z axis (the palm side),
           and per pose ("grip_world", "relaxed_world") the bone's rotation about its head in the rest world frame
           (w, x, y, z), chained like pose bones (each in its parent's rest frame)
  shares   (npz) rest positions of the hand's vertices and how each one's hand weight splits over the hand and its
           finger bones: smoothstep bands at the finger joints, so the wrist blend stays as it was
"ok": false (with "reason") means the mesh did not give a believable hand; the build then keeps the fixed grip
(retarget.grip_matrix, no finger bones) there and says so.

Retarget uses it like this (each behind a setting, see rybakina3d.config.Retarget):
  seat_frame          the racket's rest world matrix on the palm (handle on the skin, butt at the heel, handle from
                      the heel pad to the index knuckle) and the constant hand-bone offset that keeps the racket's
                      world direction what attach_racket's fixed grip gives
  add_finger_bones    the 30 bones, in edit mode, under the hand bones (before rig_from_blender)
  finger_weights      each skinned mesh's hand weight split over the finger bands (after the limb smoothing)
  finger_pose_quats   pose-bone quaternions from relaxed (0) to grip (1)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Matrix, Vector
except ImportError:  # imported outside Blender (unit tests of the numpy part)
    bpy = None
    Matrix = Vector = None

try:
    from . import retarget as rt
except ImportError:  # inside Blender: a top-level script next to retarget.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import retarget as rt

HAND_RIG_VERSION = 1
SIDES = ("Right", "Left")
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FINGER_NAMES = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Pinky"}
SHARES = 1 + 3 * len(FINGERS)   # share columns: the hand, then thumb1..3, index1..3, middle1..3, ring1..3, pinky1..3

# Finger weight bands: half-widths (m) of the smoothstep at joint 0 (knuckle; thumb: along its metacarpal), 1 and 2.
BAND_HALF_M = {"thumb": (0.014, 0.008, 0.007), "index": (0.012, 0.0065, 0.0055), "middle": (0.012, 0.0065, 0.0055),
               "ring": (0.012, 0.0065, 0.0055), "pinky": (0.011, 0.006, 0.005)}
THUMB_BAND_START = 0.3          # of the thumb's metacarpal: the thenar's proximal part stays on the hand
# Up to the web, a vertex belongs to a finger's bones out to `reach` finger radii from its bone chain, fading over
# `fade` radii (the palm next to a knuckle stays on the hand); past the web (WEB_BAND_M) the whole finger does.
RADIAL = {"thumb": (2.4, 1.0), "index": (2.0, 2.0), "middle": (2.0, 2.0), "ring": (2.0, 2.0), "pinky": (2.0, 2.0)}
WEB_BAND_M = 0.005              # half-width of that switch, centred on where the finger leaves the palm
MAX_SHARES = 3                  # bones per vertex out of the hand weight: with the forearm, glTF's 4 influences
MATCH_TOL_M = 0.002             # a skinned vertex takes the shares of the stored rest vertex this close, else hand only

# Racket seat. The handle's flat rests on the palm skin; the butt sits at the heel of the hand, its rim just
# past the wrist's narrowest section (out of a wristband); the shaft runs across the palm from the heel pad to the
# index knuckle. Racket constants mirror blender/racket.py (which needs bpy to import).
HANDLE_RADIUS = 0.016           # racket.GRIP / 2
BUTT_RADIUS = 0.0185            # racket.BUTT / 2
BUTT_LENGTH = 0.012             # racket.BUTT_LENGTH
SINK_M = 0.001                  # the palm's skin gives this much under the handle
PAD_SINK_M = 0.005              # and the pads between its ends (thenar, palm centre) this much more: a straight
                                # handle bridges the palm's hollow and presses into the thenar's bulge
GRIP_ANGLE_RANGE_DEG = (30.0, 50.0)   # heel pad -> index knuckle, from the hand's long axis towards the thumb
BUTT_PAST_WRIST_M = (0.03, 0.05)      # butt centre past the wrist joint, along the hand
BUTT_CLEAR_M = 0.003            # the butt cap's rim this far past the wrist's narrowest section
KNUCKLE_SETBACK_M = HANDLE_RADIUS   # the shaft's line passes this far proximal of the index knuckle: the knuckle
                                    # rests on the handle's flat and the index can still curl round it
SEAT_PAST_KNUCKLE_M = 0.01      # the palm under the handle counts up to this far past the index knuckle
MAX_TILT_DEG = 15.0             # the handle may tilt this much out of the palm plane to rest on heel and knuckle
MIN_SEAT_POINTS = 12            # palm skin points under the handle needed to seat it
QA_HANDLE_M = 0.20              # QA: the handle as a cylinder of HANDLE_RADIUS this long from the butt
# QA of the whole lower racket (inside_racket), from racket.py: the grip to HANDLE_LENGTH, the shaft (a box) to
# THROAT_Y, then the open throat's two arms (boxes) to where they meet the head
HANDLE_LENGTH = 0.19            # racket.HANDLE_LENGTH
THROAT_Y = 0.255                # racket.THROAT_Y
SHAFT_HALF = (0.010, 0.0105)    # racket.SHAFT_WIDTH / 2 (across, in the string plane), SHAFT_DEPTH / 2 (along +Z)
THROAT_ARM_END = (0.0784, 0.4058)   # an arm's end at the head (x, y): racket._head_point(THROAT_ANGLE) minus its beam
THROAT_ARM_HALF = (0.0066, 0.0109)  # racket.BEAM_WIDTH / 2 * 1.1 (in plane), BEAM_DEPTH / 2 * 0.95
PALM_SKIN_NEAR_M = 0.01         # palm_in_bone: the palm skin within this of the centre, across the palm
HAND_POINT_CLOSINGS = (0.0, 0.25, 0.5, 0.75, 1.0)   # hand_points: the finger closings the skin is posed at ...
HAND_POINT_EVERY = 8            # ... every this many of the hand's vertices (~470 on the variant's left hand)


# ------------------------------------------------------------------------------------------------ geometry
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def smoothstep_pm1(x: np.ndarray) -> np.ndarray:
    """0 at x <= -1, 1 at x >= 1, 0.5 at 0, flat at both ends (twohand.smoothstep spans 0 .. 1 instead)."""
    t = np.clip(0.5 * (np.asarray(x, np.float64) + 1), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bone_names(hand_bone: str) -> list[str]:
    """The 15 finger bones under one hand bone: RightHandThumb1..3, RightHandIndex1..3, ... RightHandPinky1..3."""
    return [f"{hand_bone}{FINGER_NAMES[f]}{i}" for f in FINGERS for i in (1, 2, 3)]


def bone_frame(head, tail, z_axis) -> np.ndarray:
    """(3,3) world rest rotation of a bone (columns X, Y, Z) as Blender builds it from head, tail and align_roll:
    Y along the bone, Z towards `z_axis`, X = Y x Z."""
    y = unit(np.asarray(tail, np.float64) - np.asarray(head, np.float64))
    z = np.asarray(z_axis, np.float64)
    z = unit(z - (z @ y) * y)
    return np.stack([np.cross(y, z), y, z], 1)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Spherical interpolation of unit quaternions (w, x, y, z); t scalar or (...) array -> (..., 4)."""
    q0, q1 = unit(q0), unit(q1)
    t = np.asarray(t, np.float64)[..., None]
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    theta = math.acos(min(d, 1.0))
    if theta < 1e-6:
        return unit((1 - t) * q0 + t * q1)
    return unit((np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / math.sin(theta))


def bone_rest(rig: rt.Rig) -> dict[str, np.ndarray]:
    """World rest rotation (3,3) of every bone of a retarget.Rig (finger_pose_quats' `bone_rest`)."""
    return {n: rig.obj_rot @ rig.rest[i, :3, :3] for i, n in enumerate(rig.names)}


# ------------------------------------------------------------------------------------------- skin weights
def band_shares(P: np.ndarray, joints: dict, radius: dict, labels: np.ndarray, free: dict | None = None) -> np.ndarray:
    """(N, SHARES) split of each vertex's hand weight over [hand, thumb1..3, ..., pinky1..3]; rows sum to 1.

    joints[f] (4,3): the finger's joints (knuckle or thumb base, two more, tip); radius[f]: the finger's mean radius;
    labels (N,5): how much each vertex belongs to each finger (rows sum to 1, from geodesic distances to the tips);
    free[f]: how far from the tip the finger is free of the palm (None: the radial fade applies all along).
    Along a finger, u is the arc length of the closest point on its bone chain: the hand -> bone 1 -> 2 -> 3
    handovers are smoothsteps of BAND_HALF_M half-width centred on the joints; up to the web, vertices far from the
    chain (palm next to a knuckle) fade back to the hand. Each vertex keeps its MAX_SHARES largest shares, so the
    exported GLB (4 influences, the forearm's included) deforms like the Blender render."""
    P = np.asarray(P, np.float64)
    S = np.zeros((len(P), SHARES))
    for k, f in enumerate(FINGERS):
        J = np.asarray(joints[f], np.float64)
        L = np.linalg.norm(np.diff(J, axis=0), axis=1)
        best, u, acc = np.full(len(P), np.inf), np.zeros(len(P)), 0.0
        for i in range(3):
            d = (J[i + 1] - J[i]) / L[i]
            lo = -0.05 if i == 0 else 0.0              # before the first joint and past the tip the chain goes on
            hi = L[i] + (0.05 if i == 2 else 0.0)
            s = np.clip((P - J[i]) @ d, lo, hi)
            dist = np.linalg.norm(P - (J[i] + s[:, None] * d), axis=1)
            closer = dist < best
            best[closer], u[closer] = dist[closer], acc + s[closer]
            acc += L[i]
        h = BAND_HALF_M[f]
        u0 = THUMB_BAND_START * L[0] if f == "thumb" else 0.0
        b0, b1, b2 = (smoothstep_pm1((u - c) / w) for c, w in ((u0, h[0]), (L[0], h[1]), (L[0] + L[1], h[2])))
        reach, fade = RADIAL[f]
        on = 1 - smoothstep_pm1((best - reach * radius[f]) / (fade * radius[f]))
        if free is not None:
            on = 1 - (1 - on) * (1 - smoothstep_pm1((u - (L.sum() - free[f])) / WEB_BAND_M))
        parts = np.clip([1 - b0 * on, (b0 - b1) * on, (b1 - b2) * on, b2 * on], 0.0, None)
        lam = np.asarray(labels, np.float64)[:, k]
        S[:, 0] += lam * parts[0]
        S[:, 1 + 3 * k:4 + 3 * k] = (lam * parts[1:]).T
    S[np.arange(len(S))[:, None], np.argsort(-S, 1)[:, MAX_SHARES:]] = 0.0
    return S / np.maximum(S.sum(1, keepdims=True), 1e-12)


def split_hand_weight(w_hand: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """(N, SHARES) new weights [hand, fingers...]: each vertex's hand weight times its shares (same total)."""
    return np.asarray(w_hand, np.float64)[:, None] * np.asarray(shares, np.float64)


def match_shares(co: np.ndarray, ref_co: np.ndarray, ref_shares: np.ndarray, tol: float = MATCH_TOL_M,
                 chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Shares of the nearest stored rest vertex (within tol) for each of `co`; the rest stay on the hand.
    Returns (shares (N, SHARES), distance to the nearest stored vertex (N,))."""
    co, ref_co = np.asarray(co, np.float64), np.asarray(ref_co, np.float64)
    near, dist = np.zeros(len(co), np.int64), np.full(len(co), np.inf)
    sq = np.sum(ref_co ** 2, 1)
    for i in range(0, len(co), chunk):
        c = co[i:i + chunk]
        d2 = np.sum(c ** 2, 1)[:, None] - 2 * c @ ref_co.T + sq[None]
        near[i:i + chunk] = np.argmin(d2, 1)
        dist[i:i + chunk] = np.sqrt(np.maximum(d2[np.arange(len(c)), near[i:i + chunk]], 0.0))
    out = np.zeros((len(co), SHARES))
    out[:, 0] = 1.0
    ok = dist <= tol
    out[ok] = np.asarray(ref_shares, np.float64)[near[ok]]
    return out, dist


# --------------------------------------------------------------------------------------------- racket seat
def rest_line(y: np.ndarray, c: np.ndarray, max_slope: float, at: float = 0.5,
              steps: int = 801) -> tuple[float, float]:
    """(h0, k): the line h(y) = h0 + k y (|k| <= max_slope) on or above every point (y_i, c_i) that is lowest at
    the fraction `at` of the points' y range (0.5: lowest on average): a straight handle resting on the palm's
    highest points, pressed down nearest to `at`."""
    y, c = np.asarray(y, np.float64), np.asarray(c, np.float64)
    mid = y.min() + at * (y.max() - y.min())
    ks = np.linspace(-max_slope, max_slope, steps)
    height = np.max(c[None] - ks[:, None] * (y[None] - mid), 1)   # the line's height there for each slope
    k = float(ks[np.argmin(height)])
    return float(height.min() - k * mid), k


def palm_coords(palm: dict, p: np.ndarray) -> np.ndarray:
    """(..., 3) points -> (s along the hand, w towards the thumb, h out of the palm) from the wrist joint."""
    F = np.stack([unit(palm["long_axis"]), unit(palm["thumb_side"]), unit(palm["normal"])], 1)
    return (np.asarray(p, np.float64) - np.asarray(palm["wrist"], np.float64)) @ F


def seat_from_palm(palm: dict, skin: np.ndarray) -> dict:
    """Where the racket sits on this palm, from the stored palm frame and its skin points (K,3).

    The handle's plan line runs from the heel pad to the index knuckle, KNUCKLE_SETBACK_M proximal of it (angle
    clamped to GRIP_ANGLE_RANGE_DEG, through that point); the butt's centre sits BUTT_PAST_WRIST_M past the wrist
    joint, its rim just past the wrist's narrowest section; the handle (butt cap and grip as cylinders) rests on the
    highest palm skin under it, out to SEAT_PAST_KNUCKLE_M past the knuckle, sunk SINK_M into it (PAD_SINK_M more
    away from its two ends), tilting at most MAX_TILT_DEG. The string normal is the palm normal (eastern grip: palm
    flat on the back bevel). "clamped" lists the limits the seat ran into ("angle", "butt_past_wrist", "tilt"):
    there the hand is outside what the constants expect. Raises ValueError without enough skin."""
    w0 = np.asarray(palm["wrist"], np.float64)
    a, t, n = unit(palm["long_axis"]), unit(palm["thumb_side"]), unit(palm["normal"])
    (sK, wK, _), (sH, wH, _) = palm_coords(palm, palm["index_knuckle"]), palm_coords(palm, palm["heel"])
    sK -= KNUCKLE_SETBACK_M
    measured = math.degrees(math.atan2(wK - wH, sK - sH))
    angle = float(np.clip(measured, *GRIP_ANGLE_RANGE_DEG))
    th = math.radians(angle)
    d = math.cos(th) * a + math.sin(th) * t                 # along the handle, in the palm plane
    q = math.cos(th) * t - math.sin(th) * a                 # across it, in the palm plane
    butt_s = float(np.clip(palm["wrist_clear_s"] + BUTT_RADIUS * math.sin(th) + BUTT_CLEAR_M, *BUTT_PAST_WRIST_M))
    base = w0 + a * butt_s + t * (wK + (butt_s - sK) * math.tan(th))
    rel = np.asarray(skin, np.float64) - base
    y, lat, z = rel @ d, rel @ q, rel @ n
    y_end = (np.asarray(palm["index_knuckle"], np.float64) - base) @ d + SEAT_PAST_KNUCKLE_M
    r = np.where(y < BUTT_LENGTH, BUTT_RADIUS, HANDLE_RADIUS)
    under = (y >= 0) & (y <= y_end) & (np.abs(lat) < r)
    if under.sum() < MIN_SEAT_POINTS:
        raise ValueError(f"{int(under.sum())} palm skin points under the handle (< {MIN_SEAT_POINTS})")
    yu = y[under]
    pads = np.clip(1 - np.abs((yu - yu.min()) / max(np.ptp(yu), 1e-9) - 0.5) / 0.3, 0.0, 1.0)  # 0 at the ends
    top = z[under] + np.sqrt(r[under] ** 2 - lat[under] ** 2) - SINK_M - PAD_SINK_M * pads   # axis height on each
    max_slope = math.tan(math.radians(MAX_TILT_DEG))
    h0, k = rest_line(yu, top, max_slope)
    axis = unit(d + k * n)
    normal = unit(n - (n @ axis) * axis)
    free_s = palm["wrist_clear_s"] + BUTT_RADIUS * math.sin(th) + BUTT_CLEAR_M
    clamped = [name for name, hit in (("angle", angle != measured), ("butt_past_wrist", butt_s != free_s),
                                      ("tilt", abs(k) >= max_slope * (1 - 1e-9))) if hit]
    return {"butt": base + n * h0, "axis": axis, "normal": normal, "angle_deg": round(angle, 2),
            "measured_angle_deg": round(measured, 2), "tilt_deg": round(math.degrees(math.atan(k)), 2),
            "butt_past_wrist_m": round(butt_s, 4), "axis_height_m": round(float(h0), 4),
            "knuckle_along_m": round(float(y_end - SEAT_PAST_KNUCKLE_M), 4), "skin_points": int(under.sum()),
            "clamped": clamped}


def seat_matrix(seat: dict) -> np.ndarray:
    """4x4 racket rest world matrix: +Y along the shaft from the butt, +Z the string normal, origin at the butt."""
    M = np.eye(4)
    r, n = np.asarray(seat["axis"], np.float64), np.asarray(seat["normal"], np.float64)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = np.cross(r, n), r, n, seat["butt"]
    return M


def load_hand_rig(path: str | Path) -> dict | None:
    """hand_rig.json with its arrays (hand_rig.npz) under "arrays"; None when there is no file."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    npz = path.with_suffix(".npz")
    data["arrays"] = dict(np.load(npz)) if npz.exists() else {}
    return data


def side_ok(hand_rig: dict | None, side: str) -> tuple[bool, str | None]:
    """Whether hand_rig can drive this hand, and why not."""
    if not hand_rig:
        return False, "no hand_rig.json"
    if hand_rig.get("version") != HAND_RIG_VERSION:
        return False, f"hand_rig version {hand_rig.get('version')} (expected {HAND_RIG_VERSION})"
    if not hand_rig.get("ok"):
        return False, f"hand_rig not ok: {hand_rig.get('reason')}"
    s = hand_rig.get("sides", {}).get(side)
    if not s or not s.get("ok"):
        return False, f"{side} hand not ok: {(s or {}).get('reason')}"
    if f"{side}_co" not in hand_rig.get("arrays", {}):
        return False, "hand_rig.npz missing"
    return True, None


def _bone_world_rot(rig: rt.Rig, bone: str) -> np.ndarray:
    B = rig.obj_rot @ rig.rest[rig.index[bone], :3, :3]
    return B / np.linalg.norm(B, axis=0)


def seat_frame(hand_rig: dict | None, rig: rt.Rig, mapping: dict[str, str], side: str = "Right") -> dict:
    """The racket's rest world matrix on this hand's palm, and the hand offset that keeps it aimed as in the fixed grip.

    Returns {"source": "palm" | "fixed", "matrix" (4,4) racket rest world, "hand_seat" (4,4) the racket in the
    hand bone's frame (unit scale, origin at the bone head), "offset_world" (3,3), "offset_quat" (w, x, y, z),
    "warning", diagnostics}. offset_world C is the rotation about the wrist joint taking the seated racket frame onto
    the fixed grip's (retarget.grip_matrix, as attach_racket places it: shaft and string normal); offset_quat is the
    same rotation in the hand bone's rest frame, so right-multiplying the hand's pose (or its basis quaternion:
    q' = q * offset_quat, retarget.quat_mul) by it keeps the racket's world orientation in every frame what the fixed
    grip gives (its butt moves a few cm), while the hand turns about the wrist by that constant (ulnar deviation and
    extension, a little twist). Right hand only: the left hand gets its seat (for the two-handed IK target) and no
    offset. Without a usable hand_rig: the fixed grip (right hand), no offset, and a warning."""
    ok, why = side_ok(hand_rig, side)
    hand = mapping.get(f"{side.lower()}_wrist")
    if hand is None:
        return {"source": "fixed", "matrix": None, "warning": f"no {side.lower()} hand bone"}
    H = np.eye(4)
    H[:3, :3] = _bone_world_rot(rig, hand)
    H[:3, 3] = rig.rest_heads_world()[rig.index[hand]]
    fixed = rt.grip_matrix(rig, mapping) if side == "Right" and "pelvis" in mapping else None
    seat, warning = None, None
    if ok:
        try:
            s = hand_rig["sides"][side]
            seat = seat_from_palm(s["palm"], hand_rig["arrays"][f"{side}_palm"])
        except (KeyError, ValueError) as e:
            warning = f"palm seat failed ({e}): the fixed grip"
    else:
        warning = f"{why}: the fixed grip"
    if seat is None:
        out = {"source": "fixed", "matrix": fixed, "offset_world": np.eye(3), "offset_quat": np.array([1.0, 0, 0, 0]),
               "warning": warning}
        if fixed is not None:
            out["hand_seat"] = np.linalg.inv(H) @ fixed
        return out
    M = seat_matrix(seat)
    stored = hand_rig["sides"][side].get("seat")
    if stored and (np.linalg.norm(M[:3, 3] - stored["butt"]) > 1e-3 or rt._angle_deg(M[:3, 1], stored["axis"]) > 0.5):
        warning = "the palm seat differs from the one the grip was solved for (hand_rig built elsewhere?)"
    C = fixed[:3, :3] @ M[:3, :3].T if fixed is not None else np.eye(3)
    B = H[:3, :3]
    q = rt.matrix_to_quat(B.T @ C @ B)
    palm = hand_rig["sides"][side]["palm"]
    cq = rt.matrix_to_quat(C)
    rotvec = cq[1:] / max(np.linalg.norm(cq[1:]), 1e-12) * 2 * math.acos(min(float(cq[0]), 1.0))
    about = {k: round(math.degrees(float(rotvec @ unit(palm[key]))), 2)
             for k, key in (("long_axis", "long_axis"), ("thumb_side", "thumb_side"), ("palm_normal", "normal"))}
    out = {"source": "palm", "matrix": M, "hand_seat": np.linalg.inv(H) @ M, "offset_world": C, "offset_quat": q,
           "warning": warning, "seat": {k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in seat.items()},
           "offset_deg": round(rt.rotation_angle_deg(C), 2), "offset_about_deg": about}
    if fixed is not None:
        out["fixed_matrix"] = fixed
        out["axis_vs_fixed_deg"] = round(rt._angle_deg(C @ M[:3, 1], fixed[:3, 1]), 3)
        out["normal_vs_fixed_deg"] = round(rt._angle_deg(C @ M[:3, 2], fixed[:3, 2]), 2)
        out["uncompensated_axis_vs_fixed_deg"] = round(rt._angle_deg(M[:3, 1], fixed[:3, 1]), 2)
        out["butt_shift_m"] = round(float(np.linalg.norm(H[:3, 3] + C @ (M[:3, 3] - H[:3, 3]) - fixed[:3, 3])), 4)
    return out


def palm_in_bone(hand_rig: dict | None, rig: rt.Rig, side: str) -> tuple[np.ndarray, np.ndarray] | None:
    """(palm centre on the skin, outward palm normal) of one hand in its hand bone's rest frame (twohand's palm
    arguments), or None without a usable hand. The stored centre lies mid-way through the hand; the point returned is
    the palm skin over it: the skin points' 97th percentile height along the normal within PALM_SKIN_NEAR_M of it."""
    if not side_ok(hand_rig, side)[0] or hand_rig["sides"][side]["hand_bone"] not in rig.index:
        return None
    s = hand_rig["sides"][side]
    palm = s["palm"]
    c = palm_coords(palm, palm["centre"])
    q = palm_coords(palm, hand_rig["arrays"][f"{side}_palm"])
    near = np.hypot(q[:, 0] - c[0], q[:, 1] - c[1]) < PALM_SKIN_NEAR_M
    if near.sum() >= 5:
        c[2] = np.percentile(q[near, 2], 97)
    F = np.stack([unit(palm["long_axis"]), unit(palm["thumb_side"]), unit(palm["normal"])], 1)
    R = _bone_world_rot(rig, s["hand_bone"])
    head = rig.rest_heads_world()[rig.index[s["hand_bone"]]]
    return R.T @ (np.asarray(palm["wrist"], np.float64) + F @ c - head), R.T @ unit(palm["normal"])


def left_ik_palm(hand_rig: dict | None, rig: rt.Rig, mapping: dict[str, str], above_m: float) -> dict | None:
    """twohand.left_hand_on_racket's palm arguments that put the handle where the left hand's grip was solved (its
    seat): palm_centre the point of the seat's axis nearest the palm centre, moved above_m (twohand's
    handle_above_palm_m) back along the seat normal; palm_normal that normal; handle_dir the shaft. All in the
    left-hand bone's frame; None without a usable left hand (twohand then takes its default palm)."""
    pc = palm_in_bone(hand_rig, rig, "Left")
    s = seat_frame(hand_rig, rig, mapping, "Left") if pc is not None else {}
    if s.get("source") != "palm":
        return None
    S = s["hand_seat"]
    butt, d, n = S[:3, 3], S[:3, 1], S[:3, 2]
    return {"palm_centre": butt + d * float((pc[0] - butt) @ d) - n * above_m, "palm_normal": n, "handle_dir": d}


def hand_points(hand_rig: dict | None, rig: rt.Rig, side: str, co: np.ndarray, weights: np.ndarray,
                closings: tuple[float, ...] = HAND_POINT_CLOSINGS,
                every: int = HAND_POINT_EVERY) -> np.ndarray | None:
    """(N,K,3) one hand's skin in its hand bone's rest frame at each finger closing of `closings` (twohand's
    hand_points, which the left-hand IK pushes off the racket), or None without a usable hand: every `every`-th of
    the hand's vertices (co (V,3) rest world positions, weights (V,B) over the rig's bones, as GripProbe reads
    them), skinned as Blender does with the hand bone at rest and its finger bones posed by finger_pose_quats;
    the weight off the hand's bones (the forearm's share at the wrist) follows the hand bone."""
    if not side_ok(hand_rig, side)[0] or hand_rig["sides"][side]["hand_bone"] not in rig.index or not len(co):
        return None
    name = hand_rig["sides"][side]["hand_bone"]
    hand = rig.index[name]
    fingers = sorted((rig.index[n] for n in bone_names(name) if n in rig.index), key=rig.order.index)
    bones = [hand, *fingers]
    W = np.asarray(weights, np.float64)[::every]
    x = np.c_[rig.to_armature(np.asarray(co, np.float64)[::every]), np.ones(len(W))]
    share = W[:, bones].copy()
    share[:, 0] += W.sum(1) - share.sum(1)
    share /= np.maximum(share.sum(1, keepdims=True), 1e-12)
    inv_rest = np.linalg.inv(rig.rest)
    rest_rot = bone_rest(rig)
    out = []
    for amount in closings:
        quats = finger_pose_quats(hand_rig, side, amount, rest_rot)
        pose = {hand: rig.rest[hand]}
        for b in fingers:
            basis = np.eye(4)
            basis[:3, :3] = rt.quat_to_matrix(np.asarray(quats[rig.names[b]], np.float64)[None])[0]
            pose[b] = pose[rig.parents[b]] @ inv_rest[rig.parents[b]] @ rig.rest[b] @ basis
        X = sum(share[:, i, None] * (x @ (pose[b] @ inv_rest[b]).T) for i, b in enumerate(bones))
        out.append((X @ inv_rest[hand].T)[:, :3])
    return np.stack(out)


def inside_handle(co: np.ndarray, racket: np.ndarray, radius: float = HANDLE_RADIUS,
                  length: float = QA_HANDLE_M) -> np.ndarray:
    """Which points lie inside the handle: a cylinder of `radius` along the racket's first `length` from the butt."""
    rel = np.asarray(co, np.float64) - racket[:3, 3]
    y = rel @ racket[:3, 1]
    radial = np.linalg.norm(rel - np.outer(y, racket[:3, 1]), axis=1)
    return (y >= 0) & (y <= length) & (radial < radius)


def inside_racket(co: np.ndarray, racket: np.ndarray) -> np.ndarray:
    """Which points lie inside the lower racket (racket.py's shapes): the butt cap and the grip as cylinders to
    HANDLE_LENGTH, the shaft as a box to THROAT_Y, the throat's two arms as boxes to the head."""
    rel = np.asarray(co, np.float64) - racket[:3, 3]
    x, y, z = rel @ racket[:3, 0], rel @ racket[:3, 1], rel @ racket[:3, 2]
    radial = np.hypot(x, z)
    inside = (y >= 0) & (y <= HANDLE_LENGTH) & (radial < np.where(y < BUTT_LENGTH, BUTT_RADIUS, HANDLE_RADIUS))
    inside |= (y > HANDLE_LENGTH) & (y <= THROAT_Y) & (np.abs(x) < SHAFT_HALF[0]) & (np.abs(z) < SHAFT_HALF[1])
    length = math.hypot(THROAT_ARM_END[0], THROAT_ARM_END[1] - THROAT_Y)
    for sign in (-1.0, 1.0):
        u = np.array([sign * THROAT_ARM_END[0], THROAT_ARM_END[1] - THROAT_Y]) / length
        along = x * u[0] + (y - THROAT_Y) * u[1]
        across = np.abs(x * u[1] - (y - THROAT_Y) * u[0])
        inside |= (along >= 0) & (along <= length) & (across < THROAT_ARM_HALF[0]) & (np.abs(z) < THROAT_ARM_HALF[1])
    return inside


def palm_gap(points: np.ndarray, racket: np.ndarray, span: float = 0.12, near: float = 0.03) -> np.ndarray:
    """Gaps (m) from palm skin points to the handle's surface (negative: the handle is inside the skin), for the points
    over the handle's first `span` from the butt and within `near` of its axis: the butt cap (BUTT_RADIUS) over its
    BUTT_LENGTH, the grip (HANDLE_RADIUS) past it."""
    rel = np.asarray(points, np.float64) - racket[:3, 3]
    y = rel @ racket[:3, 1]
    radial = np.linalg.norm(rel - np.outer(y, racket[:3, 1]), axis=1)
    m = (y > 0) & (y < span) & (radial < near)
    return radial[m] - np.where(y[m] < BUTT_LENGTH, BUTT_RADIUS, HANDLE_RADIUS)


# ---------------------------------------------------------------------------------------------- finger pose
def finger_pose_quats(hand_rig: dict, side: str, amount, bone_rest: dict | None = None) -> dict[str, np.ndarray]:
    """Pose-bone quaternions (w, x, y, z; Blender's rotation_quaternion) of one hand's 15 finger bones, slerped
    from the relaxed pose (amount 0) to the grip (1). `amount` is a scalar or an (L,) track -> (4,) or (L,4) each.
    The stored rotations are about each bone's head in the rest world frame; a bone's local rotation is
    B^T R B with B its world rest rotation, from the stored head / tail / Z axis or, given, `bone_rest` (name ->
    (3,3), e.g. from arm.data.bones after add_finger_bones)."""
    out = {}
    for b in hand_rig["sides"][side]["bones"]:
        B = np.asarray(bone_rest[b["name"]], np.float64) if bone_rest else bone_frame(b["head"], b["tail"], b["z_axis"])
        B = B / np.linalg.norm(B, axis=0)
        q0, q1 = (rt.matrix_to_quat(B.T @ rt.quat_to_matrix(np.asarray(b[k], np.float64)) @ B)
                  for k in ("relaxed_world", "grip_world"))
        out[b["name"]] = quat_slerp(q0, q1, amount)
    return out


def finger_tracks(hand_rig: dict, amounts: dict, frames: int,
                  bone_rest: dict | None = None) -> tuple[list[str], np.ndarray]:
    """(names, (frames, K, 4) quaternions) of the finger bones of the usable hands, e.g. amounts {"Right": 1.0,
    "Left": the IK's finger closing}: what retarget() adds to its animated bones (keys and FK)."""
    names, tracks = [], []
    for side in SIDES:
        if side not in amounts or not side_ok(hand_rig, side)[0]:
            continue
        amount = np.broadcast_to(np.asarray(amounts[side], np.float64), (frames,))
        for name, q in finger_pose_quats(hand_rig, side, amount, bone_rest).items():
            names.append(name)
            tracks.append(q)
    return names, (np.stack(tracks, 1) if tracks else np.zeros((frames, 0, 4)))


# ============================================================================== Blender side (bpy below)
def add_finger_bones(arm, hand_rig: dict) -> list[str]:
    """15 deforming bones per usable hand, in edit mode, at the stored joints: each finger a chain under the hand
    bone (not connected, so every head is exactly its joint), Z axis towards the palm side. Returns their names."""
    sides = [s for s in SIDES if side_ok(hand_rig, s)[0] and hand_rig["sides"][s]["hand_bone"] in arm.data.bones]
    if not sides:
        return []
    inv = arm.matrix_world.inverted()
    before = {b.name: np.array(b.matrix_local) for b in arm.data.bones}
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    names = []
    for side in sides:
        s = hand_rig["sides"][side]
        for b in s["bones"]:
            e = eb.get(b["name"]) or eb.new(b["name"])
            e.head = inv @ Vector(b["head"])
            e.tail = inv @ Vector(b["tail"])
            e.align_roll(inv.to_3x3() @ Vector(b["z_axis"]))
            e.parent = eb[b["parent"]]
            e.use_connect = False
            e.use_deform = True
            names.append(b["name"])
    bpy.ops.object.mode_set(mode="OBJECT")
    moved = max(float(np.abs(np.array(arm.data.bones[n].matrix_local) - M).max()) for n, M in before.items())
    if moved > 1e-5:  # the edit-mode round trip must leave the auto-rig's own bones where they were
        rt._log(f"add_finger_bones: edit mode moved existing bones by up to {moved:.2e}")
    for name in names:
        arm.pose.bones[name].rotation_mode = "QUATERNION"
    return names


def finger_weights(hand_rig: dict, meshes=None) -> dict:
    """Split every skinned mesh's hand weights over the finger bones (after add_finger_bones): each vertex carrying
    hand weight takes the shares of the stored rest vertex at its position (MATCH_TOL_M; a welded or smoothed skin
    still matches), its hand + finger total stays what the hand had, so the wrist blend is untouched. Vertices with
    no stored neighbour (weight smoothed up the forearm) stay on the hand. Per mesh and hand: counts."""
    out = {}
    for o in meshes or rt._skinned_meshes():
        res, co = {}, None
        for side in SIDES:
            if not side_ok(hand_rig, side)[0]:
                continue
            s, arrays = hand_rig["sides"][side], hand_rig["arrays"]
            names = [s["hand_bone"], *(b["name"] for b in s["bones"])]
            if names[0] not in o.vertex_groups:
                continue
            co = rt._world_co(o) if co is None else co
            W = rt._read_weights(o, names)
            verts = np.flatnonzero(W.sum(1) > 0)
            shares, dist = match_shares(co[verts], arrays[f"{side}_co"], arrays[f"{side}_shares"])
            new = split_hand_weight(W[verts].sum(1), shares)
            matched = dist <= MATCH_TOL_M
            res[side] = {
                "vertices": int(len(verts)), "matched": int(matched.sum()),
                "exact": int((dist < 1e-5).sum()), "changed": rt._write_weights(o, names, verts, W[verts], new),
                "mostly_finger": int((shares[:, 1:].sum(1) > 0.5).sum()),
                "per_finger": {f: int((shares[:, 1 + 3 * k:4 + 3 * k].sum(1) > 0.5).sum())
                               for k, f in enumerate(FINGERS)},
            }
        if res:
            out[o.name] = res
    return out


def export_hand_mesh(rig_path: str, out: str, weld_m: float) -> dict:
    """The rig as retarget.build sees it (imported, unit scale, welded with the job's weld_m): the vertices with any
    hand or forearm weight and their triangles (rest world), those weights, every bone's rest world matrix and
    retarget.map_bones' mapping, as an .npz for rybakina3d.character.hands."""
    rt._reset_scene(25.0)
    arm = rt._import_rig(rig_path)
    weld = rt.weld_meshes(weld_m) if weld_m > 0 else None
    rig = rt.rig_from_blender(arm)
    mapping, _ = rt.map_bones(rig.names, rig.parents)
    joints = ("right_wrist", "left_wrist", "right_elbow", "left_elbow")
    missing = [j for j in joints if j not in mapping]
    if missing:
        raise RuntimeError(f"{rig_path}: no bone for {missing}")
    groups = [mapping[j] for j in joints]
    cos, tris, weights, total, offset = [], [], [], 0, 0
    for o in rt._skinned_meshes():
        W = rt._read_weights(o, groups)
        keep = W.sum(1) > 0
        total += len(W)
        if not keep.any():
            continue
        me = o.data
        me.calc_loop_triangles()
        tri = np.empty(len(me.loop_triangles) * 3, np.int64)
        me.loop_triangles.foreach_get("vertices", tri)
        tri = tri.reshape(-1, 3)
        remap = np.full(len(W), -1, np.int64)
        remap[keep] = offset + np.arange(int(keep.sum()))
        tri = remap[tri[keep[tri].all(1)]]
        cos.append(rt._world_co(o)[keep])
        tris.append(tri)
        weights.append(W[keep])
        offset += int(keep.sum())
    if not cos:
        raise RuntimeError(f"{rig_path}: no skinned vertex carries hand or forearm weight")
    np.savez_compressed(out, co=np.concatenate(cos), tri=np.concatenate(tris), weights=np.concatenate(weights),
                        groups=np.array(groups), bones=np.array(rig.names), parents=np.array(rig.parents),
                        rest_world=rig.obj @ rig.rest, mapping=json.dumps(mapping), vertices=np.int64(total),
                        weld=json.dumps(weld), blender=bpy.app.version_string)
    rt._log(f"hand export: {offset} of {total} vertices, weld {weld_m} -> {out}")
    return {"vertices": total, "exported": offset}


def main(argv: list[str]) -> None:
    if argv[:1] == ["export"] and len(argv) == 4:
        export_hand_mesh(argv[1], argv[2], float(argv[3]))
    else:
        raise SystemExit("usage: fingers.py -- export rig.glb out.npz weld_m")


if __name__ == "__main__" and bpy is not None:
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
