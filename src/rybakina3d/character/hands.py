"""Finger joints, finger weights, the racket's seat and a grip pose from the rigged character's own hand mesh (numpy +
scipy, and Blender for the mesh export) -> character/hand_rig.json + hand_rig.npz next to rigged.glb. `r3d retarget`
runs it first, on whichever machine runs that stage, when a hand setting is on and the file is not fresh
(assemble.ensure_hand_rig); by hand: `python -m rybakina3d.character.hands --config <yaml>`.

The auto-rig has no finger bones and the meshes are modelled with open hands. Per hand, on the welded rest mesh
exactly as retarget.build sees it (blender/fingers.py `export`, the job's weld_m):
1. Hand frame: PCA of the strongly hand-weighted vertices (long axis a: wrist -> fingers; thickness n towards the
   body midline at rest; thumb side t = a x n on the right hand, n x a on the left).
2. Fingertips: the farthest points from the wrist over the mesh surface (geodesic maxima); the thumb is the most
   thumb-side tip, the four longest others are index -> pinky. Every vertex belongs softly to the finger whose tip
   is nearest over the surface.
3. Joints: each finger's centre line from rings of equal distance from its tip; the joints at standard proportions
   of the hand length (Buchholz, Armstrong & Goldstein 1992), the knuckles snapped to mid-thickness of the hand.
4. Weights: smoothstep bands at the joints (fingers.band_shares), stored as each vertex's split of its hand weight.
5. Palm frame from the mesh: the palm normal fitted to the palm-side skin; palm centre; where the wrist is clear
   (the palm-side skin lowest past the wrist joint: past a wristband); the heel pad (the hypothenar, a fifth of the
   knuckle span in from the little finger, just past the clear wrist); the index knuckle.
6. Seat (fingers.seat_from_palm; a seat at one of its limits is logged) and grip: each finger curled (scipy Powell
   over its joint angles, linear-blend skinning of the bands) until every phalanx rests on the seated handle without
   entering it or the palm; the thumb turns about the shaft to clear the handle and wraps over its top. A relaxed
   pose (a mild curl) for the hand off the racket.
7. Checks: enough vertices, five tips, free (unwebbed) fingers, plausible segment lengths, finger order, joints
   inside the mesh, a seat, a grip that keeps the handle clear, left/right agreement. A failure writes "ok": false
   with the reason (no arrays), and retarget then keeps the fixed grip there (no finger bones), with a warning.

Coordinates are the rig's rest world frame after import (metres, Z-up); nothing refers to vertex indices.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from ..blender import SCRIPTS_DIR
from ..blender import fingers as fg
from ..blender import retarget as rt
from ..config import PROJECT_ROOT, Config
from ..util import log, step, write_json
from .mesh import _blender_local

# segment length / hand length (wrist crease -> middle tip), Buchholz et al. 1992: distal segment first
RATIOS = {"thumb": (0.158, 0.196, 0.251), "index": (0.097, 0.143, 0.265), "middle": (0.108, 0.170, 0.277),
          "ring": (0.107, 0.165, 0.259), "pinky": (0.093, 0.117, 0.206)}
HL_SCALE = 0.9               # the auto-rig's wrist joint sits ~1-2 cm proximal of the wrist crease
HAND_WEIGHT_MIN = 0.02       # vertices with more hand weight make up the hand mesh
MIN_HAND_VERTICES = 1500     # fewer: a coarse hand (fused fingers, a few edge loops) that does not bend believably
TIP_SPACING_M = 0.028        # fingertips at least this far apart over the surface
FREE_MIN = 0.45              # each finger free of its neighbours over at least this share of its bone chain
SEGMENT_RANGE = (0.5, 1.8)   # segment length / its standard proportion
HEEL_ACROSS = 0.2            # heel pad: this share of the knuckle span from the little finger's knuckle line
HEEL_PAST_CLEAR_M = 0.015    # and this far past where the wrist is clear
MIN_KNUCKLE_GAP_M = 0.008    # neighbouring knuckles at least this far apart across the hand
MIRROR_MAX_M = 0.025         # a joint of the left hand (in its palm frame) this close to the right's
MIRROR_SEGMENT = 0.25        # and its segments within this share of the right's lengths
CELL_M = 0.005               # palm skin grid
GRIP_INSIDE_MAX = 0.02       # share of the hand's vertices the gripping hand may leave inside the handle
# grip solve
GRIP_RADIUS = fg.HANDLE_RADIUS + 0.001   # the fingers rest on an overgrip
MAX_FLEX_DEG = (85.0, 105.0, 75.0)       # knuckle, middle and end joint
# thumb on the handle: its base turns about the shaft (+ over the top of the handle, - out to clear it) and across
# the palm (about the palm normal, + towards the index), its middle and end joints turn about the shaft too
THUMB_BOUNDS_DEG = ((-40, 40), (-25, 15), (-30, 60), (0, 80))
THUMB_STARTS = ((-0.3, 0.0, 0.0, 0.8), (-0.2, -0.2, 0.2, 1.0), (0.0, 0.0, 0.0, 0.5))
PALM_CLEAR_M = 0.002         # curled fingertips stay this far above the palm skin
RELAXED_DEG = {"finger": (15.0, 25.0, 12.0), "thumb": (5.0, 8.0, 8.0)}


class HandRigError(ValueError):
    """The hand mesh does not give a believable finger rig (the message is the reason)."""


# ------------------------------------------------------------------------------------------- mesh helpers
def dedupe(co: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coincident vertices (an unwelded glTF splits them at seams) merged for the surface graph: (unique positions,
    triangles on them, index of each input vertex's unique position)."""
    key = np.round(np.asarray(co, np.float64) * 1e6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    t = inv[tri]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return np.asarray(co, np.float64)[first], t, inv


def surface_graph(P: np.ndarray, tri: np.ndarray):
    """Sparse edge-length graph of a triangle mesh (each edge once)."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(P[e[:, 0]] - P[e[:, 1]], axis=1)
    n = len(P)
    return coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()


def winding_number(P: np.ndarray, tri: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Generalized winding number of points q (M,3) w.r.t. a triangle mesh: ~1 inside, ~0 outside (robust to the
    hand being cut open at the wrist, far from the finger joints)."""
    out = np.zeros(len(q))
    for i, p in enumerate(np.asarray(q, np.float64)):
        a, b, c = P[tri[:, 0]] - p, P[tri[:, 1]] - p, P[tri[:, 2]] - p
        la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb \
            + np.einsum("ij,ij->i", b, c) * la
        out[i] = np.sum(2 * np.arctan2(det, den)) / (4 * math.pi)
    return out


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    return rt.axis_angle_to_matrix(fg.unit(axis) * angle)


def _chain(J: np.ndarray, Rs: list[np.ndarray]) -> list[np.ndarray]:
    """World delta transforms (4,4) of a finger's three bones for joint rotations Rs about the rest joints."""
    out, M = [], np.eye(4)
    for i in range(3):
        T = np.eye(4)
        T[:3, :3] = Rs[i]
        T[:3, 3] = J[i] - Rs[i] @ J[i]
        M = M @ T
        out.append(M)
    return out


# ------------------------------------------------------------------------------------------------ analysis
def _tips(P: np.ndarray, G, g: np.ndarray) -> list[int]:
    """Geodesic maxima: vertices farther from the wrist (over the surface) than every vertex within 1 cm."""
    tree = cKDTree(P)
    taken = np.zeros(len(P), bool)
    tips = []
    for i in np.argsort(-g):
        if g[i] < 0.35 * g.max() or len(tips) >= 10:
            break
        if taken[i]:
            continue
        near = tree.query_ball_point(P[i], 0.010)
        if g[near].max() > g[i] + 1e-9:
            taken[near] = True
            continue
        tips.append(int(i))
        taken |= np.isfinite(dijkstra(G, indices=i, limit=TIP_SPACING_M))
    return tips


def _centerline(P: np.ndarray, tip: int, mask: np.ndarray, step: float = 0.003, max_d: float = 0.16):
    """Ring centres, mean ring radii and distances from the tip along one finger (rings of equal distance)."""
    dist = np.linalg.norm(P - P[tip], axis=1)
    pts, rad, ds = [], [], []
    for d0 in np.arange(0.0, max_d, step):
        m = mask & (dist >= d0) & (dist < d0 + step)
        if m.sum() < 3:
            continue
        c = P[m].mean(0)
        pts.append(c)
        rad.append(np.linalg.norm(P[m] - c, axis=1).mean())
        ds.append(d0 + step / 2)
    return np.array(pts), np.array(rad), np.array(ds)


def _finger_joints(f: str, P: np.ndarray, lab: np.ndarray, k: int, tip: int, wrist: np.ndarray,
                   frame: tuple[np.ndarray, np.ndarray, np.ndarray], HL: float) -> dict:
    a, t, n = frame
    pts, rad, ds = _centerline(P, tip, lab == k)
    if len(ds) < 8:
        raise HandRigError(f"{f}: no finger centre line")
    rmed = float(np.median(rad[(ds > 0.01) & (ds < 0.045)]))
    free = (ds > 0.006) & (rad < 1.5 * rmed)          # rings before the radius jumps (the palm)
    ends = ~free[1:] & (ds[1:] > 0.03)
    cut = int(np.argmax(ends)) if ends.any() else len(ds) - 1
    fp, fd = pts[:cut + 1], ds[:cut + 1]
    web = float(fd.max())
    tail = fp[(fd > max(0.02, web - 0.025)) & (fd < web - 0.003)]
    if len(tail) < 2:
        tail = fp[-3:]
    _, _, vt = np.linalg.svd(tail - tail.mean(0))
    tdir = vt[0] * np.sign(vt[0] @ (tail[-1] - tail[0]))

    def at(d_tip: float) -> np.ndarray:
        if d_tip <= web:
            return fp[int(np.argmin(np.abs(fd - d_tip)))]
        return fp[-1] + tdir * (d_tip - web)          # inside the palm: on along the finger's own axis

    rr = RATIOS[f]
    if f == "thumb":
        mcp = at(HL * (rr[0] + rr[1]))
        goal = wrist + t * 0.012 + n * 0.008             # the thumb's base sits by the wrist, thumb side, palm side
        J = [mcp + fg.unit(goal - mcp) * HL * rr[2], mcp, at(HL * rr[0]), P[tip]]
    else:
        J = [at(HL * sum(rr)), at(HL * (rr[0] + rr[1])), at(HL * rr[0]), P[tip]]
        # knuckle (and a middle joint still inside the palm): across the hand on the finger's own line, in depth
        # at the hand's local mid-thickness
        m = (fd > 0.015) & (fd < max(0.035, web - 0.01))
        fa, ft = (fp[m] - wrist) @ a, (fp[m] - wrist) @ t
        k1, k0 = np.polyfit(fa, ft, 1) if m.sum() > 2 else (0.0, float(ft.mean()) if len(ft) else 0.0)
        for ji in (0, 1):
            if ji == 1 and HL * (rr[0] + rr[1]) <= web:
                continue
            qa = (J[ji] - wrist) @ a
            qt = k0 + k1 * qa
            near = (np.abs((P - wrist) @ a - qa) < 0.006) & (np.abs((P - wrist) @ t - qt) < 0.006)
            pn = (P[near] - wrist) @ n
            qn = 0.5 * (pn.min() + pn.max()) if near.sum() > 4 else (J[ji] - wrist) @ n
            J[ji] = wrist + a * qa + t * qt + n * qn
    return {"joints": np.array(J), "free_m": web, "radius_m": rmed}


def _skin_grid(S: np.ndarray, lo: tuple[float, float], hi: tuple[float, float]):
    """Per CELL_M cell of (s, w): highest and lowest h of the points (palm coords) inside; NaN where empty."""
    ns, nw = int(math.ceil((hi[0] - lo[0]) / CELL_M)), int(math.ceil((hi[1] - lo[1]) / CELL_M))
    i = np.floor((S[:, 0] - lo[0]) / CELL_M).astype(int)
    j = np.floor((S[:, 1] - lo[1]) / CELL_M).astype(int)
    ok = (i >= 0) & (i < ns) & (j >= 0) & (j < nw)
    top, bottom = np.full((ns, nw), -np.inf), np.full((ns, nw), np.inf)
    np.maximum.at(top, (i[ok], j[ok]), S[ok, 2])
    np.minimum.at(bottom, (i[ok], j[ok]), S[ok, 2])
    top[~np.isfinite(top)] = np.nan
    bottom[~np.isfinite(bottom)] = np.nan
    return top, bottom, (i, j, ok)


def analyse_hand(mesh: dict, side: str) -> dict:
    """One hand of an exported mesh (hands mesh npz as a dict) -> joints, shares, palm frame. Raises HandRigError."""
    mapping = json.loads(str(mesh["mapping"]))
    groups = [str(g) for g in mesh["groups"]]
    bones = [str(b) for b in mesh["bones"]]
    hand_bone = mapping[f"{side.lower()}_wrist"]
    H = np.asarray(mesh["rest_world"][bones.index(hand_bone)], np.float64)
    wrist = H[:3, 3]
    pelvis = np.asarray(mesh["rest_world"][bones.index(mapping["pelvis"])], np.float64)[:3, 3]
    co, tri, inv = dedupe(mesh["co"], mesh["tri"])
    W = np.zeros((len(co), mesh["weights"].shape[1]))
    np.maximum.at(W, inv, np.asarray(mesh["weights"], np.float64))
    wh_all = W[:, groups.index(hand_bone)]
    sel = wh_all > HAND_WEIGHT_MIN
    if sel.sum() < MIN_HAND_VERTICES:
        raise HandRigError(f"coarse hand: {int(sel.sum())} vertices (< {MIN_HAND_VERTICES})")
    vid = np.flatnonzero(sel)
    remap = np.full(len(co), -1)
    remap[vid] = np.arange(len(vid))
    t_sub = remap[tri[sel[tri].all(1)]]
    P, wh = co[vid], wh_all[vid]
    G = surface_graph(P, t_sub)
    _, comp = connected_components(G, directed=False)
    main = comp == np.argmax(np.bincount(comp))

    # 1. hand frame
    core = main & (wh > 0.5)
    c = P[core].mean(0)
    _, _, Vt = np.linalg.svd(P[core] - c, full_matrices=False)
    a = Vt[0] * np.sign(Vt[0] @ (c - wrist))
    to_mid = pelvis - wrist
    to_mid[2] = 0
    n = Vt[2] - (Vt[2] @ a) * a
    n = fg.unit(n * np.sign(n @ to_mid))
    t = np.cross(a, n) if side == "Right" else np.cross(n, a)

    # 2. fingertips and soft finger labels
    s = (P - wrist) @ a
    seeds = np.flatnonzero(main & (s < s[main].min() + 0.01))
    g = dijkstra(G, indices=seeds, min_only=True)
    g[~np.isfinite(g)] = -1
    tips = _tips(P, G, g)
    if len(tips) < 5:
        raise HandRigError(f"{len(tips)} fingertips found")
    lat = {i: (P[i] - c) @ t for i in tips}
    thumb = max(tips, key=lambda i: lat[i])
    rest = sorted(sorted((i for i in tips if i != thumb), key=lambda i: -s[i])[:4], key=lambda i: -lat[i])
    tip = dict(zip(fg.FINGERS, [thumb, *rest], strict=True))
    if s[thumb] >= min(s[i] for i in rest) or lat[thumb] <= lat[rest[0]]:
        raise HandRigError("no thumb apart from the fingers")
    HL = float((P[tip["middle"]] - wrist) @ a) * HL_SCALE
    gk = np.stack([dijkstra(G, indices=tip[f]) for f in fg.FINGERS], 1)
    gk[~np.isfinite(gk)] = 1e3
    lab = np.argmin(gk, 1)
    mid = lab == 2
    stretch = np.median(gk[mid, 2] / np.maximum(np.linalg.norm(P[mid] - P[tip["middle"]], axis=1), 1e-4))
    soft = np.exp(-(gk - gk.min(1, keepdims=True)) / (0.004 * stretch))
    soft /= soft.sum(1, keepdims=True)

    # 3. joints, 4. bands
    fingers = {f: _finger_joints(f, P, lab, k, tip[f], wrist, (a, t, n), HL) for k, f in enumerate(fg.FINGERS)}
    joints = {f: v["joints"] for f, v in fingers.items()}
    if not all(np.isfinite(j).all() for j in joints.values()):
        raise HandRigError("non-finite finger joints")
    shares = fg.band_shares(P, joints, {f: v["radius_m"] for f, v in fingers.items()}, soft,
                            {f: v["free_m"] for f, v in fingers.items()})

    # 5. palm frame from the palm-side skin
    knuckles = np.array([(joints[f][0] - wrist) @ np.stack([a, t, n], 1) for f in fg.FINGERS[1:]])
    s_mcp, w_index, w_pinky = knuckles[:, 0].min(), knuckles[0, 1], knuckles[-1, 1]
    palm_proper = (shares[:, 0] + shares[:, 1] > 0.5) & (wh > 0.5) & main
    palm_any = (shares[:, 0] + shares[:, 1] > 0.5) & main
    F0 = np.stack([a, t, n], 1)
    S = (P - wrist) @ F0
    box = palm_proper & (S[:, 0] > 0.03) & (S[:, 0] < s_mcp - 0.01) & (S[:, 1] > w_pinky) & (S[:, 1] < w_index)
    top, _, (ci, cj, cok) = _skin_grid(S[box], (0.0, w_pinky), (s_mcp, w_index))
    cell_top = top[ci[cok], cj[cok]]
    palm_side = np.flatnonzero(box)[cok][S[box][cok, 2] > cell_top - 0.003]
    if len(palm_side) < 30:
        raise HandRigError(f"{len(palm_side)} palm-side skin vertices")
    q = P[palm_side]
    _, _, vt = np.linalg.svd(q - q.mean(0), full_matrices=False)
    n = fg.unit(vt[2] * np.sign(vt[2] @ n))
    a = fg.unit(a - (a @ n) * n)
    t = np.cross(a, n) if side == "Right" else np.cross(n, a)
    F = np.stack([a, t, n], 1)
    S = (P - wrist) @ F
    # palm-side skin height across the knuckle span, every 5 mm along the hand (median of the slice's upper half):
    # the wrist is clear (past a wristband) where it is lowest between the wrist joint and mid-palm
    ss = np.arange(0.0, 0.0651, 0.005)
    span = palm_any & (S[:, 1] > w_pinky) & (S[:, 1] < w_index)
    top_s, thick = [], []
    for s0 in ss:
        z = S[span & (np.abs(S[:, 0] - s0) < 0.0025), 2]
        if len(z) < 6:
            top_s.append(np.inf)
            continue
        top_s.append(np.median(z[z > 0.5 * (z.min() + z.max())]))
        thick.append(np.percentile(z, 95) - np.percentile(z, 5))
    top_s = np.convolve(np.r_[top_s[0], top_s, top_s[-1]], np.ones(3) / 3, "valid")
    wrist_clear = float(ss[int(np.argmin(top_s))])
    # heel pad: the hypothenar's centre, HEEL_ACROSS of the knuckle span from the little finger's knuckle line,
    # HEEL_PAST_CLEAR past the clear wrist; the palm-side skin point there
    heel_sw = np.array([wrist_clear + HEEL_PAST_CLEAR_M, w_pinky + HEEL_ACROSS * (w_index - w_pinky)])
    side_skin = np.flatnonzero(palm_proper & (S[:, 2] > np.nanmedian(top) - 0.02))
    heel = P[side_skin[int(np.argmin(np.linalg.norm(S[side_skin, :2] - heel_sw, axis=1)))]]
    centre_box = box & (S[:, 0] > wrist_clear)
    centre = wrist + F @ np.array([S[centre_box, 0].mean(), S[centre_box, 1].mean(), S[centre_box, 2].mean()])
    skin = palm_proper & (S[:, 0] > wrist_clear - 0.01)
    palm = {"wrist": wrist, "long_axis": a, "thumb_side": t, "normal": n, "centre": centre, "heel": heel,
            "index_knuckle": joints["index"][0], "wrist_clear_s": wrist_clear,
            "thickness_m": round(float(np.median(thick)), 4) if thick else None,
            "width_m": float(w_index - w_pinky + fingers["index"]["radius_m"] + fingers["pinky"]["radius_m"]),
            "hand_bone_angle_deg": round(rt._angle_deg(a, H[:3, 1]), 2)}
    return {"side": side, "hand_bone": hand_bone, "P": P, "tri": t_sub, "wh": wh, "shares": shares,
            "fingers": fingers, "joints": joints, "palm": palm, "skin": P[skin], "HL": HL,
            "tip_count": len(tips), "palm_proper": palm_proper}


# ------------------------------------------------------------------------------------------------- grip
class _Hand:
    """Linear-blend skinning of one hand's finger bands about the rest pose (the hand bone itself stays put)."""

    def __init__(self, h: dict, handle: np.ndarray | None = None):
        self.P, self.wh, self.S, self.J = h["P"], h["wh"], h["shares"], h["joints"]
        palm = h["palm"]
        self.a, self.t, self.n = (np.asarray(palm[k]) for k in ("long_axis", "thumb_side", "normal"))
        self.R = {f: [np.eye(3)] * 3 for f in fg.FINGERS}
        self.T = {f: _chain(self.J[f], self.R[f]) for f in fg.FINGERS}
        if handle is not None:  # thumb axes: the shaft, turning the thumb side of the handle over its top; the palm
            J, r = self.J["thumb"], handle[:3, 1]   # normal
            side = J[0] - handle[:3, 3]
            side -= (side @ r) * r + (side @ self.n) * self.n
            self.wrap = r if (_rot(r, 0.3) @ side) @ self.n > 0 else -r
            y = J[3] - J[0]
            self.across = self.n if (_rot(self.n, 0.3) @ y) @ self.t < y @ self.t else -self.n

    def flex_axis(self, f: str, i: int) -> np.ndarray:
        """Axis about which joint i of finger f bends its bone towards the palm side."""
        y = fg.unit(self.J[f][i + 1] - self.J[f][i])
        x = fg.unit(np.cross(y, self.n))
        return x if (_rot(x, 0.3) @ y) @ self.n > y @ self.n else -x

    def rotations(self, f: str, th) -> list[np.ndarray]:
        """Joint rotations (rest world frame) for angles th: three flexions; the gripping thumb (four angles) swings
        over the shaft, across the palm, then wraps round the shaft at its middle and end joints."""
        if f == "thumb" and len(th) == 4:
            return [_rot(self.wrap, th[0]) @ _rot(self.across, th[1]), _rot(self.wrap, th[2]),
                    _rot(self.wrap, th[3])]
        return [_rot(self.flex_axis(f, i), th[i]) for i in range(3)]

    def set(self, f: str, Rs: list[np.ndarray]) -> None:
        self.R[f] = Rs
        self.T[f] = _chain(self.J[f], Rs)

    def posed(self, idx=slice(None), only: str | None = None, base: np.ndarray | None = None) -> np.ndarray:
        """Posed positions of vertices idx; with `only`, finger `only` moves on top of `base` (the other fingers'
        displacement, precomputed)."""
        P, w = self.P[idx], self.wh[idx]
        out = P.copy() if base is None else base.copy()
        for k, f in enumerate(fg.FINGERS):
            if only is not None and f != only:
                continue
            for i, T in enumerate(self.T[f]):
                out += (w * self.S[idx, 1 + 3 * k + i])[:, None] * (P @ T[:3, :3].T + T[:3, 3] - P)
        return out


def handle_gap(P: np.ndarray, M: np.ndarray, radius: float = GRIP_RADIUS) -> np.ndarray:
    """Distance from the seated handle's surface (negative inside): butt cap and grip as one segment of cylinders."""
    rel = P - M[:3, 3]
    y = rel @ M[:3, 1]
    yc = np.clip(y, 0.0, 0.2)
    d = np.linalg.norm(rel - np.outer(yc, M[:3, 1]), axis=1)
    return d - np.where(y < fg.BUTT_LENGTH, fg.BUTT_RADIUS + radius - fg.HANDLE_RADIUS, radius)


def _palm_floor(h: dict):
    """f(points) -> how far each point sits below the palm skin (positive = inside the palm), over the palm grid."""
    palm = h["palm"]
    F = np.stack([palm["long_axis"], palm["thumb_side"], palm["normal"]], 1)
    S = (h["P"][h["palm_proper"] & (h["shares"][:, 0] > 0.5)] - palm["wrist"]) @ F
    lo = (S[:, 0].min(), S[:, 1].min())
    top, _, _ = _skin_grid(S, lo, (S[:, 0].max() + 1e-6, S[:, 1].max() + 1e-6))

    def below(p: np.ndarray) -> np.ndarray:
        q = (p - palm["wrist"]) @ F
        i = np.floor((q[:, 0] - lo[0]) / CELL_M).astype(int)
        j = np.floor((q[:, 1] - lo[1]) / CELL_M).astype(int)
        ok = (i >= 0) & (i < top.shape[0]) & (j >= 0) & (j < top.shape[1])
        out = np.zeros(len(p))
        z = np.full(len(p), np.nan)
        z[ok] = top[i[ok], j[ok]]
        has = np.isfinite(z)
        out[has] = z[has] + PALM_CLEAR_M - q[has, 2]
        return out

    return below


def solve_grip(h: dict, M: np.ndarray) -> dict:
    """Curl each finger (index, middle, ring, pinky, then the thumb) onto the seated handle M: penetration of the
    handle (with overgrip) is penalised hard, every phalanx should touch it (the thumb's last two), fingertips stay
    out of the palm, small angles preferred. Returns per finger the angles (deg), energy and world rotations."""
    hand = _Hand(h, M)
    below = _palm_floor(h)
    log_ = {}
    for k, f in ((1, "index"), (2, "middle"), (3, "ring"), (4, "pinky"), (0, "thumb")):
        own_w = hand.S[:, 1 + 3 * k:4 + 3 * k]
        mov = np.flatnonzero(own_w.sum(1) > 0.2)
        base = hand.posed(mov) - hand.posed(mov, only=f, base=hand.P[mov]) + hand.P[mov]   # the other fingers
        own = [own_w[mov, i] > 0.5 for i in range(3)]
        touch = own[1:] if f == "thumb" else own
        distal = own_w[mov, 1:].sum(1) > 0.5
        # skin already inside the handle at rest near the knuckles (the web between thumb and index, where the handle
        # leaves the fist) is no reason to bend: only more of it counts; the last two bones must get out
        before = np.minimum(handle_gap(hand.P[mov], M), 0) ** 2 * ~distal
        # the four fingers' last two bones stay out of the palm (the thumb lies over the thenar at rest)
        floor = None if f == "thumb" else distal
        floor_rest = None if floor is None else np.maximum(below(hand.P[mov][floor]), 0)

        def energy(th, f=f, mov=mov, base=base, touch=touch, floor=floor, floor_rest=floor_rest,
                   before=before) -> float:
            hand.set(f, hand.rotations(f, th))
            P = hand.posed(mov, only=f, base=base)
            d = handle_gap(P, M)
            e = 50.0 * np.sum(np.maximum(np.minimum(d, 0) ** 2 - before, 0)) / 1e-6
            for o in touch:
                if o.any():
                    m = -0.001 * np.log(np.sum(np.exp(-np.maximum(d[o], 0) / 0.001)))   # soft minimum gap
                    e += (max(m, 0) / 0.001) ** 2
            if floor is not None:
                e += 20.0 * np.sum(np.maximum(below(P[floor]), floor_rest) ** 2 - floor_rest ** 2) / 1e-6
            return e + 0.5 * float(np.sum(np.asarray(th) ** 2))

        if f == "thumb":
            bounds = [tuple(math.radians(v) for v in b) for b in THUMB_BOUNDS_DEG]
            starts = THUMB_STARTS
        else:
            bounds = [(0.0, math.radians(v)) for v in MAX_FLEX_DEG]
            starts = ([0.6, 0.8, 0.4], [b[1] * 0.5 for b in bounds], [b[1] * 0.8 for b in bounds])
        best = None
        for start in starts:
            r = minimize(energy, start, method="Powell", bounds=bounds,
                         options={"xtol": 1e-3, "ftol": 1e-4, "maxfev": 3000})
            if best is None or r.fun < best.fun:
                best = r
        hand.set(f, hand.rotations(f, best.x))
        log_[f] = {"angles_deg": [round(math.degrees(v), 1) for v in best.x], "energy": round(float(best.fun), 2)}
    return {"fingers": log_, "R": {f: hand.R[f] for f in fg.FINGERS}, "posed": hand.posed()}


def relaxed_pose(h: dict) -> dict:
    """A mild curl (RELAXED_DEG about the same axes as the grip): the hand off the racket."""
    hand = _Hand(h)
    out = {}
    for f in fg.FINGERS:
        deg = RELAXED_DEG["thumb" if f == "thumb" else "finger"]
        out[f] = hand.rotations(f, [math.radians(v) for v in deg])
        hand.set(f, out[f])
    return {"R": out, "posed": hand.posed()}


# -------------------------------------------------------------------------------------------------- checks
def check_hand(h: dict) -> list[str]:
    """Reasons the analysed hand is not believable (empty: ok)."""
    problems = []
    HL = h["HL"]
    for f, v in h["fingers"].items():
        J = v["joints"]
        L = np.linalg.norm(np.diff(J, axis=0), axis=1)
        rr = RATIOS[f]
        expect = np.array([rr[2], rr[1], rr[0]]) * HL
        ratio = L / expect
        if not (SEGMENT_RANGE[0] <= ratio.min() and ratio.max() <= SEGMENT_RANGE[1]):
            problems.append(f"{f} segments {np.round(L * 1000).astype(int).tolist()} mm vs standard "
                            f"{np.round(expect * 1000).astype(int).tolist()}")
        if f != "thumb" and v["free_m"] < FREE_MIN * L.sum():
            problems.append(f"{f} webbed: free for {v['free_m'] * 100:.1f} of {L.sum() * 100:.1f} cm")
    palm = h["palm"]
    t = np.asarray(palm["thumb_side"])
    knuckle_w = [float((h["joints"][f][0] - palm["wrist"]) @ t) for f in fg.FINGERS[1:]]
    if np.min(-np.diff(knuckle_w)) < MIN_KNUCKLE_GAP_M:
        problems.append(f"knuckles out of order across the hand: {np.round(np.array(knuckle_w) * 100, 1).tolist()} cm")
    inner = np.concatenate([h["joints"][f][:3] for f in fg.FINGERS])
    wn = np.abs(winding_number(h["P"], h["tri"], inner))
    if (wn < 0.5).any():
        out = [f"{f}{i + 1}" for k, f in enumerate(fg.FINGERS) for i in range(3) if wn[3 * k + i] < 0.5]
        problems.append(f"joints outside the mesh: {out}")
    return problems


def check_mirror(right: dict, left: dict) -> tuple[list[str], float]:
    """Left and right hands should agree in their own palm frames (the meshes are near mirror images)."""
    problems, worst = [], 0.0
    for f in fg.FINGERS:
        JR = fg.palm_coords(right["palm"], right["joints"][f])
        JL = fg.palm_coords(left["palm"], left["joints"][f])
        worst = max(worst, float(np.linalg.norm(JR - JL, axis=1).max()))
        LR, LL = (np.linalg.norm(np.diff(J, axis=0), axis=1) for J in (right["joints"][f], left["joints"][f]))
        if (np.abs(LR - LL) / LR).max() > MIRROR_SEGMENT:
            problems.append(f"{f} segments differ left/right: {np.round(LR * 1000).astype(int).tolist()} vs "
                            f"{np.round(LL * 1000).astype(int).tolist()} mm")
    if worst > MIRROR_MAX_M:
        problems.append(f"a joint differs {worst * 100:.1f} cm between the hands")
    return problems, worst


# --------------------------------------------------------------------------------------------- hand_rig
def _bones(h: dict, grip: dict, relaxed: dict) -> list[dict]:
    n, t = np.asarray(h["palm"]["normal"]), np.asarray(h["palm"]["thumb_side"])
    names = fg.bone_names(h["hand_bone"])
    out = []
    for k, f in enumerate(fg.FINGERS):
        J = h["joints"][f]
        for i in range(3):
            y = fg.unit(J[i + 1] - J[i])
            z = n if abs(y @ n) < 0.95 else t
            out.append({"name": names[3 * k + i], "parent": h["hand_bone"] if i == 0 else names[3 * k + i - 1],
                        "finger": f, "head": J[i].round(6).tolist(), "tail": J[i + 1].round(6).tolist(),
                        "z_axis": z.round(6).tolist(),
                        "grip_world": rt.matrix_to_quat(grip["R"][f][i]).round(7).tolist(),
                        "relaxed_world": rt.matrix_to_quat(relaxed["R"][f][i]).round(7).tolist()})
    return out


def _vec(x) -> list:
    return np.round(np.asarray(x, np.float64), 6).tolist()


def build_hand_rig(mesh: dict, rig_label: str = "", weld_m: float = 0.0) -> tuple[dict, dict]:
    """(hand_rig.json content, hand_rig.npz arrays) from an exported hand mesh; never raises on a bad hand."""
    t0 = time.time()
    data = {"version": fg.HAND_RIG_VERSION, "ok": False, "reason": None, "rig": rig_label, "weld_m": weld_m,
            "vertices": int(mesh["vertices"]), "frame": "rig rest world after import: metres, Z-up",
            "blender": str(mesh["blender"]), "sides": {}}
    arrays, hands = {}, {}
    names = [str(b) for b in mesh["bones"]]
    parents = [int(p) for p in mesh["parents"]]
    rig = rt.Rig(names, parents, np.asarray(mesh["rest_world"], np.float64), np.eye(4))
    mapping = json.loads(str(mesh["mapping"]))
    for side in fg.SIDES:
        res = {"ok": False, "reason": None}
        data["sides"][side] = res
        try:
            h = analyse_hand(mesh, side)
            res["hand_bone"] = h["hand_bone"]
            res["hand_vertices"] = int(len(h["P"]))
            res["tips"] = h["tip_count"]
            problems = check_hand(h)
            seat = fg.seat_from_palm(h["palm"], h["skin"])
            M = fg.seat_matrix(seat)
            grip, relaxed = solve_grip(h, M), relaxed_pose(h)
            inside = {"open": int(fg.inside_handle(h["P"], M).sum()),
                      "grip": int(fg.inside_handle(grip["posed"], M).sum())}
            if side == "Right":
                inside["fixed"] = int(fg.inside_handle(h["P"], rt.grip_matrix(rig, mapping)).sum())
            if inside["grip"] > GRIP_INSIDE_MAX * len(h["P"]):
                problems.append(f"grip leaves {inside['grip']} vertices inside the handle")
            gap = handle_gap(grip["posed"], M, fg.HANDLE_RADIUS)
            palm = {k: (_vec(v) if isinstance(v, np.ndarray) else v) for k, v in h["palm"].items()}
            res.update({
                "ok": not problems, "reason": "; ".join(problems) or None,
                "palm": palm, "hand_length_m": round(h["HL"] / HL_SCALE, 4),
                "fingers": {f: {"joints": _vec(v["joints"]), "free_m": round(v["free_m"], 4),
                                "radius_m": round(v["radius_m"], 4)} for f, v in h["fingers"].items()},
                "bones": _bones(h, grip, relaxed),
                "seat": {k: (_vec(v) if isinstance(v, np.ndarray) else v) for k, v in seat.items()},
                "grip_solve": grip["fingers"],
                "metrics": {"inside_handle": inside, "contact_2mm": int((np.abs(gap) < 0.002).sum()),
                            "palm_gap_mm": round(float(handle_gap(h["P"][h["palm_proper"]], M,
                                                                  fg.HANDLE_RADIUS).min()) * 1000, 1)},
            })
            hands[side] = h
            arrays[f"{side}_co"] = h["P"].astype(np.float32)
            arrays[f"{side}_shares"] = h["shares"].astype(np.float32)
            arrays[f"{side}_palm"] = h["skin"].astype(np.float32)
        except Exception as e:  # noqa: BLE001 - any failure is a reason to keep the fixed grip, never a crash
            res["reason"] = str(e) if isinstance(e, HandRigError) else f"analysis failed: {type(e).__name__}: {e}"
    reasons = [f"{s}: {r['reason']}" for s, r in data["sides"].items() if not r["ok"]]
    if len(hands) == 2:
        problems, worst = check_mirror(hands["Right"], hands["Left"])
        data["mirror_max_m"] = round(worst, 4)
        reasons += [f"mirror: {p}" for p in problems]
    data["ok"] = not reasons
    data["reason"] = "; ".join(reasons) or None
    data["seconds"] = round(time.time() - t0, 1)
    return data, (arrays if data["ok"] else {})


def export_mesh(rig: Path, weld_m: float, out: Path, blender: str | None = None) -> Path:
    """The rig's hand region as retarget.build sees it (blender/fingers.py export) -> out (.npz). Blender: `blender`,
    else $R3D_BLENDER (as retarget on the GPU host), else the local install."""
    binary = blender or os.environ.get("R3D_BLENDER") or _blender_local()
    cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1", "--python", str(SCRIPTS_DIR / "fingers.py"),
           "--", "export", str(rig), str(out), repr(float(weld_m))]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode or not out.exists():
        raise RuntimeError(f"blender fingers.py export failed:\n{res.stdout[-3000:]}\n{res.stderr[-2000:]}")
    return out


def write_hand_rig(out: Path, data: dict, arrays: dict) -> Path:
    """hand_rig.json, and hand_rig.npz when the rig is ok (a stale .npz of an earlier run goes)."""
    npz = out.with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    if arrays:
        np.savez_compressed(npz, **arrays)
    else:
        npz.unlink(missing_ok=True)
    return write_json(out, data)


def make(rig: Path, out: Path, weld_m: float, blender: str | None = None, label: str | None = None) -> dict:
    """rigged.glb -> out (hand_rig.json) + its .npz. A rig Blender cannot export hands from (no hand bones) also
    gets a hand_rig.json, "ok": false."""
    label = label if label is not None else _label(rig)
    with tempfile.TemporaryDirectory() as tmp:
        mesh_npz = Path(tmp) / "hand_mesh.npz"
        try:
            with step(f"hand mesh export (Blender, weld {weld_m:g} m)"):
                export_mesh(rig, weld_m, mesh_npz, blender)
        except RuntimeError as e:
            data = {"version": fg.HAND_RIG_VERSION, "ok": False, "rig": label, "weld_m": weld_m,
                    "reason": "hand mesh export failed: " + str(e).splitlines()[-1][:300], "sides": {}}
            write_hand_rig(out, data, {})
            log(f"[yellow]hand rig NOT ok: {data['reason']}[/] -> {out}")
            return data
        with step("finger joints, weights, seat and grip"):
            data, arrays = build_hand_rig(dict(np.load(mesh_npz)), label, weld_m)
    write_hand_rig(out, data, arrays)
    for side, r in data["sides"].items():
        if r["ok"]:
            m = r["metrics"]
            log(f"{side} hand: {r['hand_vertices']} vertices, seat {r['seat']['angle_deg']} deg, butt "
                f"{r['seat']['butt_past_wrist_m'] * 100:.1f} cm past the wrist, inside the handle {m['inside_handle']}")
            if r["seat"].get("clamped"):
                log(f"[yellow]{side} hand: the seat ran into its limits: {', '.join(r['seat']['clamped'])} (tilt "
                    f"{r['seat']['tilt_deg']} deg)[/]")
    log(f"hand rig {'ok' if data['ok'] else '[yellow]NOT ok[/]: ' + str(data['reason'])} -> {out}")
    return data


def _label(p: Path) -> str:
    p = Path(p).resolve()
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else p.name


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, help="config yaml: its run's rigged.glb and retarget.weld_m")
    ap.add_argument("--rig", type=Path, help="rigged.glb (instead of --config)")
    ap.add_argument("--weld", type=float, default=0.0, help="weld_m with --rig")
    ap.add_argument("--out", type=Path, help="hand_rig.json (default: next to the rig)")
    ap.add_argument("--blender", help="Blender binary (default: $R3D_BLENDER, else the local install)")
    args = ap.parse_args()
    if args.rig:
        make(args.rig, args.out or args.rig.with_name("hand_rig.json"), args.weld, args.blender)
    else:
        cfg = Config.load(args.config)
        make(cfg.run.rig, args.out or cfg.run.hand_rig, cfg.retarget.weld_m, args.blender,
             label=str(cfg.run.rig.relative_to(cfg.run.root)))
