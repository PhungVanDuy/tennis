"""Stage `camera`: calibrate the broadcast camera from the painted court lines.

GVHMR otherwise assumes one guessed focal length, but a broadcast camera is a long zoom lens on a pan/tilt
head. The court is a known planar model, so every frame that shows enough of it fixes the focal length and
the full camera pose (square pixels, principal point at the image centre, no skew, no lens distortion):

  1. line response  white top-hat of the luminance, weighted by whiteness (paint is unsaturated);
  2. initialise     Hough lines → near-horizontal / steep families → every ordered 2 + 2 image lines ×
                    2 + 2 model lines gives a closed-form homography → decompose it into (f, R, t) → keep
                    cameras that are physically plausible (above the court, at the near end, sane field of
                    view) → score against the distance transform of the line mask → snap the best to every
                    detected line they overlay (a 2 + 2 hypothesis is exact only near its own four lines);
  3. refine         sample the projected model lines, find the paint ridge along each normal (sub-pixel) and
                    minimise point-to-line distances over the 7 camera parameters (tight gate, repeated);
  4. verify         a frame holds when most visible model samples sit on paint, no visible court line is
                    missing its paint, and the mean distance is small;
  5. track          anchor on the best of a few initialised frames, then track forwards and backwards (each
                    frame starts from its neighbours), re-initialising every few lost frames;
  6. clip bundle    a main broadcast camera pans, tilts and zooms on a fixed head: one optical centre per clip
                    with per-frame focal length and rotation removes the focal / distance ambiguity of single
                    frames. Kept only when it does not degrade the fit (low cameras behind the baseline ride
                    on rails).

It runs on preview.mp4 (720p, frames 1:1 with clip.mp4; it lacks the clip's last frame, which is held) and
writes clip.mp4 pixels. A calibration is written only when ≥ 90 % of the frames hold with a low line
reprojection error — a wrong camera is worse than GVHMR's default.

clips/<id>/camera.npz   world = court frame: metres, origin at the court centre, X across the court (right in
                        the image), Y along it away from the camera, Z up
  K            (L,3,3)  intrinsics in clip.mp4 pixels (square pixels, principal point at the image centre
                        ((w − 1)/2, (h − 1)/2) in OpenCV pixel-centre coordinates)
  R_w2c, t_w2c (L,3,3), (L,3)   x_cam = R_w2c · X_world + t_w2c
  H_img2court  (L,3,3)  clip.mp4 pixels → court plane (X, Y) in metres
  ok           (L,)     frame calibrated on its own lines (the others are interpolated from their neighbours)
  reproj_px    (L,)     mean distance of the detected line ridges to the projected court lines (clip pixels,
                        NaN where the frame has no fit of its own)
  method       str
  centre       (3,)     the optical centre when the camera only pans / tilts / zooms (else NaN)
clips/<id>/camera.jpg   the court model drawn over the start / contact / end frames, for review
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ClipLayout, Config
from .util import log, read_json, step

CALIBRATED_VIEWS = ("main_broadcast", "behind_low")
METHOD = "court-lines/v2: ridge Hough init, point-to-line refine, fixed-centre clip bundle"
WORK_HEIGHT = 720             # px; preview.mp4 resolution, every pixel threshold below is at this height
MIN_OK_FRACTION = 0.90        # of the clip frames
MAX_CLIP_REPROJ_PX = 1.5      # mean over the ok frames, clip.mp4 pixels
MAX_FRAME_REPROJ_PX = 2.5     # clip.mp4 pixels
MIN_MATCH_FRACTION = 0.65     # of the visible, unoccluded model samples that found their paint line
DEAD_LINE_FRACTION = 0.2      # a visible court line (≥ 10 samples) with fewer matches is not on its paint
MAX_DEAD_LINES = 0
HFOV_RANGE_DEG = (2.0, 75.0)

TOPHAT_PX = 15                # wider than any painted line, narrower than logos and players
SAT_MAX, SAT_RAMP = 140.0, 80.0
MASK_MIN = 30.0               # response threshold for the Hough mask
PEAK_MIN = 12.0               # minimum ridge response along a normal
SAMPLE_PX = 6.0               # spacing of the refinement samples along the projected lines
MAX_HOR_LINES, MAX_VER_LINES = 24, 8    # hypothesis budget: stadium, ads and net crowd the horizontal family
BANDS_INIT = (4.0, 4.0, 4.0, 3.0, 3.0, 2.5)   # tight gate, repeated: the aligned region grows
BANDS_TRACK = (16.0, 8.0, 4.0, 2.5)
REINIT_EVERY = 4              # lost frames between full re-initialisations
MAX_ZOOM_JUMP = 0.25          # |log f| change allowed against the nearest tracked frame
MAX_MOVE_JUMP_M = 3.0         # optical centre change allowed against the nearest tracked frame

# ITF court, metres. Dimensions are to the outer edges of the lines; baselines are 10 cm wide, others 5 cm.
HALF_LENGTH, HALF_DOUBLES, HALF_SINGLES, SERVICE_Y = 11.885, 5.485, 4.115, 6.40
N_HOR = 4                     # the first 4 lines run parallel to the net (near → far), then 5 along it


def court_lines() -> np.ndarray:
    """(9, 4) painted line centres (x0, y0, x1, y1): near baseline, near/far service line, far baseline,
    then left doubles, left singles, centre service line, right singles, right doubles sidelines."""
    yb, ys = HALF_LENGTH - 0.05, SERVICE_Y - 0.025
    xd, xs = HALF_DOUBLES - 0.025, HALF_SINGLES - 0.025
    return np.array([
        [-HALF_DOUBLES, -yb, HALF_DOUBLES, -yb], [-xs, -ys, xs, -ys], [-xs, ys, xs, ys],
        [-HALF_DOUBLES, yb, HALF_DOUBLES, yb],
        [-xd, -HALF_LENGTH, -xd, HALF_LENGTH], [-xs, -HALF_LENGTH, -xs, HALF_LENGTH], [0.0, -ys, 0.0, ys],
        [xs, -HALF_LENGTH, xs, HALF_LENGTH], [xd, -HALF_LENGTH, xd, HALF_LENGTH],
    ])


COURT = court_lines()
# line coordinate: y for the lines parallel to the net, x for the others
COURT_COORD = np.r_[COURT[:N_HOR, 1], COURT[N_HOR:, 0]]
COURT_HOMOG = np.cross(np.c_[COURT[:, :2], np.ones(len(COURT))], np.c_[COURT[:, 2:], np.ones(len(COURT))])


# ---------------------------------------------------------------------------------------------- camera model

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle (…,3) → rotation matrices (…,3,3)."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.asarray(rvec).reshape(-1, 3)).as_matrix()
    return R.reshape(*np.shape(rvec)[:-1], 3, 3)


def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(np.asarray(R).reshape(-1, 3, 3)).as_rotvec().reshape(*np.shape(R)[:-2], 3)


def principal_point(size: tuple[int, int]) -> tuple[float, float]:
    """The image centre in pixel-centre coordinates (OpenCV): the point a centre-aligned rescale keeps fixed,
    so preview.mp4 and clip.mp4 cameras differ by the focal length alone."""
    return (size[0] - 1) / 2, (size[1] - 1) / 2


def intrinsics(f: float, size: tuple[int, int]) -> np.ndarray:
    cx, cy = principal_point(size)
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def homography(p: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Court plane (X, Y, 1) → image for camera parameters p = (f, rvec, t)."""
    R = rodrigues(p[1:4])
    return intrinsics(p[0], size) @ np.column_stack([R[:, 0], R[:, 1], p[4:7]])


def centre(p: np.ndarray) -> np.ndarray:
    """Optical centre in court coordinates."""
    return -rodrigues(p[1:4]).T @ p[4:7]


def decompose(H: np.ndarray, size: tuple[int, int], cos_max: float = 0.2,
              ratio_max: float = 1.25) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Court → image homographies (N,3,3) → f (N,), R (N,3,3), t (N,3), plausible (N,).

    With the principal point known, the two columns of K⁻¹H must be orthogonal and of equal length: two
    equations for 1/f², solved in least squares; their disagreement rejects inconsistent correspondences.
    """
    H = np.asarray(H, np.float64).reshape(-1, 3, 3)
    cx, cy = principal_point(size)
    A = H.copy()
    A[:, 0] -= cx * H[:, 2]
    A[:, 1] -= cy * H[:, 2]
    a1, a2, a3 = A[:, :, 0], A[:, :, 1], A[:, :, 2]
    p1 = a1[:, 0] * a2[:, 0] + a1[:, 1] * a2[:, 1]
    q1 = a1[:, 2] * a2[:, 2]
    p2 = a1[:, 0] ** 2 + a1[:, 1] ** 2 - a2[:, 0] ** 2 - a2[:, 1] ** 2
    q2 = a1[:, 2] ** 2 - a2[:, 2] ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        w = -(p1 * q1 + p2 * q2) / (p1 ** 2 + p2 ** 2)
        f = 1.0 / np.sqrt(np.where(w > 0, w, np.nan))
        s = np.stack([1 / f, 1 / f, np.ones_like(f)], 1)
        b1, b2, b3 = a1 * s, a2 * s, a3 * s
        n1, n2 = np.linalg.norm(b1, axis=1), np.linalg.norm(b2, axis=1)
        lam = (n1 + n2) / 2 * np.sign(b3[:, 2])           # the court centre lies in front of the camera
        r1, r2, t = b1 / lam[:, None], b2 / lam[:, None], b3 / lam[:, None]
        cos12 = np.abs((b1 * b2).sum(1)) / (n1 * n2)
        ratio = n1 / n2
    # Gram-Schmidt is enough here: hypotheses are refined afterwards
    r1, r2, t = np.nan_to_num(r1), np.nan_to_num(r2), np.nan_to_num(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        e1 = r1 / np.linalg.norm(r1, axis=1, keepdims=True)
        e2 = r2 - (e1 * r2).sum(1, keepdims=True) * e1
        e2 /= np.linalg.norm(e2, axis=1, keepdims=True)
    R = np.nan_to_num(np.stack([e1, e2, np.cross(e1, e2)], 2))
    C = -(R * t[:, :, None]).sum(1)
    ok = (np.isfinite(f) & (cos12 < cos_max) & (ratio > 1 / ratio_max) & (ratio < ratio_max)
          & (np.linalg.det(R) > 0) & plausible(f, C, size))
    return f, R, t, ok


def plausible(f: np.ndarray, C: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A broadcast camera: above the court, at the near end, not absurdly far, a sane field of view."""
    C = np.atleast_2d(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        hfov = 2 * np.degrees(np.arctan(size[0] / 2 / np.asarray(f)))
    return ((C[:, 2] > 1.0) & (C[:, 2] < 80) & (C[:, 1] < 0) & (np.linalg.norm(C, axis=1) < 200)
            & (hfov > HFOV_RANGE_DEG[0]) & (hfov < HFOV_RANGE_DEG[1]))


def params_from(f: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.r_[f, rotvec(R), t]


def project_lines(p: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Court lines as image lines (9,3), normalised so (a, b) is the unit normal."""
    Hinv = np.linalg.inv(homography(p, size))
    lines = COURT_HOMOG @ Hinv
    return lines / np.linalg.norm(lines[:, :2], axis=1, keepdims=True)


def project_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Court points (P,2) through homographies (N,3,3) → image (N,P,2); points behind the camera → nan."""
    X, Y = pts[None, :, 0], pts[None, :, 1]
    u = H[:, 0, 0, None] * X + H[:, 0, 1, None] * Y + H[:, 0, 2, None]
    v = H[:, 1, 0, None] * X + H[:, 1, 1, None] * Y + H[:, 1, 2, None]
    z = H[:, 2, 0, None] * X + H[:, 2, 1, None] * Y + H[:, 2, 2, None]
    front = z > 1e-9
    z = np.where(front, z, 1.0)
    return np.where(front[..., None], np.stack([u / z, v / z], -1), np.nan)


# --------------------------------------------------------------------------------------------- line evidence

def line_response(bgr: np.ndarray) -> np.ndarray:
    """Ridge strength of white paint: top-hat of the luminance (thin bright structures) × whiteness."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_PX, TOPHAT_PX))
    top = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    white = np.clip((SAT_MAX - sat) / SAT_RAMP, 0, 1)
    return cv2.GaussianBlur(top * white, (0, 0), 0.8)


def detect_lines(mask: np.ndarray, max_lines: int = 16) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Long straight lines in a binary mask: homogeneous (M,3) with unit normals, their support (px) and
    their segment end points."""
    import cv2

    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=30, minLineLength=40, maxLineGap=6)
    if segs is None:
        return np.zeros((0, 3)), np.zeros(0), []
    segs = segs.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    clusters: list[dict] = []
    for i in np.argsort(-lengths):
        a, b = segs[i, :2], segs[i, 2:]
        for c in clusters:
            if abs(c["line"] @ np.r_[a, 1]) < 4 and abs(c["line"] @ np.r_[b, 1]) < 4:
                c["pts"] += [a, b]
                c["w"] += [lengths[i]] * 2
                c["len"] += lengths[i]
                c["line"] = _fit_line(np.array(c["pts"]), np.array(c["w"]))
                break
        else:
            clusters.append({"pts": [a, b], "w": [lengths[i]] * 2, "len": lengths[i],
                             "line": _fit_line(np.array([a, b]), np.ones(2))})
    clusters.sort(key=lambda c: -c["len"])
    # a second pass merges clusters that are the same line over their extent (thick or slightly curved
    # paint, the sagging net tape) so duplicates do not crowd real lines out of the hypothesis budget
    merged: list[dict] = []
    for c in clusters:
        pts = np.array(c["pts"])
        for m in merged:
            if np.abs(np.c_[pts, np.ones(len(pts))] @ m["line"]).max() < 6:
                m["len"] += c["len"]
                m["pts"] += c["pts"]
                break
        else:
            merged.append(c)
    merged = [c for c in merged if c["len"] >= 60][:max_lines]
    if not merged:
        return np.zeros((0, 3)), np.zeros(0), []
    return (np.array([c["line"] for c in merged]), np.array([c["len"] for c in merged]),
            [np.array(c["pts"]) for c in merged])


def _fit_line(pts: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted total-least-squares line through points: (a, b, c), a² + b² = 1."""
    m = (pts * w[:, None]).sum(0) / w.sum()
    d = pts - m
    cov = (d * w[:, None]).T @ d
    n = np.linalg.eigh(cov)[1][:, 0]
    return np.r_[n, -n @ m]


# ------------------------------------------------------------------------------------------- initialisation

def square_to_quad(q: np.ndarray) -> np.ndarray:
    """Homographies (N,3,3) mapping the unit square (0,0), (1,0), (1,1), (0,1) onto quads q (N,4,2)."""
    x0, y0, x1, y1, x2, y2, x3, y3 = (q[:, k // 2, k % 2] for k in range(8))
    sx, sy = x0 - x1 + x2 - x3, y0 - y1 + y2 - y3
    dx1, dx2, dy1, dy2 = x1 - x2, x3 - x2, y1 - y2, y3 - y2
    with np.errstate(divide="ignore", invalid="ignore"):
        den = dx1 * dy2 - dx2 * dy1
        g, h = (sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den
    return np.stack([np.stack([x1 - x0 + g * x1, x3 - x0 + h * x3, x0], -1),
                     np.stack([y1 - y0 + g * y1, y3 - y0 + h * y3, y0], -1),
                     np.stack([g, h, np.ones_like(g)], -1)], 1)


def _model_rects() -> np.ndarray:
    """(60,3,3): unit square → court rectangle for every (near < far) × (left < right) choice of lines."""
    ys, xs = COURT_COORD[:N_HOR], COURT_COORD[N_HOR:]
    out = []
    for i, j in itertools.combinations(range(N_HOR), 2):
        for k, m in itertools.combinations(range(len(xs)), 2):
            out.append([[xs[m] - xs[k], 0, xs[k]], [0, ys[j] - ys[i], ys[i]], [0, 0, 1]])
    return np.linalg.inv(np.array(out))


MODEL_RECTS_INV = _model_rects()


def _score_samples(spacing: float = 0.3) -> np.ndarray:
    pts = []
    for x0, y0, x1, y1 in COURT:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / spacing))
        s = np.linspace(0, 1, n)[:, None]
        pts.append(np.c_[x0 + s * (x1 - x0), y0 + s * (y1 - y0)])
    return np.concatenate(pts)


SCORE_SAMPLES = _score_samples()


def initialise(resp: np.ndarray, exclude: np.ndarray | None = None, keep: int = 12) -> list[np.ndarray]:
    """Camera hypotheses (best first) from the lines visible in one frame.

    Lines parallel to the net are near-horizontal in a broadcast view, the others steep (both sets overlap
    for safety). The court is below the horizon, so the upper of two image lines is the farther court line
    and, along the bottom row, left-to-right order is court X order: only ordered model choices are tried.
    """
    import cv2

    h, w = resp.shape
    size = (w, h)
    mask = (resp > MASK_MIN).astype(np.uint8) * 255
    lines, _, support = detect_lines(mask, 40)
    if len(lines) < 4:
        return []
    tilt = np.degrees(np.arctan2(np.abs(lines[:, 0]), np.abs(lines[:, 1])))   # 0 = horizontal image line
    with np.errstate(divide="ignore", invalid="ignore"):
        y_mid = -(lines[:, 2] + lines[:, 0] * w / 2) / lines[:, 1]
        x_bot = -(lines[:, 2] + lines[:, 1] * h) / lines[:, 0]
    hor = [i for i in range(len(lines)) if tilt[i] < 20][:MAX_HOR_LINES]
    ver = [i for i in range(len(lines)) if tilt[i] > 8][:MAX_VER_LINES]
    combos = []
    for a, b in itertools.combinations(hor, 2):
        far, near = (a, b) if y_mid[a] < y_mid[b] else (b, a)
        for c, d in itertools.combinations(ver, 2):
            left, right = (c, d) if x_bot[c] < x_bot[d] else (d, c)
            if len({a, b, c, d}) == 4:
                combos.append((near, far, left, right))
    if not combos:
        return []
    L = lines[np.array(combos)]
    corners = np.stack([np.cross(L[:, 0], L[:, 2]), np.cross(L[:, 0], L[:, 3]),
                        np.cross(L[:, 1], L[:, 3]), np.cross(L[:, 1], L[:, 2])], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        quads = corners[..., :2] / corners[..., 2:3]
    with np.errstate(invalid="ignore"):
        finite = np.all(np.isfinite(quads), axis=(1, 2)) & np.all(np.abs(quads) < 20 * w, axis=(1, 2))
        # perspective: the near edge of a court rectangle is longer than its far edge
        near_len = np.linalg.norm(quads[:, 1] - quads[:, 0], axis=1)
        finite &= near_len > np.linalg.norm(quads[:, 2] - quads[:, 3], axis=1)
    if not finite.any():
        return []
    S = square_to_quad(quads[finite])
    T = np.einsum("aij,bjk->abik", S, MODEL_RECTS_INV).reshape(-1, 3, 3)
    T = T[np.all(np.isfinite(T), axis=(1, 2))]
    f, R, t, ok = decompose(T, size, cos_max=0.3, ratio_max=1.4)
    if not ok.any():
        return []
    T, f, R, t = T[ok], f[ok], R[ok], t[ok]

    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
    # coarse score on every 3rd sample, then the full score for the best few thousand
    coarse = _score(T, SCORE_SAMPLES[::3], dt, exclude, min_vis=15)
    top = np.argsort(-coarse)[:8000]
    T, f, R, t = T[top], f[top], R[top], t[top]
    scores = _score(T, SCORE_SAMPLES, dt, exclude, min_vis=40)
    box = np.array([[-HALF_DOUBLES, -HALF_LENGTH], [HALF_DOUBLES, -HALF_LENGTH],
                    [-HALF_DOUBLES, HALF_LENGTH], [HALF_DOUBLES, HALF_LENGTH]])
    seen: list[np.ndarray] = []

    def novel(p: np.ndarray) -> bool:
        c = project_points(homography(p, size)[None], box)[0]
        if any(np.nanmax(np.abs(c - s)) < 10 for s in seen):
            return False
        seen.append(c)
        return True

    # both the raw hypotheses and their line-snapped versions: snapping fixes extrapolated far courts but
    # can also pull a right hypothesis onto a wrong line, the refined and verified fit decides
    raw, snapped = [], []
    for i in np.argsort(-scores)[:4 * keep]:
        if scores[i] <= 0:
            break
        p = params_from(f[i], R[i], t[i])
        if len(raw) < keep and novel(p):
            raw.append(p)
        p = snap_to_lines(p, support, size)
        if p is not None and len(snapped) < keep and novel(p):
            snapped.append(p)
    return [p for pair in itertools.zip_longest(raw, snapped) for p in pair if p is not None]


def snap_to_lines(p: np.ndarray, support: list[np.ndarray], size: tuple[int, int],
                  tols: tuple[float, ...] = (30.0, 15.0, 8.0)) -> np.ndarray | None:
    """Line-level consensus: attach every detected line to the projected court line it lies along, then
    fit the camera to all attached lines. A 2 + 2 hypothesis is exact only near its four lines; whole lines
    (the long sidelines above all) fix the extrapolated far court before the pixel-level refinement."""
    from scipy.optimize import least_squares

    for tol in tols:
        if not plausible(p[0], centre(p), size)[0]:
            return None
        proj = project_lines(p, size)
        q, j = [], []
        for pts in support:
            d = np.median(np.abs(pts @ proj[:, :2].T + proj[:, 2]), axis=0)   # to each projected court line
            k, k2 = np.argsort(d)[:2]
            if d[k] < tol and d[k2] > 2 * d[k]:
                q.append(pts)
                j.append(np.full(len(pts), k))
        if len(q) < 4 or len({int(x[0]) for x in j}) < 4:
            return None
        q, j = np.concatenate(q), np.concatenate(j)
        try:
            p = least_squares(residuals, p, args=(q, j, size), loss="soft_l1", f_scale=2.0, x_scale="jac",
                              max_nfev=60).x
        except (ValueError, np.linalg.LinAlgError):
            return None
    return p if np.all(np.isfinite(p)) and plausible(p[0], centre(p), size)[0] else None


def _score(T: np.ndarray, pts: np.ndarray, dt: np.ndarray, exclude: np.ndarray | None,
           min_vis: int) -> np.ndarray:
    """Model samples on line pixels minus half the visible ones that are not (distance transform)."""
    h, w = dt.shape
    out = np.empty(len(T))
    for s in range(0, len(T), 8192):
        uv = project_points(T[s:s + 8192], pts)
        vis = _visible(uv, (w, h), exclude)
        ui = np.clip(np.nan_to_num(uv[..., 0]), 0, w - 1).astype(np.int32)
        vi = np.clip(np.nan_to_num(uv[..., 1]), 0, h - 1).astype(np.int32)
        d = dt[vi, ui]
        hit = vis & (d <= 2.5)
        miss = vis & (d > 5)
        out[s:s + 8192] = hit.sum(1) - 0.5 * miss.sum(1) - 1000 * (vis.sum(1) < min_vis)
    return out


def _visible(uv: np.ndarray, size: tuple[int, int], exclude: np.ndarray | None,
             margin: float = 3) -> np.ndarray:
    u, v = uv[..., 0], uv[..., 1]
    with np.errstate(invalid="ignore"):
        vis = (u > margin) & (u < size[0] - margin) & (v > margin) & (v < size[1] - margin)
        if exclude is not None:
            for x0, y0, x1, y1 in np.atleast_2d(exclude):
                vis &= ~((u > x0) & (u < x1) & (v > y0) & (v < y1))
    return vis


# ----------------------------------------------------------------------------------------------- refinement

@dataclass
class Fit:
    p: np.ndarray
    n_vis: int = 0
    n_match: int = 0
    err_px: float = math.inf          # working pixels
    lines_hor: int = 0
    lines_ver: int = 0
    dead_lines: int = 0
    q: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    j: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    @property
    def frac(self) -> float:
        return self.n_match / max(self.n_vis, 1)

    @property
    def score(self) -> float:
        return self.n_match - 0.5 * (self.n_vis - self.n_match)


def samples(p: np.ndarray, size: tuple[int, int], exclude: np.ndarray | None,
            spacing: float = SAMPLE_PX) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points along the projected court lines, ~`spacing` px apart: image points (N,2), normals, line ids."""
    H = homography(p, size)
    lines = project_lines(p, size)
    pts, nrm, ids = [], [], []
    s = np.linspace(0, 1, 400)[:, None]
    for j, (x0, y0, x1, y1) in enumerate(COURT):
        uv = project_points(H[None], np.c_[x0 + s * (x1 - x0), y0 + s * (y1 - y0)])[0]
        vis = _visible(uv, size, None)
        uv = uv[vis]
        if len(uv) < 2:
            continue
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
        keep = np.r_[True, np.diff(np.floor(arc / spacing)) > 0]
        uv = uv[keep]
        if exclude is not None:
            uv = uv[_visible(uv[None], size, exclude)[0]]
        pts.append(uv)
        nrm.append(np.broadcast_to(lines[j, :2], uv.shape))
        ids.append(np.full(len(uv), j))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(ids)


def match(resp: np.ndarray, pts: np.ndarray, nrm: np.ndarray, band: float,
          step_px: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Nearest strong paint ridge along each normal within ±band: sub-pixel points (N,2), found (N,)."""
    import cv2

    if not len(pts):
        return pts, np.zeros(0, bool)
    offs = np.arange(-band, band + 1e-6, step_px)
    mx = (pts[:, :1] + offs * nrm[:, :1]).astype(np.float32)
    my = (pts[:, 1:] + offs * nrm[:, 1:]).astype(np.float32)
    prof = cv2.remap(resp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    left, mid, right = prof[:, :-2], prof[:, 1:-1], prof[:, 2:]
    strong = np.maximum(PEAK_MIN, 0.5 * prof.max(1, keepdims=True))
    peak = (mid >= left) & (mid > right) & (mid >= strong)
    dist = np.where(peak, np.abs(offs[1:-1]), np.inf)
    k = dist.argmin(1)
    found = np.isfinite(dist[np.arange(len(k)), k])
    r = np.arange(len(k))
    y0, y1, y2 = left[r, k], mid[r, k], right[r, k]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-6, den, 1), 0)
    off = offs[1:-1][k] + np.clip(sub, -0.5, 0.5) * step_px
    return pts + off[:, None] * nrm, found


def residuals(p: np.ndarray, q: np.ndarray, j: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    lines = project_lines(p, size)[j]
    return lines[:, 0] * q[:, 0] + lines[:, 1] * q[:, 1] + lines[:, 2]


def evaluate(resp: np.ndarray, p: np.ndarray, exclude: np.ndarray | None,
             band: float = BANDS_INIT[-1]) -> Fit:
    size = resp.shape[::-1]
    pts, nrm, ids = samples(p, size, exclude)
    q, found = match(resp, pts, nrm, band)
    q, j = q[found], ids[found]
    fit = Fit(p, n_vis=len(pts), n_match=len(q), q=q, j=j)
    if len(q) and plausible(p[0], centre(p), size)[0]:
        fit.err_px = float(np.abs(residuals(p, q, j, size)).mean())
        counts = np.bincount(j, minlength=len(COURT))
        visible = np.bincount(ids, minlength=len(COURT))
        fit.lines_hor = int((counts[:N_HOR] >= 5).sum())
        fit.lines_ver = int((counts[N_HOR:] >= 5).sum())
        fit.dead_lines = int(((visible >= 10) & (counts < DEAD_LINE_FRACTION * visible)).sum())
    return fit


def refine(resp: np.ndarray, p: np.ndarray, exclude: np.ndarray | None,
           bands: tuple[float, ...] = BANDS_INIT) -> Fit:
    """Iterate ridge matching and point-to-line least squares over (f, rvec, t), one pass per search band."""
    from scipy.optimize import least_squares

    size = resp.shape[::-1]
    for band in bands:
        pts, nrm, ids = samples(p, size, exclude)
        q, found = match(resp, pts, nrm, band)
        if found.sum() < 12:
            return Fit(p)
        q, j = q[found], ids[found]
        try:
            sol = least_squares(residuals, p, args=(q, j, size), loss="soft_l1", f_scale=max(1.0, band / 4),
                                x_scale="jac", max_nfev=60)
        except (ValueError, np.linalg.LinAlgError):
            return Fit(p)
        if not np.all(np.isfinite(sol.x)) or sol.x[0] <= 0:
            return Fit(p)
        p = sol.x
    return evaluate(resp, p, exclude, bands[-1])


def frame_ok(fit: Fit, scale: float) -> bool:
    return (fit.frac >= MIN_MATCH_FRACTION and fit.err_px * scale <= MAX_FRAME_REPROJ_PX
            and fit.lines_hor >= 2 and fit.lines_ver >= 2 and fit.dead_lines <= MAX_DEAD_LINES)


def calibrate_frame(resp: np.ndarray, exclude: np.ndarray | None, scale: float,
                    guesses: list[np.ndarray] = (), init: bool = True) -> Fit | None:
    """Best refined fit from tracking guesses, falling back to a full initialisation."""
    best = None
    for g in guesses:
        fit = refine(resp, g, exclude, BANDS_TRACK)
        if frame_ok(fit, scale) and (best is None or fit.score > best.score):
            best = fit
    if best is not None or not init:
        return best
    for g in initialise(resp, exclude):
        fit = refine(resp, g, exclude, BANDS_INIT)
        if frame_ok(fit, scale) and (best is None or fit.score > best.score):
            best = fit
    return best


# ---------------------------------------------------------------------------------------------- clip bundle

def _split(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return x[:3], x[3:].reshape(n, 4)


def _cam(C: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """(f, rvec) + optical centre → p = (f, rvec, t)."""
    R = rodrigues(fr[1:4])
    return np.r_[fr[0], fr[1:4], -R @ C]


def bundle(fits: dict[int, Fit], size: tuple[int, int]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Shared optical centre + per-frame (f, rvec) from the frames' ridge matches."""
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    frames = sorted(fits)
    n = len(frames)
    C0 = np.median([centre(fits[i].p) for i in frames], axis=0)
    x0 = np.r_[C0, np.concatenate([fits[i].p[:4] for i in frames])]
    rows = np.cumsum([0] + [len(fits[i].q) for i in frames])

    def fun(x):
        C, fr = _split(x, n)
        return np.concatenate([residuals(_cam(C, fr[k]), fits[i].q, fits[i].j, size)
                               for k, i in enumerate(frames)])

    sparsity = lil_matrix((rows[-1], len(x0)), dtype=int)
    for k in range(n):
        sparsity[rows[k]:rows[k + 1], :3] = 1
        sparsity[rows[k]:rows[k + 1], 3 + 4 * k: 7 + 4 * k] = 1
    sol = least_squares(fun, x0, jac_sparsity=sparsity, loss="soft_l1", f_scale=1.0, x_scale="jac",
                        max_nfev=100)
    C, fr = _split(sol.x, n)
    return C, {i: _cam(C, fr[k]) for k, i in enumerate(frames)}


def _fixed_residuals(fr: np.ndarray, C: np.ndarray, q: np.ndarray, j: np.ndarray,
                     size: tuple[int, int]) -> np.ndarray:
    return residuals(_cam(C, fr), q, j, size)


def refine_fixed_centre(resp: np.ndarray, p: np.ndarray, C: np.ndarray, exclude: np.ndarray | None,
                        bands: tuple[float, ...]) -> Fit:
    """Per-frame refinement of (f, rvec) only, about the clip's optical centre."""
    from scipy.optimize import least_squares

    size = resp.shape[::-1]
    fr = p[:4].copy()
    for band in bands:
        pts, nrm, ids = samples(_cam(C, fr), size, exclude)
        q, found = match(resp, pts, nrm, band)
        if found.sum() < 8:
            return Fit(_cam(C, fr))
        q, j = q[found], ids[found]
        try:
            sol = least_squares(_fixed_residuals, fr, args=(C, q, j, size), loss="soft_l1",
                                f_scale=max(1.0, band / 4), x_scale="jac", max_nfev=40)
        except (ValueError, np.linalg.LinAlgError):
            return Fit(_cam(C, fr))
        if not np.all(np.isfinite(sol.x)) or sol.x[0] <= 0:
            return Fit(_cam(C, fr))
        fr = sol.x
    return evaluate(resp, _cam(C, fr), exclude, bands[-1])


# ----------------------------------------------------------------------------------------------- per clip

@dataclass
class ClipCalibration:
    p: np.ndarray                 # (L,7) working-resolution parameters
    ok: np.ndarray                # (L,)
    err_px: np.ndarray            # (L,) working pixels
    frac: np.ndarray              # (L,)
    centre: np.ndarray | None     # shared optical centre (court frame) or None
    size: tuple[int, int]


def read_preview(path: Path) -> list[np.ndarray]:
    import cv2

    from .media import read_frames

    frames = read_frames(path)
    out = []
    for f in frames:
        if f.shape[0] != WORK_HEIGHT:
            f = cv2.resize(f, (round(f.shape[1] * WORK_HEIGHT / f.shape[0]), WORK_HEIGHT),
                           interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def calibrate_clip(frames: list[np.ndarray], boxes: np.ndarray | None, scale: float,
                   label: str = "", fixed_centre: bool = True) -> ClipCalibration | None:
    """Track the camera through the frames; `boxes` (L,4) are player boxes in working pixels to ignore."""
    n = len(frames)
    size = (frames[0].shape[1], frames[0].shape[0])
    resps = [line_response(f) for f in frames]

    def excl(i):
        if boxes is None or i >= len(boxes):
            return None
        x0, y0, x1, y1 = boxes[i]
        pw, ph = 0.15 * (x1 - x0), 0.08 * (y1 - y0)
        return np.array([[x0 - pw, y0 - ph, x1 + pw, y1 + ph]])

    fits: dict[int, Fit] = {}

    def guesses(i: int, d: int) -> list[np.ndarray]:
        out = []
        if i - d in fits and i - 2 * d in fits:
            out.append(2 * fits[i - d].p - fits[i - 2 * d].p)   # constant velocity
        near = [k for k in fits if abs(k - i) <= 10]
        if near:
            out.append(fits[min(near, key=lambda k: abs(k - i))].p)
        return out

    # anchor on the best of a few initialised frames (single frames can be ambiguous: a line in shadow, a
    # faint far baseline), then track forwards and backwards, re-initialising every few lost frames
    def consistent(fit: Fit, i: int) -> bool:
        """A re-initialised frame must agree with the tracked camera (no zoom or move jumps)."""
        ref = fits[min(fits, key=lambda k: abs(k - i))].p
        return (abs(math.log(fit.p[0] / ref[0])) < MAX_ZOOM_JUMP
                and np.linalg.norm(centre(fit.p) - centre(ref)) < MAX_MOVE_JUMP_M)

    probes = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    anchors = {i: fit for i in probes if (fit := calibrate_frame(resps[i], excl(i), scale)) is not None}
    if not anchors:
        return None
    a = max(anchors, key=lambda i: anchors[i].score)
    fits[a] = anchors[a]
    for d in (1, -1):
        lost = 0
        for i in range(a + d, n if d > 0 else -1, d):
            fit = calibrate_frame(resps[i], excl(i), scale, guesses(i, d),
                                  init=lost % REINIT_EVERY == REINIT_EVERY - 1)
            if fit is None and i in anchors:
                fit = anchors[i]
            if fit is None or not consistent(fit, i):
                lost += 1
            else:
                fits[i], lost = fit, 0

    C = None
    free_err = np.mean([f.err_px for f in fits.values()])
    for _ in range(2 if len(fits) >= 3 and fixed_centre else 0):
        try:
            C_new, ps = bundle(fits, size)
        except (ValueError, np.linalg.LinAlgError):
            break
        guess = _interpolate(ps, n, C_new)
        new = {i: refine_fixed_centre(resps[i], guess[i], C_new, excl(i), BANDS_TRACK[1:]) for i in range(n)}
        new = {i: f for i, f in new.items() if frame_ok(f, scale)}
        new_err = np.mean([f.err_px for f in new.values()]) if new else math.inf
        if len(new) < len(fits) or new_err > 1.25 * free_err + 0.05:
            # not a fixed pan/tilt head (a rail or handheld camera): keep the free cameras
            log(f"{label}: fixed centre rejected ({len(new)} vs {len(fits)} frames, "
                f"err {free_err * scale:.2f} → {new_err * scale:.2f}px)")
            break
        fits, C = new, C_new
    if C is not None:
        log(f"{label}: {len(fits)}/{n} frames, err {free_err * scale:.2f} → "
            f"{np.mean([f.err_px for f in fits.values()]) * scale:.2f}px with a fixed centre")
    ok = np.zeros(n, bool)
    err = np.full(n, np.nan)
    frac = np.zeros(n)
    for i, fit in fits.items():
        ok[i] = frame_ok(fit, scale)
        err[i], frac[i] = fit.err_px, fit.frac
    ps = {i: fits[i].p for i in fits if ok[i]}
    if not ps:
        return None
    return ClipCalibration(_interpolate(ps, n, C), ok, err, frac, C, size)


def _interpolate(ps: dict[int, np.ndarray], n: int, C: np.ndarray | None = None) -> np.ndarray:
    """Fill the missing frames: log-focal and rotation interpolated, held at the ends."""
    from scipy.spatial.transform import Rotation, Slerp

    known = sorted(ps)
    P = np.stack([ps[i] for i in known])
    idx = np.clip(np.arange(n), known[0], known[-1])
    out = np.empty((n, 7))
    out[:, 0] = np.exp(np.interp(idx, known, np.log(P[:, 0])))
    if len(known) > 1:
        out[:, 1:4] = Slerp(known, Rotation.from_rotvec(P[:, 1:4]))(idx).as_rotvec()
    else:
        out[:, 1:4] = P[0, 1:4]
    for k in range(3):
        out[:, 4 + k] = np.interp(idx, known, P[:, 4 + k])
    if C is not None:
        out[:, 4:7] = -np.einsum("nij,j->ni", rodrigues(out[:, 1:4]), C)
    for i in known:
        out[i] = ps[i]
    return out


# ------------------------------------------------------------------------------------------------- output

def to_clip_pixels(p: np.ndarray, work: tuple[int, int], clip: tuple[int, int]) -> dict[str, np.ndarray]:
    """Working-resolution parameters (L,7) → the camera.npz arrays in clip.mp4 pixels.

    preview.mp4 is a centre-aligned rescale of clip.mp4 (u_clip = s·u_work + (s − 1)/2); with the principal
    point at the pixel-centre image centre that is exactly a focal length scaled by s.
    """
    s = clip[0] / work[0]
    L = len(p)
    K = np.stack([intrinsics(p[i, 0] * s, clip) for i in range(L)])
    R = rodrigues(p[:, 1:4])
    t = p[:, 4:7]
    Hc = K @ np.concatenate([R[:, :, :2], t[:, :, None]], 2)
    Hinv = np.linalg.inv(Hc)
    Hinv /= Hinv[:, 2:3, 2:3]
    return {"K": K.astype(np.float32), "R_w2c": R.astype(np.float32), "t_w2c": t.astype(np.float32),
            "H_img2court": Hinv.astype(np.float64)}


def draw_court(img: np.ndarray, p: np.ndarray, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Project the court model onto an image at the working resolution."""
    import cv2

    size = img.shape[1::-1]
    H = homography(p, size)
    s = np.linspace(0, 1, 60)[:, None]
    for x0, y0, x1, y1 in COURT:
        uv = project_points(H[None], np.c_[x0 + s * (x1 - x0), y0 + s * (y1 - y0)])[0]
        uv = uv[np.all(np.isfinite(uv), 1)]
        if len(uv) > 1:
            pts = np.round(uv * 4).astype(np.int32)
            cv2.polylines(img, [pts], False, color, thickness, cv2.LINE_AA, shift=2)
    return img


def review_image(frames: list[np.ndarray], cal: ClipCalibration, picks: list[int], out: Path) -> None:
    """Court overlay on a few frames, side by side (green = calibrated frame, orange = interpolated)."""
    import cv2

    tiles = []
    for i in picks:
        i = min(i, len(frames) - 1)
        img = draw_court(frames[i].copy(), cal.p[i], (0, 255, 0) if cal.ok[i] else (0, 140, 255))
        cv2.putText(img, f"#{i}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(out), np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])


def summarise(p: np.ndarray, ok: np.ndarray, err: np.ndarray, clip_size: tuple[int, int],
              scale: float) -> dict:
    f = p[ok, 0] * scale if ok.any() else p[:, 0] * scale
    fm = float(np.median(f))
    return {
        "ok_fraction": float(ok.mean()),
        "f_px_median": fm,
        "hfov_deg": float(2 * np.degrees(np.arctan(clip_size[0] / 2 / fm))),
        "f_range_pct": float((f.max() - f.min()) / fm * 100),
        "reproj_px_mean": float(np.nanmean(err[ok])) if ok.any() else math.inf,
    }


def calibrate_and_write(clip: ClipLayout, meta: dict) -> bool:
    """Calibrate one clip from its preview and write camera.npz + camera.jpg if it clears the bar."""
    cid = clip.id
    n = int(meta["n_frames"])
    frames = read_preview(clip.preview)[:n]
    if not frames:
        log(f"[yellow]{cid}: no readable preview.mp4, nothing written[/]")
        return False
    work = (frames[0].shape[1], frames[0].shape[0])
    clip_size = tuple(meta["size"])
    scale = clip_size[0] / work[0]
    boxes = np.load(clip.track)["bbx_xyxy"] / scale if clip.track.exists() else None
    cal = calibrate_clip(frames, boxes, scale, cid)
    if cal is None:
        log(f"[yellow]{cid}: no frame calibrated, nothing written[/]")
        return False
    # preview.mp4 lacks clip.mp4's last frame: hold the last camera there (not ok)
    pad = n - len(frames)
    p = np.concatenate([cal.p, np.repeat(cal.p[-1:], pad, 0)])
    ok = np.r_[cal.ok, np.zeros(pad, bool)]
    err = np.r_[cal.err_px, np.full(pad, np.nan)] * scale
    report = summarise(p, ok, err, clip_size, scale)
    msg = (f"{cid}: ok {report['ok_fraction']:.0%}, f {report['f_px_median']:.0f}px "
           f"(hfov {report['hfov_deg']:.1f}°, zoom range {report['f_range_pct']:.1f}%), "
           f"reproj {report['reproj_px_mean']:.2f}px, ")
    msg += (f"fixed camera centre ({cal.centre[0]:.1f}, {cal.centre[1]:.1f}, {cal.centre[2]:.1f}) m"
            if cal.centre is not None else "moving camera centre")
    if report["ok_fraction"] < MIN_OK_FRACTION or report["reproj_px_mean"] > MAX_CLIP_REPROJ_PX:
        log(f"[yellow]{msg}: below the bar, nothing written[/]")
        return False
    centre_ = cal.centre if cal.centre is not None else np.full(3, np.nan)
    # written under a temporary name and renamed: `r3d remote` pushes camera.npz and must never see half
    # a file (remote.py never syncs *.part.*)
    tmp = clip.camera.with_name(f"{clip.camera.stem}.part.npz")
    np.savez_compressed(tmp, **to_clip_pixels(p, work, clip_size), ok=ok, reproj_px=err.astype(np.float32),
                        method=np.array(METHOD), centre=np.asarray(centre_, np.float32))
    tmp.replace(clip.camera)
    review_image(frames, cal, [0, meta["contact_index"], len(frames) - 1], clip.camera_preview)
    log(msg)
    return True


def run(cfg: Config, only: list[str] | None = None, force: bool = False) -> None:
    tried, written = [], []
    for cid in cfg.run.clip_ids():
        if only and cid not in only:
            continue
        clip = cfg.run.clip(cid)
        if clip.camera.exists() and not force:
            continue
        # a calibration that no longer passes must not survive a re-run
        clip.camera.unlink(missing_ok=True)
        clip.camera_preview.unlink(missing_ok=True)
        meta = read_json(clip.meta)
        if meta["view"] not in CALIBRATED_VIEWS:
            log(f"{cid}: {meta['view']} view, not calibrated")
            continue
        tried.append(cid)
        with step(f"{cid}: court calibration"):
            try:
                if calibrate_and_write(clip, meta):
                    written.append(cid)
            except Exception as e:  # noqa: BLE001 - one unreadable clip must not stop the others
                log(f"[red]{cid}: calibration failed, nothing written: {e!r}[/]")
    if tried:
        missed = [c for c in tried if c not in written]
        log(f"camera: {len(written)}/{len(tried)} clips calibrated"
            + (f", nothing written for {', '.join(missed)}" if missed else ""))
