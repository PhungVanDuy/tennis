"""Court-line camera calibration: geometry helpers, recovery of a known camera from synthetic frames and the
camera.npz contract written by the stage."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from rybakina3d import camera as cm
from rybakina3d.config import Config
from rybakina3d.util import write_json

SIZE = (1280, 720)


def look_at(C: np.ndarray, target: np.ndarray) -> np.ndarray:
    """World-to-camera rotation for a camera at C looking at target, court Z up, no roll."""
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def broadcast_camera(f: float = 2100.0, C=(0.5, -33.0, 9.0), target=(0.0, 1.0, 0.0)) -> np.ndarray:
    C = np.asarray(C, float)
    R = look_at(C, np.asarray(target, float))
    return cm.params_from(f, R, -R @ C)


def render(p: np.ndarray, seed: int = 0) -> np.ndarray:
    """A purple hard court on grey with white painted lines, a net tape and an ad board as distractors."""
    rng = np.random.default_rng(seed)
    img = np.full((SIZE[1], SIZE[0], 3), (70, 70, 70), np.uint8)
    H = cm.homography(p, SIZE)
    surface = np.array([[-7, -15], [7, -15], [7, 15], [-7, 15]], float)
    poly = cm.project_points(H[None], surface)[0]
    cv2.fillPoly(img, [np.round(poly).astype(np.int32)], (150, 60, 110))
    for x0, y0, x1, y1 in cm.COURT:
        s = np.linspace(0, 1, 50)[:, None]
        uv = cm.project_points(H[None], np.c_[x0 + s * (x1 - x0), y0 + s * (y1 - y0)])[0]
        cv2.polylines(img, [np.round(uv * 16).astype(np.int32)], False, (235, 235, 235), 3, cv2.LINE_AA, 4)
    # net tape ~1 m above the court centre line: a long white line that is not on the court plane
    R, t = cm.rodrigues(p[1:4]), p[4:7]
    K = cm.intrinsics(p[0], SIZE)
    ends = [K @ (R @ np.array([x, 0.0, 1.0]) + t) for x in (-6.4, 6.4)]
    a, b = (np.round(e[:2] / e[2]).astype(int) for e in ends)
    cv2.line(img, tuple(a), tuple(b), (240, 240, 240), 3, cv2.LINE_AA)
    cv2.line(img, (0, 60), (SIZE[0], 60), (230, 230, 230), 2, cv2.LINE_AA)   # ad board edge
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def rot_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1 @ R2.T) - 1) / 2
    return math.degrees(math.acos(np.clip(c, -1, 1)))


def test_court_model():
    assert cm.COURT.shape == (9, 4)
    widths = cm.COURT[: cm.N_HOR, 2] - cm.COURT[: cm.N_HOR, 0]
    assert np.allclose(widths[[0, 3]], 2 * cm.HALF_DOUBLES)              # baselines span the doubles court
    assert np.allclose(cm.COURT[: cm.N_HOR, 1], np.sort(cm.COURT[: cm.N_HOR, 1]))   # near → far
    assert np.all(np.diff(cm.COURT[cm.N_HOR:, 0]) > 0)                  # left → right
    centre = cm.COURT[6]
    assert centre[0] == 0 and np.isclose(centre[3], cm.COURT[2, 1])     # centre line ends at the service line


def test_square_to_quad_maps_corners():
    q = np.array([[[10, 20], [110, 25], [100, 130], [5, 120]]], float)
    H = cm.square_to_quad(q)[0]
    for (u, v), target in zip([(0, 0), (1, 0), (1, 1), (0, 1)], q[0], strict=True):
        x = H @ [u, v, 1.0]
        assert np.allclose(x[:2] / x[2], target)


def test_decompose_recovers_camera_and_rejects_symmetric_solutions():
    p = broadcast_camera()
    H = cm.homography(p, SIZE)
    f, R, t, ok = cm.decompose(H[None] * 3.7, SIZE)                       # any scale
    assert ok[0]
    assert abs(f[0] - p[0]) < 1e-6 * p[0]
    assert rot_angle_deg(R[0], cm.rodrigues(p[1:4])) < 1e-6
    assert np.allclose(t[0], p[4:7])
    mirror = H @ np.diag([-1.0, 1, 1])              # left/right swapped correspondences
    rotated = H @ np.diag([-1.0, -1, 1])            # the court seen from the far end
    assert not cm.decompose(np.stack([mirror, rotated]), SIZE)[3].any()


def test_project_lines_pass_through_projected_points():
    p = broadcast_camera()
    H = cm.homography(p, SIZE)
    lines = cm.project_lines(p, SIZE)
    for j, (x0, y0, x1, y1) in enumerate(cm.COURT):
        uv = cm.project_points(H[None], np.array([[x0, y0], [x1, y1]]))[0]
        assert np.abs(uv @ lines[j, :2] + lines[j, 2]).max() < 1e-6


def test_calibrates_synthetic_broadcast_frame():
    p = broadcast_camera()
    resp = cm.line_response(render(p))
    fit = cm.calibrate_frame(resp, None, scale=1.5)
    assert fit is not None and cm.frame_ok(fit, 1.5)
    assert abs(fit.p[0] / p[0] - 1) < 0.01
    assert rot_angle_deg(cm.rodrigues(fit.p[1:4]), cm.rodrigues(p[1:4])) < 0.2
    assert np.linalg.norm(cm.centre(fit.p) - cm.centre(p)) < 0.5
    assert fit.err_px < 1.0 and fit.frac > 0.9


def test_calibrate_clip_tracks_a_zoom_with_a_fixed_centre():
    cams = [broadcast_camera(f=2100.0), broadcast_camera(f=2140.0, target=(0.3, 1.2, 0.0)),
            broadcast_camera(f=2180.0, target=(0.6, 1.5, 0.0))]
    cal = cm.calibrate_clip([render(p, seed=i) for i, p in enumerate(cams)], None, scale=1.5)
    assert cal is not None and cal.ok.all()
    assert cal.centre is not None and np.linalg.norm(cal.centre - cm.centre(cams[0])) < 0.5
    for got, want in zip(cal.p, cams, strict=True):
        assert abs(got[0] / want[0] - 1) < 0.01
        assert rot_angle_deg(cm.rodrigues(got[1:4]), cm.rodrigues(want[1:4])) < 0.2


def test_blank_frame_is_not_calibrated():
    blank = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    assert cm.calibrate_frame(cm.line_response(blank), None, 1.5) is None


def test_to_clip_pixels_scales_intrinsics_and_inverts_the_homography():
    p = broadcast_camera()
    out = cm.to_clip_pixels(p[None], SIZE, (1920, 1080))
    K = out["K"][0]
    assert np.isclose(K[0, 0], 1.5 * p[0]) and np.allclose(K[:2, 2], [959.5, 539.5])
    court = np.array([3.0, -8.0])
    X = np.r_[court, 0.0]
    x = K @ (out["R_w2c"][0] @ X + out["t_w2c"][0])
    uv = x[:2] / x[2]
    # preview.mp4 is a centre-aligned downscale of clip.mp4: u_clip = 1.5 u_work + 0.25
    work = cm.project_points(cm.homography(p, SIZE)[None], court[None])[0, 0]
    assert np.allclose(uv, 1.5 * work + 0.25, atol=1e-3)
    back = out["H_img2court"][0] @ np.r_[uv, 1.0]
    assert np.allclose(back[:2] / back[2], court, atol=1e-3)


def test_interpolate_fills_gaps_and_holds_ends():
    a, b = broadcast_camera(f=2000.0), broadcast_camera(f=2200.0, target=(1.0, 1.0, 0.0))
    out = cm._interpolate({1: a, 3: b}, 5)
    assert np.allclose(out[0], a) and np.allclose(out[4], b)
    assert a[0] < out[2, 0] < b[0]


@pytest.mark.parametrize("hfov, ok", [(1.0, False), (30.0, True), (100.0, False)])
def test_plausible_field_of_view(hfov, ok):
    f = SIZE[0] / 2 / math.tan(math.radians(hfov) / 2)
    assert bool(cm.plausible(f, np.array([0.0, -30.0, 8.0]), SIZE)[0]) is ok


def _clip(root, cid: str, view: str, frames: list[np.ndarray] | None, n_frames: int) -> None:
    d = root / "clips" / cid
    d.mkdir(parents=True)
    write_json(d / "clip.json", {"view": view, "n_frames": n_frames, "contact_index": n_frames // 2,
                                 "size": [1920, 1080]})
    if frames:
        vw = cv2.VideoWriter(str(d / "preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, SIZE)
        for f in frames:
            vw.write(f)
        vw.release()


def test_run_writes_the_camera_contract_and_skips_what_it_cannot_calibrate(tmp_path):
    cams = [broadcast_camera(f=2100.0 + 4 * i, target=(0.05 * i, 1.0, 0.0)) for i in range(10)]
    # like the real previews: one frame short of clip.mp4, whose last camera is then held (not ok)
    _clip(tmp_path, "serve_01", "main_broadcast", [render(p, seed=i) for i, p in enumerate(cams)], 11)
    _clip(tmp_path, "side_01", "side", [render(cams[0])], 1)
    _clip(tmp_path, "blank_01", "main_broadcast", [np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)] * 3, 4)
    _clip(tmp_path, "nopreview_01", "behind_low", None, 5)
    (tmp_path / "clips" / "blank_01" / "camera.npz").write_bytes(b"stale")
    cfg = Config.load()
    cfg.run_dir = tmp_path
    cm.run(cfg, force=True)

    out = tmp_path / "clips" / "serve_01"
    cam = np.load(out / "camera.npz")
    assert (out / "camera.jpg").exists() and not (out / "camera.part.npz").exists()
    shapes = {"K": (11, 3, 3), "R_w2c": (11, 3, 3), "t_w2c": (11, 3), "H_img2court": (11, 3, 3), "ok": (11,),
              "reproj_px": (11,)}
    assert {k: cam[k].shape for k in shapes} == shapes
    assert cam["ok"][:10].all() and not cam["ok"][10] and np.isnan(cam["reproj_px"][10])
    assert np.nanmax(cam["reproj_px"]) < 1.5 and str(cam["method"]).startswith("court-lines")
    assert np.allclose(cam["K"][:10, 0, 0], 1.5 * np.array([p[0] for p in cams]), rtol=0.01)
    for i in (0, 9):
        assert rot_angle_deg(cam["R_w2c"][i].astype(float), cm.rodrigues(cams[i][1:4])) < 0.2
    assert np.allclose(cam["K"][10], cam["K"][9]) and np.allclose(cam["R_w2c"][10], cam["R_w2c"][9])
    for cid in ("side_01", "blank_01", "nopreview_01"):
        assert not (tmp_path / "clips" / cid / "camera.npz").exists()
