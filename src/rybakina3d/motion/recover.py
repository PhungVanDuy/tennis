"""Stage `motion`: world-grounded SMPL-X motion for every clip with GVHMR (GPU host).

We hand GVHMR our own inputs instead of its defaults:
  * person boxes from `track` (preprocess/bbx.pt) — its largest-person tracker picks the wrong player;
  * the camera: the court-line calibration (clips/<id>/camera.npz) gives per-frame intrinsics and decides
    static vs moving (rotation + optical-centre travel); its rotations drive GVHMR when the camera moves.
    Without a calibration: the view's typical focal length, static if the background barely moves;
  * GVHMR's own world trajectory (velocity prior + foot contacts) even for a static camera — the wrapper's
    default for static cameras rebuilds the world path from the in-camera translation, whose depth noise
    shows up as 6-20 cm/frame foot skate on broadcast footage (0.2-0.5 cm/frame without it);
  * full-resolution crops (img_ds=1.0) — a near-court player is only ~250 px tall in broadcast.

Output contract — clips/<id>/motion.npz (all SMPL quantities Y-up, metres):
  fps                play rate (= source fps; frames are 1:1 with the source)
  global_orient      (L,3)   axis-angle, world
  body_pose          (L,63)  21 SMPL-X body joints, axis-angle, local
  betas              (10,)   shape (mean over the clip)
  transl             (L,3)
  joints_world       (L,24,3) SMPL joints in the world frame (ground at y=0)
  rest_joints        (24,3)  SMPL joints in the zero pose (T-pose) for `betas`, pelvis-relative
  static_conf        (L,6)   P(static) for [l_ankle, l_foot, r_ankle, r_foot, l_wrist, r_wrist]
  joints_camera      (L,24,3) in the camera frame;  K (L,3,3) intrinsics of clip.mp4
  kp2d               (L,17,3) ViTPose COCO-17 (x, y, conf) in clip.mp4 pixels
  contact_index, static_camera, camera_motion_px, f_px, stroke
  calibrated, camera_rotation_deg, camera_shift_m   (the last two NaN without a court calibration)
"""

from __future__ import annotations

import math
import shutil
import sys

import numpy as np

from ..config import Config
from ..util import log, read_json, step
from .smpl import JOINTS, PARENTS

VIEW_HFOV = {"main_broadcast": 38.0, "behind_low": 42.0, "side": 42.0}
STATIC_ROT_DEG = 0.05    # a calibrated camera that turns less than this over the clip …
STATIC_SHIFT_M = 0.05    # … and whose optical centre moves less than this counts as static


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _full_res_crops() -> None:
    """Make every GVHMR crop read the full-resolution frame (it decodes at half resolution by default)."""
    import gvhmr.utils.imgcrop as imgcrop

    orig = getattr(imgcrop.get_batch, "__wrapped_orig__", imgcrop.get_batch)

    def get_batch(input_path, bbx_xys, img_ds=0.5, *args, **kwargs):
        return orig(input_path, bbx_xys, 1.0, *args, **kwargs)

    get_batch.__wrapped_orig__ = orig
    for name, mod in list(sys.modules.items()):
        if name.startswith("gvhmr") and getattr(mod, "get_batch", None) is orig:
            mod.get_batch = get_batch
    imgcrop.get_batch = get_batch


def camera_motion_px(video, bbx: np.ndarray, width: int = 480) -> float:
    """Median background optical flow per frame (full-res pixels), with the player masked out."""
    import cv2

    from ..media import read_frames

    frames = read_frames(video)
    if len(frames) < 3:
        return 0.0
    h0, w0 = frames[0].shape[:2]
    s = width / w0
    grays = [cv2.cvtColor(cv2.resize(f, (width, round(h0 * s))), cv2.COLOR_BGR2GRAY) for f in frames]
    mags = []
    for i in range(0, len(grays) - 1, 2):
        flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mask = np.ones(grays[i].shape, bool)
        x0, y0, x1, y1 = (bbx[i] * s).astype(int)
        pad_x, pad_y = (x1 - x0) // 2, (y1 - y0) // 4
        mask[max(0, y0 - pad_y): y1 + pad_y, max(0, x0 - pad_x): x1 + pad_x] = False
        mags.append(float(np.median(np.linalg.norm(flow[mask], axis=-1))))
    return float(np.median(mags)) / s


def _export(res, clip, meta: dict, extra: dict) -> None:
    import torch
    from gvhmr.inference._smpl import params_to_verts_joints

    def np_(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    w = {k: np_(v) for k, v in res.smpl_params_world.items()}
    betas = w["betas"].mean(0) if w["betas"].ndim == 2 else w["betas"]
    n = w["global_orient"].shape[0]
    zero = {
        "global_orient": torch.zeros(1, 3), "body_pose": torch.zeros(1, 63),
        "betas": torch.from_numpy(betas[None]).float(), "transl": torch.zeros(1, 3),
    }
    _, rest, _ = params_to_verts_joints(zero, device="cuda")
    rest = np_(rest)[0]
    rest = rest - rest[0]
    raw = res.raw
    static = _sigmoid(np_(raw["net_outputs"]["static_conf_logits"]).reshape(n, -1))
    kp2d = torch.load(clip.gvhmr_dir / "preprocess" / "vitpose.pt", weights_only=False)
    np.savez_compressed(
        clip.motion,
        fps=float(meta["play_fps"]),
        global_orient=w["global_orient"].astype(np.float32),
        body_pose=w["body_pose"].reshape(n, -1)[:, :63].astype(np.float32),
        betas=betas.astype(np.float32),
        transl=w["transl"].astype(np.float32),
        joints_world=np_(res.joints_world).astype(np.float32),
        rest_joints=rest.astype(np.float32),
        static_conf=static.astype(np.float32),
        joints_camera=np_(res.joints_camera).astype(np.float32),
        K=np_(res.intrinsics).astype(np.float32),
        kp2d=np_(kp2d).astype(np.float32),
        contact_index=int(meta["contact_index"]),
        stroke=meta["stroke"],
        **extra,
    )


def _overlay(clip, meta: dict) -> None:
    """Projected SMPL skeleton over the clip, re-stamped at the source rate for side-by-side review."""
    import cv2

    from ..media import read_frames, retime

    m = np.load(clip.motion)
    J, K = m["joints_camera"], m["K"]
    bones = [(j, p) for j, p in enumerate(PARENTS) if p >= 0]
    frames = read_frames(clip.video)
    tmp = clip.root / "overlay_raw.mp4"
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), meta["play_fps"], (w, h))
    for f, img in enumerate(frames[: len(J)]):
        p = (K[f] @ J[f].T).T
        uv = (p[:, :2] / p[:, 2:3]).astype(int)
        for a, b in bones:
            col = (80, 200, 255) if JOINTS[a].startswith("left_") else (255, 120, 80)
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), col, 4, cv2.LINE_AA)
        for u in uv[:22]:
            cv2.circle(img, tuple(u), 5, (255, 255, 255), -1, cv2.LINE_AA)
        vw.write(img)
    vw.release()
    retime(tmp, clip.overlay, meta["play_fps"], height=720)
    tmp.unlink(missing_ok=True)


def run(cfg: Config, only: list[str] | None = None, force: bool = False) -> None:
    import gvhmr
    import torch
    from gvhmr.utils.geo.hmr_cam import get_bbx_xys_from_xyxy

    _full_res_crops()
    pipe = None
    for cid in cfg.run.clip_ids():
        if only and cid not in only:
            continue
        clip = cfg.run.clip(cid)
        if not clip.track.exists():
            log(f"[yellow]{cid}: no track, skipped[/]")
            continue
        if clip.motion.exists() and not force:
            continue
        meta = read_json(clip.meta)
        track = np.load(clip.track)
        pre = clip.gvhmr_dir / "preprocess"
        if force and clip.gvhmr_dir.exists():
            shutil.rmtree(clip.gvhmr_dir)
        pre.mkdir(parents=True, exist_ok=True)
        bbx = torch.from_numpy(track["bbx_xyxy"]).float()
        torch.save({"bbx_xyxy": bbx, "bbx_xys": get_bbx_xys_from_xyxy(bbx, base_enlarge=1.2).float()},
                   pre / "bbx.pt")

        motion_px = camera_motion_px(clip.video, track["bbx_xyxy"])
        cam = _calibration(clip)
        if cam is not None:
            static = cam["rotation_deg"] < STATIC_ROT_DEG and cam["shift_m"] < STATIC_SHIFT_M
            kwargs: dict = {"intrinsics": cam["K"]}
            f_px = float(np.median(cam["K"][:, 0, 0]))
            if not static:
                torch.save(_T_w2c(cam["R_w2c"]), pre / "slam_results.pt")
            how = f"calibrated, turns {cam['rotation_deg']:.2f}°, moves {cam['shift_m']:.2f} m"
        else:
            static = motion_px < cfg.motion.static_motion_px
            hfov = VIEW_HFOV.get(meta["view"], cfg.motion.default_hfov_deg)
            f_px = (meta["size"][0] / 2) / math.tan(math.radians(hfov) / 2)
            kwargs = {"f_px": f_px}
            how = f"uncalibrated, background {motion_px:.1f} px/frame"
        if pipe is None:
            pipe = gvhmr.pipeline("human-motion-recovery", device="cuda")
        with step(f"{cid}: GVHMR ({'static' if static else 'moving'} camera, {how})"):
            res = pipe(str(clip.video), static_camera=static, world_from_incam=False, flip_test=cfg.motion.flip_test,
                       output_dir=str(clip.gvhmr_dir), render=False, progress=False, **kwargs)
            _export(res, clip, meta, {
                "static_camera": static, "camera_motion_px": motion_px, "f_px": f_px, "calibrated": cam is not None,
                "camera_rotation_deg": cam["rotation_deg"] if cam else float("nan"),
                "camera_shift_m": cam["shift_m"] if cam else float("nan"),
            })
            _overlay(clip, meta)


def _calibration(clip) -> dict | None:
    """Per-frame K / R_w2c from camera.npz plus how much the camera turns and travels over the clip."""
    if not clip.camera.exists():
        return None
    cam = np.load(clip.camera)
    R, t = cam["R_w2c"].astype(np.float64), cam["t_w2c"].astype(np.float64)
    rel = np.einsum("nij,kj->nik", R, R[0])  # R_i · R_0ᵀ
    angle = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    centres = -np.einsum("nji,nj->ni", R, t)  # optical centre −Rᵀt in court metres
    return {"K": cam["K"].astype(np.float32), "R_w2c": cam["R_w2c"], "rotation_deg": float(angle.max()),
            "shift_m": float(np.linalg.norm(centres - centres[0], axis=1).max())}


def _T_w2c(R: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4, dtype=np.float32), (len(R), 1, 1))
    T[:, :3, :3] = R
    return T
