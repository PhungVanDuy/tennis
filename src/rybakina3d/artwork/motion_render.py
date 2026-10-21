"""Stage `artwork motion-render`: every moveset clip (+ a turntable) as a video in the sheet's look (GPU host).

The figures of `artwork render`, moving: the driver writes a JSON job and runs blender/artwork_motion.py headless
(Blender binary like `render`); Cycles on the GPU with the stills' lights, shadow catcher, OIDN and AgX, see that
script for the camera (one per clip, layout.MOTION's 3/4 angle relative to her facing at contact, framed on the
whole clip), the floor and the turntable. Clips in curation order, every one the moveset holds (flagged ones
too); frame i of a video is frame i of the clip at the clip's real-time rate (25 fps, like the source footage).

Video models do not take alpha, so each frame goes over the sheet's paper here, in display space like the stills
(`composite`): flat paper colour (paint.paper_colour(); no vignette or grain, which would flicker), the figure as
rendered (matte plus its rim), and outside it only the contact shadow: shadow-catcher alpha above
paint.SHADOW_FLOOR inside the floor region her shadow can reach (`shadow_region`), so the catcher's faint
noise over the empty floor never prints. Encoded with ffmpeg on the host while Blender renders the next clip.

Outputs (artwork/motion/render/):
  <clip>.mp4        H.264 yuv420p (BT.709), CRF CRF, +faststart, MOTION.size, 25 fps, n_frames frames
  <clip>_matte.mp4  the Workbench matte as luminance (255 = figure, no shadow), same frames, CRF MATTE_CRF
  <clip>.jpg        poster: the contact frame
  turntable.mp4 / turntable_matte.mp4 / turntable.jpg   the idle pose, one turn in MOTION.turntable_s
  renders.json      settings + {"clips": {id: row} in curation order, "turntable": row}; a row: id, kind, stroke,
                    label_ok, video / matte / poster (run-relative), n_frames, fps, duration_s, contact_frame
                    (index in the video), size, px_per_m (at the camera target), ground_y_px / ground_xy_px (the
                    floor under her feet at contact), ground_track_px (per frame), ground_box_px (where shadow may
                    print), floor_z (+ floor_z_track), bbox_px (union of the figure over the clip) + frame_bbox_px,
                    camera (azimuth, elevation, facing_deg, distance_m, lens_mm, f_px, shift, target, matrix_zup),
                    checks (frames touching the border, shadow flicker), fingerprint, mb, seconds
PNG frames stay on the host: artwork/motion/frames/<name>/rgba_####.png, matte_####.png (+ blender.json).
A video is skipped while its fingerprint (GLB bytes, MOTION_VERSION, MOTION's camera / size / samples, the clip
and its contact, the paper, the encoding) matches and its files exist; frames rendered under the same
fingerprint are only re-encoded; --force renders everything again.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .. import media
from ..assemble import blender_binary
from ..blender import SCRIPTS_DIR
from ..config import Config
from ..util import log, read_json, step, write_json
from .layout import MOTION
from .paint import SHADOW_FLOOR, paper_colour

MOTION_VERSION = 1   # bump when the Blender side, the composite or the encoding changes how a video looks
# below layout.MOTION.samples (64): 2681 frames at 64 take ~65 min on the A100 (path tracing is GPU-bound, ~1.3 s a
# frame); at 48 the whole run fits in an hour and, after OIDN, a frame differs from 64's by 0.7/255 on average
SAMPLES = min(MOTION.samples, 48)
FPS = 25.0           # the moveset's (and the source footage's) frame rate
TURNTABLE = "turntable"
FIRST = ("backhand_14m19s", "serve_16m13s")   # rendered first, so problems show early
CRF, MATTE_CRF, PRESET = 18, 12, "slow"
EDGE_FADE_PX = 24    # the shadow fades out over this many px at the edges of its region and of the frame
ALPHA_MIN = 8        # matte alpha (0-255) that counts as figure (bbox, border check)
WORKERS = 2          # encoders running beside Blender


# --- numpy: framing data, composite, checks -----------------------------------------------------------
def shadow_region(box: list[float] | None, size: tuple[int, int], fade: float = EDGE_FADE_PX) -> np.ndarray:
    """(H, W) float32 weight of where a contact shadow may print: 1 inside `box` (px), easing to 0 over `fade`
    px outside it and over `fade` px inside the frame's border."""
    W, H = size
    if box is None:
        return np.ones((H, W), np.float32)
    x0, y0, x1, y1 = box
    ys, xs = np.arange(H, dtype=np.float32)[:, None], np.arange(W, dtype=np.float32)[None, :]

    def ease(d: np.ndarray) -> np.ndarray:   # d = px inside the edge
        t = np.clip(d / max(fade, 1e-6), 0, 1)
        return t * t * (3 - 2 * t)

    inside_box = np.minimum(np.minimum(xs - x0, x1 - xs), np.minimum(ys - y0, y1 - ys)) + fade
    inside_frame = np.minimum(np.minimum(xs, W - 1 - xs), np.minimum(ys, H - 1 - ys))
    return (ease(inside_box) * ease(inside_frame)).astype(np.float32)


def composite(rgba: np.ndarray, matte: np.ndarray, paper: tuple[int, int, int], region: np.ndarray) -> np.ndarray:
    """One RGB video frame: the render (straight-alpha RGBA, 8 bit) over flat paper. Within a few px of the
    figure (matte alpha) everything as rendered; elsewhere only the contact shadow (alpha above SHADOW_FLOOR,
    rescaled, weighted by `region`), like paint.render_rgba does for the stills."""
    a = rgba[..., 3].astype(np.float32) / 255
    near = cv2.dilate((matte > 5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    shadow = np.clip((a - SHADOW_FLOOR) / (1 - SHADOW_FLOOR), 0, 1) * region
    alpha = np.where(near, a, shadow)[..., None]
    out = np.asarray(paper, np.float32) * (1 - alpha) + rgba[..., :3].astype(np.float32) * alpha
    return out.round().clip(0, 255).astype(np.uint8)


def figure_bbox(matte: np.ndarray, alpha_min: int = ALPHA_MIN) -> list[int] | None:
    """[x0, y0, x1, y1] (inclusive) of the figure in a matte alpha."""
    ys, xs = np.nonzero(matte >= alpha_min)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def touches_border(bbox: list[int] | None, size: tuple[int, int]) -> bool:
    return bbox is not None and (bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= size[0] - 1 or bbox[3] >= size[1] - 1)


def union_bbox(boxes: list[list[int] | None]) -> list[int] | None:
    b = np.array([x for x in boxes if x is not None])
    return None if not len(b) else [int(b[:, 0].min()), int(b[:, 1].min()), int(b[:, 2].max()), int(b[:, 3].max())]


def file_digest(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(glb_sha: str, item: dict, paper: tuple[int, int, int]) -> str:
    """What a video depends on: the moveset, this script's version, MOTION's camera / size / samples, the item
    (clip + contact, or the turntable's length), the paper and the encoding."""
    spec = {"v": MOTION_VERSION, "glb": glb_sha, "azimuth": MOTION.azimuth, "elevation": MOTION.elevation,
            "size": list(MOTION.size), "samples": SAMPLES, "fps": FPS, "paper": list(paper),
            "encode": [CRF, MATTE_CRF, PRESET], **{k: item.get(k) for k in ("kind", "clip", "contact", "n_frames")}}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def row(item: dict, blender: dict, encoded: dict, root: Path) -> dict:
    """The renders.json row of one video: Blender's camera / framing record + the encoder's files and checks."""
    keep = ("n_frames", "contact_frame", "size", "px_per_m", "ground_y_px", "ground_xy_px", "ground_track_px",
            "ground_box_px", "floor_z", "floor_z_track", "air_m", "framed_bbox_px", "azimuths", "camera", "racket")
    n = int(blender["n_frames"])
    return {"id": item["name"], "kind": item["kind"], **{k: item[k] for k in ("stroke", "label_ok") if k in item},
            **{k: str(Path(encoded[k]).relative_to(root)) for k in ("video", "matte", "poster")},
            "fps": FPS, "duration_s": round(n / FPS, 3), **{k: blender[k] for k in keep if k in blender},
            "bbox_px": union_bbox(encoded["frame_bbox_px"]), "frame_bbox_px": encoded["frame_bbox_px"],
            "checks": encoded["checks"], "fingerprint": item["fingerprint"], "mb": encoded["mb"],
            "seconds": {**blender.get("seconds", {}), "encode": encoded["seconds"]}}


# --- encoding ------------------------------------------------------------------------------------------
def _encoder(out: Path, size: tuple[int, int], pix_fmt: str, crf: int) -> tuple[subprocess.Popen, Path]:
    """ffmpeg reading raw frames on stdin → H.264 yuv420p (BT.709) at FPS, written to a .part file."""
    tmp = out.with_name(f".{out.stem}.part{out.suffix}")
    cmd = [media._bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{size[0]}x{size[1]}", "-framerate", f"{FPS:g}", "-i", "-",
           "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
           "-c:v", "libx264", "-preset", PRESET, "-crf", str(crf), "-colorspace", "bt709", "-color_primaries", "bt709",
           "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart", "-r", f"{FPS:g}", str(tmp)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE), tmp


def _finish(proc: subprocess.Popen, tmp: Path, out: Path) -> None:
    proc.stdin.close()
    if proc.wait():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}) writing {out}")
    tmp.replace(out)


def encode(item: dict, blender: dict, paper: tuple[int, int, int], out_dir: Path) -> dict:
    """Composite every frame over the paper and encode the video, the matte video and the contact poster."""
    t0 = time.time()
    frames = Path(item["frames_dir"])
    n, size = int(blender["n_frames"]), tuple(blender["size"])
    region = shadow_region(blender.get("ground_box_px"), size)
    paths = {"video": out_dir / f"{item['name']}.mp4", "matte": out_dir / f"{item['name']}_matte.mp4",
             "poster": out_dir / f"{item['name']}.jpg"}
    video, video_tmp = _encoder(paths["video"], size, "rgb24", CRF)
    matte_v, matte_tmp = _encoder(paths["matte"], size, "gray", MATTE_CRF)
    boxes, flicker, prev = [], [], None
    try:
        for i in range(n):
            rgba = cv2.imread(str(frames / f"rgba_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            matte = cv2.imread(str(frames / f"matte_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            if rgba is None or matte is None:
                raise FileNotFoundError(f"{item['name']}: frame {i} missing in {frames}")
            rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            m = matte[..., 3]
            rgb = composite(rgba, m, paper, region)
            video.stdin.write(rgb.tobytes())
            matte_v.stdin.write(np.ascontiguousarray(m).tobytes())
            boxes.append(figure_bbox(m))
            if i == int(blender["contact_frame"]):
                cv2.imwrite(str(paths["poster"]), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            # shadow flicker: change on the floor between frames, away from the figure in either frame
            floor = (region > 0.5) & ~cv2.dilate((m > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if prev is not None:
                both = floor & prev[1]
                if both.any():
                    flicker.append(float(np.abs(rgb[both].astype(np.int16) - prev[0][both]).mean()))
            prev = (rgb, floor)
    except BaseException:
        for proc, tmp in ((video, video_tmp), (matte_v, matte_tmp)):
            proc.kill()
            tmp.unlink(missing_ok=True)
        raise
    _finish(video, video_tmp, paths["video"])
    _finish(matte_v, matte_tmp, paths["matte"])
    border = [i for i, b in enumerate(boxes) if touches_border(b, size)]
    return {**{k: str(v) for k, v in paths.items()}, "frame_bbox_px": boxes,
            "checks": {"empty_frames": [i for i, b in enumerate(boxes) if b is None], "border_frames": border,
                       "shadow_flicker": {"mean": round(float(np.mean(flicker)), 3) if flicker else 0.0,
                                          "max": round(float(np.max(flicker)), 3) if flicker else 0.0}},
            "mb": {k: round(paths[k].stat().st_size / 1e6, 3) for k in ("video", "matte", "poster")},
            "seconds": round(time.time() - t0, 1)}


# --- driver --------------------------------------------------------------------------------------------
def items(cfg: Config) -> list[dict]:
    """Every moveset clip in curation order (stroke, label verdict, contact) + the turntable, as job entries."""
    run_ = cfg.run
    cur = read_json(run_.curation)
    rows = {r["id"]: r for r in cur["clips"]}
    retarget = read_json(run_.retarget_report)
    in_glb = set(retarget.get("glb", {}).get("animations") or retarget.get("clips", {}))
    out: list[dict] = []
    for cid in cur["kept"]:
        if cid not in rows or cid not in in_glb:
            log(f"[yellow]{cid}: not in the moveset GLB, no video[/]")
            continue
        r, meta = rows[cid], retarget.get("clips", {}).get(cid, {})
        out.append({"kind": "clip", "name": cid, "clip": cid, "stroke": r["stroke"],
                    "label_ok": bool(r.get("label_ok", r.get("ok"))), "n_frames": meta.get("frames"),
                    "contact": int(r.get("contact_index", meta.get("contact_index", 0)))})
    out.append({"kind": TURNTABLE, "name": TURNTABLE, "n_frames": int(round(MOTION.turntable_s * FPS)),
                "start_azimuth": 0.0})
    for it in out:
        d = run_.motion_frames / it["name"]
        it |= {"frames_dir": str(d), "report": str(d / "blender.json")}
    return out


def _render_order(todo: list[dict]) -> list[dict]:
    first = [t for name in FIRST for t in todo if t["name"] == name]
    return first + [t for t in todo if t not in first and t["kind"] != TURNTABLE] + \
        [t for t in todo if t["kind"] == TURNTABLE]


def _frames_ready(item: dict) -> dict | None:
    """Blender's record for frames already rendered under this item's fingerprint, if they are all there."""
    report = Path(item["report"])
    if not report.exists():
        return None
    rec = read_json(report)
    d = Path(item["frames_dir"])
    if rec.get("fingerprint") != item["fingerprint"]:
        return None
    n = int(rec["n_frames"])
    if all((d / f"rgba_{i:04d}.png").exists() and (d / f"matte_{i:04d}.png").exists() for i in range(n)):
        return rec
    return None


def _blender(binary: str, job: Path, log_path: Path, on_done) -> None:
    if not shutil.which(binary):
        raise FileNotFoundError(f"no Blender at {binary!r} (set R3D_BLENDER or remote.blender)")
    cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(SCRIPTS_DIR / "artwork_motion.py"), "--", str(job)]
    with open(log_path, "a") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            fh.write(line)
            if line.startswith("[r3d] done "):
                on_done(line[len("[r3d] done "):].strip())
            elif line.startswith("[r3d]"):
                log(line[6:].rstrip())
        code = proc.wait()
    if code:
        tail = log_path.read_text().splitlines()[-25:]
        raise RuntimeError(f"blender failed ({code}); last lines of {log_path}:\n" + "\n".join(tail))


def run(cfg: Config, force: bool = False, only: list[str] | None = None) -> dict:
    """Render, encode and index every video; `only` (clip ids / "turntable") limits what is (re)made."""
    run_ = cfg.run
    for path, stage in ((run_.moveset_glb, "retarget"), (run_.retarget_report, "retarget"),
                        (run_.curation, "curate")):
        if not path.exists():
            raise FileNotFoundError(f"no {path} (run `r3d {stage}`)")
    out_dir = run_.motion_renders
    run_.ensure(out_dir, run_.motion_frames)
    report_path = out_dir / "renders.json"
    previous = read_json(report_path) if report_path.exists() else {}
    before = {**previous.get("clips", {}), **({TURNTABLE: previous["turntable"]} if previous.get("turntable") else {})}
    paper = paper_colour()
    glb_sha = file_digest(run_.moveset_glb)
    entries = items(cfg)
    for it in entries:
        it["fingerprint"] = fingerprint(glb_sha, it, paper)
    wanted = [it for it in entries if only is None or it["name"] in only]

    def fresh(it: dict) -> bool:
        old = before.get(it["name"], {})
        return old.get("fingerprint") == it["fingerprint"] and all(
            (run_.root / old.get(k, "-")).exists() for k in ("video", "matte", "poster"))

    redo = {it["name"] for it in wanted if force or not fresh(it)}
    rows: dict[str, dict] = {it["name"]: before[it["name"]] for it in entries if it["name"] not in redo and fresh(it)}
    pending = [it for it in wanted if it["name"] in redo]
    reencode = {it["name"]: rec for it in pending if not force and (rec := _frames_ready(it))}
    todo = _render_order([it for it in pending if it["name"] not in reencode])
    by_name = {it["name"]: it for it in entries}
    t0 = time.time()
    meta = {k: previous.get(k) for k in ("blender", "device")}
    failures: dict[str, str] = {}
    futures: dict[str, Future] = {}

    with ThreadPoolExecutor(WORKERS) as pool:
        def submit(name: str, rec: dict | None = None) -> None:
            item = by_name[name]
            rec = rec or read_json(Path(item["report"]))
            meta.update(blender=rec.get("blender"), device=rec.get("device"))
            futures[name] = pool.submit(encode, item, rec, paper, out_dir)

        for name, rec in reencode.items():
            submit(name, rec)
        if todo:
            for it in todo:  # stale frames of an older render must not mix in
                shutil.rmtree(it["frames_dir"], ignore_errors=True)
            job = {"glb": str(run_.moveset_glb), "fps": FPS, "bones": read_json(run_.retarget_report)["mapping"],
                   "settings": {"samples": SAMPLES}, "size": list(MOTION.size), "azimuth": MOTION.azimuth,
                   "elevation": MOTION.elevation, "items": todo}
            job_path = write_json(run_.motion_frames / "job.json", job)
            log_path = run_.motion_frames / "blender.log"
            log_path.unlink(missing_ok=True)
            binary = blender_binary(cfg)
            n_frames = sum(it.get("n_frames") or 0 for it in todo)
            with step(f"blender motion ({len(todo)} videos, {n_frames} frames, {binary})"):
                try:
                    _blender(binary, job_path, log_path, submit)
                except Exception as e:  # noqa: BLE001 - encode what finished, then report
                    failures["blender"] = str(e)
        elif not reencode:
            log(f"all {len(rows)} videos up to date (--force re-renders)")
        for name, fut in futures.items():
            try:
                enc = fut.result()
            except Exception as e:  # noqa: BLE001
                failures[name] = f"encode: {e}"
                continue
            item = by_name[name]
            rows[name] = row(item, read_json(Path(item["report"])) if name not in reencode else reencode[name], enc,
                             run_.root)
            c = rows[name]["checks"]
            log(f"{name}: {rows[name]['n_frames']} frames, {rows[name]['mb']['video']:.1f} MB"
                f"{', touches the border in ' + str(len(c['border_frames'])) + ' frames' if c['border_frames'] else ''}"
                f", shadow flicker {c['shadow_flicker']['max']:.2f}")

    report = {"glb": str(run_.moveset_glb.relative_to(run_.root)), "version": MOTION_VERSION, **meta,
              "samples": SAMPLES, "fps": FPS, "size": list(MOTION.size), "azimuth": MOTION.azimuth,
              "elevation": MOTION.elevation, "turntable_s": MOTION.turntable_s, "paper": list(paper),
              "encoding": {"codec": "libx264", "pix_fmt": "yuv420p", "crf": CRF, "matte_crf": MATTE_CRF,
                           "preset": PRESET, "colour": "bt709",
                           "matte": "luma = figure coverage (255 figure, 0 background), no shadow"},
              "seconds": {"this_run": round(time.time() - t0, 1),
                          "render_total": round(sum(sum(r.get("seconds", {}).get(k, 0) for k in
                                                        ("setup", "cycles", "matte")) for r in rows.values()), 1)},
              "clips": {it["name"]: rows[it["name"]] for it in entries if it["kind"] == "clip" and it["name"] in rows},
              "turntable": rows.get(TURNTABLE)}
    write_json(report_path, report)
    for name in set(before) - set(by_name):  # clips no longer in the moveset
        for k in ("video", "matte", "poster"):
            if before[name].get(k):
                (run_.root / before[name][k]).unlink(missing_ok=True)
    missing = [it["name"] for it in entries if it["name"] not in rows]
    log(f"{len(report['clips'])} clip videos{' + turntable' if report['turntable'] else ''} → {out_dir}"
        f"{f' ({len(missing)} missing)' if missing else ''}")
    if failures:
        raise RuntimeError(f"motion-render failed: {failures}")
    return report
