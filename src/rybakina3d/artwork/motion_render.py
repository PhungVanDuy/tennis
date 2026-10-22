"""Stage `artwork motion-render`: every moveset clip (+ a turntable) as a video in the sheet's look (GPU host).

The figures of `artwork render`, moving: the driver writes a JSON job and runs blender/artwork_motion.py headless
(Blender binary like `render`); Cycles on the GPU with the stills' lights, shadow catcher, OIDN and AgX, see that
script for the camera (one per clip, layout.MOTION's 3/4 angle relative to her facing at contact, framed on the
whole clip), the floor and the turntable. Clips in curation order, every one the moveset holds (flagged ones
too), always in real time.

Render profiles (layout.PROFILES, `--profile`) set the frame size, the frame rate, the samples and the encoding;
everything else (camera rule, lights, floor, paper, checks) is shared:
  web     MOTION.size (1024 x 1024), 25 fps: frame i of a video is frame i of the clip (the source footage's
          rate), 48 samples, CRF 18, + matte videos and the turntable. artwork/motion/render/, frames in
          artwork/motion/frames/<name>/. The site's videos and motion-paint's input.
  quest3  2064 x 2208 (one Meta Quest 3 eye, portrait), 60 fps: output frame k shows the action at
          blender/artwork_motion.frame_times (25/60 clip frames apart, phased so the contact is an output frame;
          the importer's linear keys interpolated in between), 16 samples, H.264 High @ 5.2, CRF 16. No matte
          video, no turntable (render-only, never painted). artwork/motion/render_quest3/, frames/quest3/<name>/.
  draft   quest3's camera, framing and timing at 1/4 linear size (516 x 552), 16 samples: a fast preview.
          artwork/motion/render_draft/, frames/draft/<name>/.
The camera is fitted to the profile's (w, h) (a portrait frame binds on width for a lunge, on height for a
serve), and the pixel constants below were tuned at 1024 px: distances (EDGE_FADE_PX, NEAR_PX, FLOOR_CLEAR_PX)
scale with RenderProfile.scale, alpha levels (ALPHA_MIN, SHADOW_FLOOR) are coverage and do not.

Video models do not take alpha, so each frame goes over the sheet's paper here, in display space like the stills
(`composite`): flat paper colour (paint.paper_colour(); no vignette or grain, which would flicker), the figure as
rendered (matte plus its rim), and outside it only the contact shadow: shadow-catcher alpha above
paint.SHADOW_FLOOR inside the floor region her shadow can reach (`shadow_region`), so the catcher's faint
noise over the empty floor never prints. Encoded with ffmpeg on the host while Blender renders the next clip.

Outputs (RunLayout.motion_renders_for(profile): artwork/motion/render/ for web, render_<profile>/ otherwise):
  <clip>.mp4        H.264 yuv420p (BT.709 limited), the profile's CRF / preset / x264 options, +faststart, the
                    profile's size and fps, n_frames frames
  <clip>_matte.mp4  (profiles with matte) the Workbench matte as luminance (255 = figure, no shadow), same frames
  <clip>.jpg        poster: the contact frame
  turntable.mp4 / turntable_matte.mp4 / turntable.jpg   (web) the idle pose, one turn in MOTION.turntable_s
  renders.json      "profile" + settings + {"clips": {id: row} in curation order, "turntable": row or null}; a row:
                    id, kind, stroke, label_ok, video / matte (if made) / poster (run-relative), n_frames, fps,
                    duration_s, contact_frame (index in the video, output frames), source (the clip's fps, frames,
                    contact and, off the clip rate, every output frame's action time), size, px_per_m (at the
                    camera target), ground_y_px / ground_xy_px (the floor under her feet at contact),
                    ground_track_px (per frame), ground_box_px (where shadow may print), floor_z (+ floor_z_track),
                    bbox_px (union of the figure over the clip) + frame_bbox_px, camera (azimuth, elevation,
                    facing_deg, distance_m, lens_mm, f_px, shift, target, matrix_zup), checks (frames touching the
                    border, shadow flicker), fingerprint, mb, frames_mb (its PNGs on the host), seconds
PNG frames stay on the host: <frames>/<name>/rgba_####.png, matte_####.png (+ blender.json), <frames> = frames/
(web) or frames/<profile>/, each with its own job.json and blender.log; a profile only ever clears its own frames.
A video is skipped while its fingerprint (GLB bytes, MOTION_VERSION, MOTION's camera, the profile's size / samples
/ fps / encoding, the clip and its contact, the paper) matches and its files exist; frames rendered under the
same render key (the fingerprint minus paper and encoding, recorded by Blender) are only re-encoded; --force
renders everything again. Web's fingerprint is the formula from before profiles existed, unchanged, so its videos
stay cached; every other profile's adds the whole profile and PROFILE_VERSION.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np

from .. import media
from ..assemble import blender_binary
from ..blender import SCRIPTS_DIR
from ..blender.artwork_motion import frame_times
from ..config import Config
from ..util import log, read_json, step, write_json
from .layout import MOTION, PROFILES, RenderProfile
from .paint import SHADOW_FLOOR, paper_colour

MOTION_VERSION = 1   # bump when the Blender side, the composite or the encoding changes how a video looks
PROFILE_VERSION = 1  # bump for a change that only touches the non-web profiles (sub-frame timing, scaled constants)
SAMPLES = PROFILES["web"].samples   # 48, below MOTION.samples: see layout.PROFILES
FPS = 25.0           # the moveset's (and the source footage's) frame rate: the actions are keyed at it
TURNTABLE = "turntable"
FIRST = ("backhand_14m19s", "serve_16m13s")   # rendered first, so problems show early
CRF, MATTE_CRF, PRESET = (PROFILES["web"].crf, PROFILES["web"].matte_crf,   # web's encoding: 18, 12, slow
                          PROFILES["web"].preset)
# pixel distances at the web frame (1024 px); a profile scales them by RenderProfile.scale
EDGE_FADE_PX = 24    # the shadow fades out over this many px at the edges of its region and of the frame
NEAR_PX = 2          # within this of the figure's matte everything prints as rendered (her rim, AA edges)
FLOOR_CLEAR_PX = 4   # the flicker check looks at the floor this far from the figure
ALPHA_MIN = 8        # matte alpha (0-255) that counts as figure (bbox, border check): one of MATTE_AA's 8 samples
                     # hitting her is 32, so this is coverage, the same at any size
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


def kernel(px: float, scale: float = 1.0) -> np.ndarray:
    """Square dilation kernel reaching `px` (at the web frame) * `scale` px from the centre: 5 x 5 for 2 px."""
    k = 2 * max(int(round(px * scale)), 1) + 1
    return np.ones((k, k), np.uint8)


def composite(rgba: np.ndarray, matte: np.ndarray, paper: tuple[int, int, int], region: np.ndarray,
              scale: float = 1.0) -> np.ndarray:
    """One RGB video frame: the render (straight-alpha RGBA, 8 bit) over flat paper. Within NEAR_PX * `scale` px
    of the figure (matte alpha) everything as rendered; elsewhere only the contact shadow (alpha above
    SHADOW_FLOOR, rescaled, weighted by `region`), like paint.render_rgba does for the stills."""
    a = rgba[..., 3].astype(np.float32) / 255
    near = cv2.dilate((matte > 5).astype(np.uint8), kernel(NEAR_PX, scale)).astype(bool)
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


def resolve(profile: RenderProfile | str) -> RenderProfile:
    """A profile by name (or as given), web's samples and encoding read from this module's constants."""
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise ValueError(f"unknown render profile {profile!r}: one of {', '.join(PROFILES)}")
        profile = PROFILES[profile]
    if profile.name == "web":
        profile = replace(profile, samples=SAMPLES, crf=CRF, matte_crf=MATTE_CRF, preset=PRESET)
    return profile


ENCODE_FIELDS = ("crf", "matte_crf", "preset", "x264", "matte", "turntable")   # RenderProfile fields Blender ignores
HOUSEKEEPING = ("keep_frames",)   # RenderProfile fields that change no output (never part of a fingerprint)


def _spec(glb_sha: str, item: dict, paper: tuple[int, int, int] | None, p: RenderProfile, encode: bool) -> dict:
    spec = {"v": MOTION_VERSION, "glb": glb_sha, "azimuth": MOTION.azimuth, "elevation": MOTION.elevation,
            "size": list(p.size), "samples": p.samples, "fps": p.fps,
            **({"paper": list(paper), "encode": [p.crf, p.matte_crf, p.preset]} if encode else {}),
            **{k: item.get(k) for k in ("kind", "clip", "contact", "n_frames")}}
    if p.name != "web":
        spec["profile"] = {**{k: v for k, v in asdict(p).items()
                              if k not in HOUSEKEEPING and (encode or k not in ENCODE_FIELDS)},
                           "source_fps": FPS, "version": PROFILE_VERSION}
    return spec


def fingerprint(glb_sha: str, item: dict, paper: tuple[int, int, int], profile: RenderProfile | str = "web") -> str:
    """What a video depends on: the moveset, this script's version, MOTION's camera, the profile's size / samples
    / fps / encoding, the item (clip + contact + its frame count at the clip rate, or the turntable's length) and
    the paper. Web's key is exactly the one from before profiles, so its videos stay cached; any other profile
    adds all its fields, the clips' rate and PROFILE_VERSION."""
    spec = _spec(glb_sha, item, paper, resolve(profile), encode=True)
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def render_key(glb_sha: str, item: dict, profile: RenderProfile | str = "web") -> str:
    """What the PNG frames depend on: the fingerprint without the paper and the encoding (both applied by
    `encode`). Blender records it, so a changed encoding re-encodes kept frames instead of rendering them again
    (hours at the headset size)."""
    spec = _spec(glb_sha, item, None, resolve(profile), encode=False)
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def row(item: dict, blender: dict, encoded: dict, root: Path, fps: float = FPS) -> dict:
    """The renders.json row of one video: Blender's camera / framing record + the encoder's files and checks;
    frame numbers are the video's own (at `fps`)."""
    keep = ("n_frames", "contact_frame", "size", "source", "px_per_m", "ground_y_px", "ground_xy_px",
            "ground_track_px", "ground_box_px", "floor_z", "floor_z_track", "air_m", "framed_bbox_px", "azimuths",
            "camera", "racket")
    n = int(blender["n_frames"])
    return {"id": item["name"], "kind": item["kind"], **{k: item[k] for k in ("stroke", "label_ok") if k in item},
            **{k: str(Path(encoded[k]).relative_to(root)) for k in ("video", "matte", "poster") if k in encoded},
            "fps": fps, "duration_s": round(n / fps, 3), **{k: blender[k] for k in keep if k in blender},
            "bbox_px": union_bbox(encoded["frame_bbox_px"]), "frame_bbox_px": encoded["frame_bbox_px"],
            "checks": encoded["checks"], "fingerprint": item["fingerprint"], "mb": encoded["mb"],
            **({"frames_mb": encoded["frames_mb"]} if "frames_mb" in encoded else {}),
            "seconds": {**blender.get("seconds", {}), "encode": encoded["seconds"]}}


# --- encoding ------------------------------------------------------------------------------------------
def encoder_args(size: tuple[int, int], pix_fmt: str, crf: int, profile: RenderProfile, out: Path) -> list[str]:
    """ffmpeg reading raw frames on stdin → H.264 yuv420p (BT.709 limited) at the profile's fps / preset / x264
    options. Web's command is the one from before profiles (it has no extra options)."""
    fps = f"{profile.fps:g}"
    return [media._bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{size[0]}x{size[1]}", "-framerate", fps, "-i", "-",
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", profile.preset, "-crf", str(crf), *profile.x264,
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-movflags", "+faststart", "-r", fps, str(out)]


def _encoder(out: Path, size: tuple[int, int], pix_fmt: str, crf: int,
             profile: RenderProfile) -> tuple[subprocess.Popen, Path]:
    """The encoder process, writing to a .part file that _finish renames."""
    tmp = out.with_name(f".{out.stem}.part{out.suffix}")
    return subprocess.Popen(encoder_args(size, pix_fmt, crf, profile, tmp), stdin=subprocess.PIPE), tmp


def _finish(proc: subprocess.Popen, tmp: Path, out: Path) -> None:
    proc.stdin.close()
    if proc.wait():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}) writing {out}")
    tmp.replace(out)


def encode(item: dict, blender: dict, paper: tuple[int, int, int], out_dir: Path,
           profile: RenderProfile | str = "web") -> dict:
    """Composite every frame over the paper and encode the video, the matte video (profiles with matte) and the
    contact poster (frame blender["contact_frame"] of the video)."""
    t0 = time.time()
    profile = resolve(profile)
    frames = Path(item["frames_dir"])
    n, size = int(blender["n_frames"]), tuple(blender["size"])
    s = profile.scale
    region = shadow_region(blender.get("ground_box_px"), size, EDGE_FADE_PX * s)
    clear = kernel(FLOOR_CLEAR_PX, s)
    paths = {"video": out_dir / f"{item['name']}.mp4", "poster": out_dir / f"{item['name']}.jpg",
             **({"matte": out_dir / f"{item['name']}_matte.mp4"} if profile.matte else {})}
    video, video_tmp = _encoder(paths["video"], size, "rgb24", profile.crf, profile)
    procs = [(video, video_tmp, paths["video"])]
    matte_v = None
    if profile.matte:
        matte_v, matte_tmp = _encoder(paths["matte"], size, "gray", profile.matte_crf, profile)
        procs.append((matte_v, matte_tmp, paths["matte"]))
    boxes, flicker, prev, png_bytes = [], [], None, 0
    try:
        for i in range(n):
            png_bytes += sum(f.stat().st_size for f in (frames / f"rgba_{i:04d}.png", frames / f"matte_{i:04d}.png")
                             if f.exists())
            rgba = cv2.imread(str(frames / f"rgba_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            matte = cv2.imread(str(frames / f"matte_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
            if rgba is None or matte is None:
                raise FileNotFoundError(f"{item['name']}: frame {i} missing in {frames}")
            rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            m = matte[..., 3]
            rgb = composite(rgba, m, paper, region, s)
            video.stdin.write(rgb.tobytes())
            if matte_v is not None:
                matte_v.stdin.write(np.ascontiguousarray(m).tobytes())
            boxes.append(figure_bbox(m))
            if i == int(blender["contact_frame"]):
                cv2.imwrite(str(paths["poster"]), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
            # shadow flicker: change on the floor between frames, away from the figure in either frame
            floor = (region > 0.5) & ~cv2.dilate((m > 0).astype(np.uint8), clear).astype(bool)
            if prev is not None:
                both = floor & prev[1]
                if both.any():
                    flicker.append(float(np.abs(rgb[both].astype(np.int16) - prev[0][both]).mean()))
            prev = (rgb, floor)
    except BaseException:
        for proc, tmp, _ in procs:
            proc.kill()
            tmp.unlink(missing_ok=True)
        raise
    for proc, tmp, out in procs:
        _finish(proc, tmp, out)
    border = [i for i, b in enumerate(boxes) if touches_border(b, size)]
    return {**{k: str(v) for k, v in paths.items()}, "frame_bbox_px": boxes,
            "checks": {"empty_frames": [i for i, b in enumerate(boxes) if b is None], "border_frames": border,
                       "shadow_flicker": {"mean": round(float(np.mean(flicker)), 3) if flicker else 0.0,
                                          "max": round(float(np.max(flicker)), 3) if flicker else 0.0}},
            "mb": {k: round(paths[k].stat().st_size / 1e6, 3) for k in ("video", "matte", "poster") if k in paths},
            "frames_mb": round(png_bytes / 1e6, 1),   # the PNG frames kept on the host for re-encoding
            "seconds": round(time.time() - t0, 1)}


# --- driver --------------------------------------------------------------------------------------------
def frames_root(run_, profile: RenderProfile | str = "web") -> Path:
    """Where a profile's PNG frames, job.json and blender.log live: frames/ for web (as before profiles),
    frames/<profile>/ otherwise (no clip is named after a profile, so web's clip folders never collide)."""
    name = resolve(profile).name
    return run_.motion_frames if name == "web" else run_.motion_frames / name


def out_frames(n: int | None, profile: RenderProfile) -> int:
    """Output frames of a clip of `n` frames at the clips' rate (frame_times' count)."""
    return len(frame_times(n, 0, FPS, profile.fps)[0]) if n else 0


def items(cfg: Config, profile: RenderProfile | str = "web") -> list[dict]:
    """Every moveset clip in curation order (stroke, label verdict, contact, its frame count at the clips' rate)
    + the turntable (profiles with one), as job entries."""
    profile = resolve(profile)
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
    if profile.turntable:   # one turn in turntable_s at the videos' own rate (web: 25 fps, its key from before)
        out.append({"kind": TURNTABLE, "name": TURNTABLE, "n_frames": int(round(MOTION.turntable_s * profile.fps)),
                    "start_azimuth": 0.0})
    root = frames_root(run_, profile)
    for it in out:
        d = root / it["name"]
        it |= {"frames_dir": str(d), "report": str(d / "blender.json")}
    return out


def _render_order(todo: list[dict]) -> list[dict]:
    first = [t for name in FIRST for t in todo if t["name"] == name]
    return first + [t for t in todo if t not in first and t["kind"] != TURNTABLE] + \
        [t for t in todo if t["kind"] == TURNTABLE]


def _frames_ready(item: dict) -> dict | None:
    """Blender's record for frames already rendered for this item (same render key; a record from before render
    keys: same fingerprint), if they are all there."""
    report = Path(item["report"])
    if not report.exists():
        return None
    rec = read_json(report)
    d = Path(item["frames_dir"])
    if (rec["render_key"] != item.get("render_key")) if rec.get("render_key") else \
            (rec.get("fingerprint") != item["fingerprint"]):
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


def run(cfg: Config, force: bool = False, only: list[str] | None = None, profile: str = "web") -> dict:
    """Render, encode and index every video of one render profile (layout.PROFILES); `only` (clip ids /
    "turntable") limits what is (re)made. A profile reads and writes only its own folder, frames and renders.json."""
    profile_ = resolve(profile)
    run_ = cfg.run
    for path, stage in ((run_.moveset_glb, "retarget"), (run_.retarget_report, "retarget"),
                        (run_.curation, "curate")):
        if not path.exists():
            raise FileNotFoundError(f"no {path} (run `r3d {stage}`)")
    out_dir, frames = run_.motion_renders_for(profile_.name), frames_root(run_, profile_)
    run_.ensure(out_dir, frames)
    report_path = out_dir / "renders.json"
    previous = read_json(report_path) if report_path.exists() else {}
    before = {**previous.get("clips", {}), **({TURNTABLE: previous["turntable"]} if previous.get("turntable") else {})}
    paper = paper_colour()
    glb_sha = file_digest(run_.moveset_glb)
    entries = items(cfg, profile_)
    for it in entries:
        it |= {"fingerprint": fingerprint(glb_sha, it, paper, profile_),
               "render_key": render_key(glb_sha, it, profile_)}
    wanted = [it for it in entries if only is None or it["name"] in only]
    files = ("video", "poster", *(("matte",) if profile_.matte else ()))

    def fresh(it: dict) -> bool:
        old = before.get(it["name"], {})
        return old.get("fingerprint") == it["fingerprint"] and all(
            (run_.root / old.get(k, "-")).exists() for k in files)

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
            futures[name] = pool.submit(encode, item, rec, paper, out_dir, profile_)

        for name, rec in reencode.items():
            submit(name, rec)
        if todo:
            for it in todo:  # stale frames of an older render must not mix in
                shutil.rmtree(it["frames_dir"], ignore_errors=True)
            # fps: the actions' key rate (the importer places keys on whole frames at it); out_fps: the videos'
            job = {"glb": str(run_.moveset_glb), "fps": FPS, "out_fps": profile_.fps,
                   "bones": read_json(run_.retarget_report)["mapping"],
                   "settings": {"samples": profile_.samples, **({"smooth_skin": True} if profile_.smooth_skin else {})},
                   "size": list(profile_.size), "azimuth": MOTION.azimuth, "elevation": MOTION.elevation,
                   "profile": profile_.name, "items": todo}
            job_path = write_json(frames / "job.json", job)
            log_path = frames / "blender.log"
            log_path.unlink(missing_ok=True)
            binary = blender_binary(cfg)
            n_frames = sum(out_frames(it.get("n_frames"), profile_) if it["kind"] == "clip" else it["n_frames"]
                           for it in todo)
            with step(f"blender motion {profile_.name} ({len(todo)} videos, {n_frames} frames, "
                      f"{profile_.size[0]}x{profile_.size[1]} at {profile_.fps:g} fps, {binary})"):
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
                             run_.root, profile_.fps)
            if not profile_.keep_frames:  # blender.json stays: the record of what was rendered
                for png in Path(item["frames_dir"]).glob("*.png"):
                    png.unlink()
            c = rows[name]["checks"]
            log(f"{name}: {rows[name]['n_frames']} frames, {rows[name]['mb']['video']:.1f} MB"
                f"{', touches the border in ' + str(len(c['border_frames'])) + ' frames' if c['border_frames'] else ''}"
                f", shadow flicker {c['shadow_flicker']['max']:.2f}")

    report = {"profile": profile_.name, "glb": str(run_.moveset_glb.relative_to(run_.root)), "version": MOTION_VERSION,
              **({"profile_version": PROFILE_VERSION} if profile_.name != "web" else {}), **meta,
              "samples": profile_.samples, "fps": profile_.fps, "source_fps": FPS, "size": list(profile_.size),
              "azimuth": MOTION.azimuth, "elevation": MOTION.elevation,
              **({"turntable_s": MOTION.turntable_s} if profile_.turntable else {}), "paper": list(paper),
              "encoding": {"codec": "libx264", "pix_fmt": "yuv420p", "crf": profile_.crf, "preset": profile_.preset,
                           "x264": list(profile_.x264), "colour": "bt709",
                           **({"matte_crf": profile_.matte_crf,
                               "matte": "luma = figure coverage (255 figure, 0 background), no shadow"}
                              if profile_.matte else {})},
              "seconds": {"this_run": round(time.time() - t0, 1),
                          "render_total": round(sum(sum(r.get("seconds", {}).get(k, 0) for k in
                                                        ("setup", "cycles", "matte")) for r in rows.values()), 1)},
              "clips": {it["name"]: rows[it["name"]] for it in entries if it["kind"] == "clip" and it["name"] in rows},
              "turntable": rows.get(TURNTABLE)}
    write_json(report_path, report)
    for name in set(before) - set(by_name):  # clips no longer in the moveset (this profile's own files only)
        for k in ("video", "matte", "poster"):
            if before[name].get(k):
                (run_.root / before[name][k]).unlink(missing_ok=True)
    missing = [it["name"] for it in entries if it["name"] not in rows]
    log(f"{profile_.name}: {len(report['clips'])} clip videos{' + turntable' if report['turntable'] else ''}"
        f" → {out_dir}{f' ({len(missing)} missing)' if missing else ''}")
    if failures:
        raise RuntimeError(f"motion-render failed: {failures}")
    return report
