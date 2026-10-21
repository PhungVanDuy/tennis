"""Stage `artwork render`: every resolved figure (artwork/poses.json) rendered from the moveset GLB (GPU host).

The driver writes a JSON job and runs blender/artwork.py headless (binary like `retarget`: $R3D_BLENDER, else
cfg.remote.blender when it exists, else `blender`); Cycles on the GPU, see that script for camera, framing
and lights. Figure heights: the hero HEIGHTS["hero"] px, other full-body figures HEIGHTS["full"], bust and
head close-ups HEIGHTS["bust"/"head"], the atlas HEIGHTS["atlas"]; widths follow the pose.

Outputs (artwork/renders/):
  <slot>.png, atlas/<clip>.png              RGBA, transparent background, soft contact shadow (full body)
  <slot>_matte.png, atlas/<clip>_matte.png  RGBA, flat dark figure, alpha = coverage without the shadow
  renders.json   {"slots": {slot: fig}, "atlas": {clip: fig}, ...}; fig = png / matte (run-relative),
                 size [w, h], bbox [x0, y0, x1, y1] of the figure (matte alpha), feet_y (= bbox bottom),
                 ground_y / ground_xy (the floor under her feet, below feet_y when she is in the air; null
                 for bust / head), framed_bbox (what the camera was fitted to: figure + floor patch), clip,
                 frame, floor_z, camera (azimuth / elevation relative to her facing, facing_deg, lens, shift,
                 world matrix Z-up)
The shadow is cleaned after Blender (`clean_shadow`): the shadow catcher's sampling speckle over the empty
floor is cut and the shadow fades out before the image border, so a figure composites without a hard edge.
A figure is re-rendered only when its spec (clip, frame, camera, size), the GLB or RENDER_VERSION changed,
or with --force.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..assemble import blender_binary
from ..blender import SCRIPTS_DIR
from ..config import Config
from ..util import log, read_json, step, write_json

RENDER_VERSION = 2   # bump when the Blender side or the clean-up changes how figures look
SAMPLES = 128
HEIGHTS = {"hero": 2000, "full": 1300, "bust": 1000, "head": 1000, "atlas": 700}
SPEC_KEYS = ("clip", "frame", "azimuth", "elevation", "framing", "height", "racket")
ALPHA_MIN = 8        # matte alpha (0-255) that counts as figure for the bbox
SPECKLE_ALPHA = 10   # shadow alpha below this is catcher noise on the empty floor
BORDER_FADE = 0.10   # shadow fades out over this fraction of the image's short side at every border


def figures(cfg: Config, poses: dict) -> list[dict]:
    """Blender job entries for every slot with a pose and every atlas clip."""
    out = cfg.run.artwork_renders
    figs = []
    for s in poses["slots"]:
        if s["pose"] is None:
            log(f"[yellow]{s['slot']}: {s.get('problem', 'no pose')}, not rendered[/]")
            continue
        height = HEIGHTS["hero"] if s["slot"] == "hero" else HEIGHTS[s["framing"]]
        figs.append({"name": s["slot"], "kind": "slot", "clip": s["clip"], "frame": s["frame"],
                     "azimuth": s["azimuth"], "elevation": s["elevation"], "framing": s["framing"], "height": height,
                     "racket": s["pose"] != "idle",  # the turnaround and the close-ups are without the racket
                     "png": str(out / f"{s['slot']}.png"), "matte": str(out / f"{s['slot']}_matte.png")})
    for a in poses["atlas"]:
        figs.append({"name": f"atlas/{a['clip']}", "kind": "atlas", "clip": a["clip"], "frame": a["frame"],
                     "azimuth": a["azimuth"], "elevation": a["elevation"], "framing": a["framing"],
                     "height": HEIGHTS["atlas"], "racket": True,
                     "png": str(out / "atlas" / f"{a['clip']}.png"),
                     "matte": str(out / "atlas" / f"{a['clip']}_matte.png")})
    return figs


def spec_key(fig: dict, glb: Path) -> str:
    st = glb.stat()
    spec = {k: fig.get(k) for k in SPEC_KEYS} | {"v": RENDER_VERSION, "samples": SAMPLES,
                                                 "glb": [st.st_size, int(st.st_mtime)]}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]


def matte_bbox(path: Path) -> list[int] | None:
    """[x0, y0, x1, y1] (inclusive) of the figure's coverage in a matte PNG."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    ys, xs = np.nonzero(img[..., 3] >= ALPHA_MIN)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def clean_shadow(png: Path, matte: Path) -> None:
    """Drop speckle from the shadow (pixels outside the figure) and fade it out towards the borders."""
    import cv2

    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    figure = cv2.imread(str(matte), cv2.IMREAD_UNCHANGED)[..., 3] > 0
    h, w = img.shape[:2]
    band = max(2.0, BORDER_FADE * min(h, w))
    ys, xs = np.arange(h)[:, None], np.arange(w)[None, :]
    t = np.clip(np.minimum(np.minimum(ys, h - 1 - ys), np.minimum(xs, w - 1 - xs)) / band, 0, 1)
    fade = t * t * (3 - 2 * t)
    a = img[..., 3].astype(np.float32)
    shadow = np.where(a < SPECKLE_ALPHA, 0, a) * fade
    img[..., 3] = np.where(figure, a, shadow).round().astype(np.uint8)
    cv2.imwrite(str(png), img)


def _blender(binary: str, job: Path, log_path: Path) -> None:
    if not shutil.which(binary):
        raise FileNotFoundError(f"no Blender at {binary!r} (set R3D_BLENDER or remote.blender)")
    cmd = [binary, "-b", "--factory-startup", "--python-exit-code", "1",
           "--python", str(SCRIPTS_DIR / "artwork.py"), "--", str(job)]
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


def _entry(fig: dict, blender: dict, root: Path, key: str) -> dict:
    png, matte = Path(fig["png"]), Path(fig["matte"])
    bbox = matte_bbox(matte)
    return {"png": str(png.relative_to(root)), "matte": str(matte.relative_to(root)), "clip": fig["clip"],
            "frame": fig["frame"], "framing": fig["framing"], **blender, "bbox": bbox,
            "feet_y": bbox[3] if bbox else None, "key": key}


def run(cfg: Config, force: bool = False) -> dict:
    run_ = cfg.run
    if not run_.artwork_poses.exists():
        raise FileNotFoundError(f"no {run_.artwork_poses} (run `r3d artwork poses`)")
    if not run_.moveset_glb.exists():
        raise FileNotFoundError(f"no moveset at {run_.moveset_glb} (run `r3d retarget`)")
    poses = read_json(run_.artwork_poses)
    out_dir = run_.artwork_renders
    run_.ensure(out_dir / "atlas")
    report_path = out_dir / "renders.json"
    previous = read_json(report_path) if report_path.exists() else {}
    before = {**previous.get("slots", {}), **{f"atlas/{k}": v for k, v in previous.get("atlas", {}).items()}}

    figs = figures(cfg, poses)
    keys = {f["name"]: spec_key(f, run_.moveset_glb) for f in figs}

    def fresh(f: dict) -> bool:
        old = before.get(f["name"], {})
        return old.get("key") == keys[f["name"]] and Path(f["png"]).exists() and Path(f["matte"]).exists()

    todo = [f for f in figs if force or not fresh(f)]
    rendered: dict = {}
    meta = {k: previous.get(k) for k in ("blender", "device", "samples")}
    if todo:
        retarget = read_json(run_.retarget_report)
        job = {"glb": str(run_.moveset_glb), "fps": poses.get("fps", 25.0), "bones": retarget["mapping"],
               "settings": {"samples": SAMPLES}, "figures": todo, "report": str(out_dir / "blender.json")}
        job_path = write_json(out_dir / "job.json", job)
        log_path = out_dir / "blender.log"
        log_path.unlink(missing_ok=True)
        binary = blender_binary(cfg)
        with step(f"blender artwork ({len(todo)}/{len(figs)} figures, {binary})"):
            _blender(binary, job_path, log_path)
        res = read_json(out_dir / "blender.json")
        rendered = res["figures"]
        for f in todo:
            if f["framing"] == "full":
                clean_shadow(Path(f["png"]), Path(f["matte"]))
        meta = {"blender": res["blender"], "device": res["device"], "samples": res["samples"]}
        secs = [r["seconds"]["cycles"] + r["seconds"]["matte"] for r in rendered.values()]
        log(f"rendered {len(secs)} figures in {res['seconds']:.0f}s ({np.mean(secs):.1f}s per figure on "
            f"{res['device']['type']})")
    else:
        log(f"all {len(figs)} figures up to date (--force re-renders)")

    report: dict = {"glb": str(run_.moveset_glb.relative_to(run_.root)), **meta, "render_version": RENDER_VERSION,
                    "slots": {}, "atlas": {}}
    for f in figs:
        entry = (_entry(f, rendered[f["name"]], run_.root, keys[f["name"]]) if f["name"] in rendered
                 else before[f["name"]])
        if f["kind"] == "slot":
            report["slots"][f["name"]] = entry
        else:
            report["atlas"][f["clip"]] = entry
    for name in set(before) - set(keys):  # figures no longer on the sheet
        for k in ("png", "matte"):
            (run_.root / before[name][k]).unlink(missing_ok=True)
    write_json(report_path, report)
    empty = [n for n, e in {**report["slots"], **report["atlas"]}.items() if not e.get("bbox")]
    if empty:
        raise RuntimeError(f"empty renders (nothing in frame): {empty}")
    log(f"{len(report['slots'])} slot figures + {len(report['atlas'])} atlas figures → {out_dir}")
    return report
