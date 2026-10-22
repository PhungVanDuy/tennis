"""Moving figures from the moveset GLB: every clip's action and a turntable, rendered with Cycles (runs INSIDE Blender).

    blender -b --factory-startup --python-exit-code 1 --python artwork_motion.py -- job.json

`rybakina3d.artwork.motion_render` writes the job. The scene, look and pose helpers are the still figures' own
(blender/artwork.py `Studio`: lights following the camera, shadow-catcher floor lit by its own high soft light,
OIDN, AgX, the Workbench matte); this script only changes what moves:

  clip       the clip's whole action in real time. At the clip rate (job fps == out_fps, the web profile) frame i
             of the render is frame i of the clip (scene at the clip rate, keys on whole frames 0..n-1, rendered as
             one animation). At another rate (out_fps, e.g. 60 for a headset) output frame k is the action at
             `frame_times` (a sub-frame between two keys, the importer's linear keys interpolated by Blender), the
             contact landing exactly on an output frame; each frame is then rendered as a still at its sub-frame
             with the floor set for it. ONE camera for the clip, at the job's azimuth / elevation relative to the
             way she faces at contact, aimed at and fitted to the union of every frame's vertices (racket included) plus
             the ground patch around her feet: she moves through a still frame and nothing leaves it. The lights
             stay fixed to that camera and the noise seed stays fixed, so nothing but her changes between frames.
             The floor (`floor_track`) follows her lowest vertex while she is on the ground, like each still's
             floor does, and is interpolated across the frames she is in the air: her feet stay on their shadow,
             a contact frame looks like its atlas still, and a jump leaves the shadow on the court.
  turntable  the idle pose (no racket) seen from a camera orbiting her once (lights orbit with it, like a
             turntable under a studio rig); one lens + shift fitted to every view of the turn.

Frames go to <frames_dir>/rgba_####.png (Cycles, RGBA on a transparent film with the contact shadow) and
matte_####.png (Workbench, alpha = figure coverage), numbered from 0; the paper goes under in the driver, in
display space like the stills. After each video a `[r3d] done <name>` line tells the driver it can encode it.
The frame is the job's size, square or not: the lens is fitted with a vertical sensor fit (sensor height = image
height) and the shift in units of the image height (Blender's viewplane factor for a vertical fit, whatever the
orientation; only AUTO uses the larger side), so a portrait frame holds the whole motion with the same margins.
The numpy half (frame timing, fixed-size framing, floor track, orbit, ground region) imports without Blender for
the unit tests.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import artwork as still  # imported as rybakina3d.blender.artwork_motion (unit tests)
except ImportError:  # inside Blender: a sibling script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import artwork as still

try:
    import bpy
    from mathutils import Matrix
except ImportError:  # imported outside Blender (unit tests of the numpy part)
    bpy = None
    Matrix = None

FEET = ("left_ankle", "right_ankle", "left_foot", "right_foot")
SHADOW_REACH_M = 0.9     # the contact shadow can fall this far from her feet (high floor light, racket overhead)
MATTE_AA = "8"           # Workbench samples for the matte (the stills use 16; a pose check needs no more)


# ------------------------------------------------------------------------------------------------ numpy
def frame_times(n: int, contact: int, fps: float, out_fps: float) -> tuple[np.ndarray, int]:
    """(action time of every output frame in clip frames, the contact's output frame) for a clip of `n` frames at
    `fps` played in real time at `out_fps`. The duration is kept (round(n * out_fps / fps) frames, so the clip's
    last frame is held for its own 1/fps like at the source rate), the step is fps / out_fps clip frames, and the
    grid is phased so the contact falls exactly on an output frame (its pose is the atlas still, and the reel can
    line contacts up on one frame); times are clamped to the clip [0, n-1]. At out_fps == fps: 0, 1, ..., n-1."""
    n, contact = int(n), int(np.clip(contact, 0, n - 1))
    step = float(fps) / float(out_fps)
    n_out = max(1, int(round(n / step)))
    k_contact = min(int(round(contact / step)), n_out - 1)
    times = contact + (np.arange(n_out) - k_contact) * step
    return np.clip(times, 0.0, n - 1.0), k_contact


def fit_square(uv: np.ndarray, size: tuple[int, int], margins: tuple[float, float, float]) -> dict:
    """Lens, shift and focal (px) that put `uv` (normalised coords) inside a fixed `size` frame (square or not),
    centred, with the still framing's margins (fractions of the framed height); whichever axis binds gets the
    margins, the other keeps the slack. Vertical sensor fit (`_setup_camera`): lens_mm = 24 mm * f_px / H, and
    Blender then measures the shift in units of the image height (its viewplane factor for a vertical fit; only
    AUTO uses the larger side), checked against Blender's own projection for square, portrait and landscape."""
    W, H = int(size[0]), int(size[1])
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    span = v1 - v0
    top, side, bottom = (m * span for m in margins)
    u0, u1, v0, v1 = u0 - side, u1 + side, v0 - bottom, v1 + top
    f_px = min(W / (u1 - u0), H / (v1 - v0))
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    return {"width": W, "height": H, "f_px": f_px, "lens_mm": 24.0 * f_px / H, "sensor_height_mm": 24.0,
            "shift_x": cu * f_px / H, "shift_y": cv * f_px / H, "centre": (cu, cv)}


def floor_track(lowest: np.ndarray, grounded_m: float = still.GROUNDED_M) -> np.ndarray:
    """Per-frame floor height from each frame's lowest vertex: at it while she is on the ground (within
    `grounded_m` of z=0, the stills' rule), linearly interpolated across the frames she is in the air (held
    before the first and after the last grounded frame), z=0 throughout if she never comes down. Never above
    her lowest vertex: the shadow catcher hides whatever is below it."""
    lowest = np.asarray(lowest, np.float64)
    down = lowest < grounded_m
    if not down.any():
        return np.zeros_like(lowest)
    t = np.arange(len(lowest))
    return np.minimum(np.interp(t, t[down], lowest[down]), lowest)


def ground_patch(ground: np.ndarray, half: float = still.GROUND_PATCH_M) -> np.ndarray:
    """Floor points `half` around every point of `ground` (the floor under her feet, any shape (..., 3))."""
    g = np.asarray(ground, np.float64).reshape(-1, 3)
    return np.concatenate([g + [dx, dy, 0.0] for dx in (-half, half) for dy in (-half, half)])


def orbit_azimuths(n: int, start: float = 0.0) -> np.ndarray:
    """Azimuths (deg) of a closed turn in n frames: frame n would repeat frame 0, so the video loops."""
    return start + np.arange(int(n), dtype=np.float64) * 360.0 / int(n)


def ground_box(ground: np.ndarray, cams: list[np.ndarray], frame: dict, reach: float = SHADOW_REACH_M) -> list[float]:
    """Pixel box [x0, y0, x1, y1] of the floor within `reach` of the floor points under her feet (any frame) seen
    from any of `cams`: the only place a contact shadow can be; elsewhere the catcher's alpha is noise."""
    g = np.asarray(ground, np.float64).reshape(-1, 3)
    ring = [(reach * math.cos(t), reach * math.sin(t), 0.0) for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
    pts = np.concatenate([g + d for d in ring] + [g])
    px = np.concatenate([still.to_pixels(still.project(pts, cam), frame) for cam in cams])
    W, H = frame["width"], frame["height"]
    (x0, y0), (x1, y1) = np.clip(px.min(axis=0), 0, [W, H]), np.clip(px.max(axis=0), 0, [W, H])
    return [round(float(v), 1) for v in (x0, y0, x1, y1)]


def camera_row(frame: dict, cam: np.ndarray, target: np.ndarray, yaw: float, azimuth: float, elevation: float,
               distance: float) -> dict:
    return {"azimuth": azimuth, "elevation": elevation, "facing_deg": round(math.degrees(yaw), 2),
            "distance_m": round(distance, 3), "lens_mm": round(frame["lens_mm"], 3),
            "sensor_height_mm": frame["sensor_height_mm"], "f_px": round(frame["f_px"], 3),
            "shift": [round(frame["shift_x"], 5), round(frame["shift_y"], 5)],
            "target": np.round(target, 4).tolist(), "matrix_zup": cam.round(5).tolist()}


def _bbox_px(px: np.ndarray) -> list[float]:
    return [round(float(v), 1) for v in (*px.min(axis=0), *px.max(axis=0))]


# ============================================================================== Blender side (bpy below)
def _log(msg: str) -> None:
    print(f"[r3d] {msg}", flush=True)


class MotionStudio(still.Studio):
    """The still figures' studio, rendering whole animations into numbered frames."""

    def __init__(self, job: dict):
        super().__init__(job)
        scene = self.scene
        scene.render.use_persistent_data = True    # keep Cycles' kernels and static scene between frames
        scene.cycles.seed = 0
        scene.cycles.use_animated_seed = False     # the same noise every frame: a static floor does not shimmer
        scene.render.use_file_extension = True
        scene.display.render_aa = MATTE_AA
        self.size = tuple(int(v) for v in job["size"])
        # one tile per frame: past the default 2048 px a 2064 x 2208 frame would render as a big tile plus thin
        # 16 / 160 px strips that starve the GPU (one tile measured 9% faster at that size); web's 1024 px frame
        # is one tile either way
        scene.cycles.tile_size = max(int(scene.cycles.tile_size), *self.size)
        self.fps = float(job.get("fps", 25.0))                  # the actions' key rate (the clips' rate)
        self.out_fps = float(job.get("out_fps", self.fps))      # the videos' rate
        self.azimuth, self.elevation = float(job["azimuth"]), float(job["elevation"])

    # ---------------------------------------------------------------------------------------- shared
    def _setup_camera(self, frame: dict, cam: np.ndarray, distance: float) -> None:
        data = self.cam.data
        data.type = "PERSP"
        data.sensor_fit = "VERTICAL"
        data.sensor_height = frame["sensor_height_mm"]
        data.lens = frame["lens_mm"]
        data.shift_x, data.shift_y = frame["shift_x"], frame["shift_y"]
        data.clip_start, data.clip_end = 0.05, distance * 4 + 50
        self.cam.matrix_world = Matrix(cam)
        self.scene.render.resolution_x, self.scene.render.resolution_y = frame["width"], frame["height"]

    def _at(self, t: float) -> None:
        """Evaluate the scene at action time `t` (clip frames; a fraction is a sub-frame)."""
        f = int(math.floor(t))
        self.scene.frame_set(f, subframe=float(t - f))

    @staticmethod
    def _keys(obj, path: str, index: int, values, interpolation: str) -> None:
        """One key per frame (0, 1, ...) on obj.<path>[index]."""
        prefs = bpy.context.preferences.edit
        before, prefs.keyframe_new_interpolation_type = prefs.keyframe_new_interpolation_type, interpolation
        try:
            for f, v in values:
                getattr(obj, path)[index] = float(v)
                obj.keyframe_insert(path, index=index, frame=f)
        finally:
            prefs.keyframe_new_interpolation_type = before

    def _animation(self, frames_dir: Path, n: int) -> dict:
        """Render frames 0..n-1 twice: Cycles with the shadow, then the Workbench matte."""
        scene = self.scene
        scene.frame_start, scene.frame_end, scene.frame_step = 0, n - 1, 1
        frames_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        self.floor.hide_render = False
        scene.render.engine = "CYCLES"
        scene.render.filepath = str(frames_dir / "rgba_####")
        bpy.ops.render.render(animation=True)
        t1 = time.time()
        self.floor.hide_render = True
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.filepath = str(frames_dir / "matte_####")
        bpy.ops.render.render(animation=True)
        scene.render.engine = "CYCLES"
        return {"cycles": round(t1 - t0, 1), "matte": round(time.time() - t1, 1)}

    def _stills(self, frames_dir: Path, times: np.ndarray, floors: np.ndarray) -> dict:
        """Render output frame k at action time times[k] as a still (a sub-frame: `animation=True` only renders
        whole frames), Cycles with the shadow on the floor at floors[k], then the Workbench matte. Persistent data
        keeps Cycles' kernels and the static scene between the stills like between animation frames; the engines
        do not alternate per frame, which would drop it."""
        scene = self.scene
        frames_dir.mkdir(parents=True, exist_ok=True)
        seconds = {}
        for engine, prefix, key in (("CYCLES", "rgba", "cycles"), ("BLENDER_WORKBENCH", "matte", "matte")):
            t0 = time.time()
            self.floor.hide_render = engine != "CYCLES"
            scene.render.engine = engine
            for k, t in enumerate(times):
                self.floor.location = (0.0, 0.0, float(floors[k]))
                self._at(float(t))
                scene.render.filepath = str(frames_dir / f"{prefix}_{k:04d}.png")
                bpy.ops.render.render(write_still=True)
            seconds[key] = round(time.time() - t0, 1)
        scene.render.engine = "CYCLES"
        return seconds

    # ------------------------------------------------------------------------------------------ clip
    def _play(self, clip: str) -> int:
        self._clear_pose()
        action = bpy.data.actions.get(clip)
        if action is None:
            raise KeyError(f"no action {clip!r} in the moveset")
        ad = self.arm.animation_data
        ad.action = action
        if action.slots:
            ad.action_slot = action.slots[0]
        start, end = (float(v) for v in action.frame_range)
        if abs(start) > 1e-3:
            raise ValueError(f"{clip}: action starts at frame {start}, expected 0 (frame i = clip frame i)")
        return int(round(end)) + 1

    def clip(self, spec: dict) -> dict:
        t0 = time.time()
        n_src = self._play(spec["clip"])
        source_contact = int(np.clip(spec["contact"], 0, n_src - 1))
        times, contact = frame_times(n_src, source_contact, self.fps, self.out_fps)
        if spec.get("frames"):          # look-dev: only the first k output frames
            times = times[:int(spec["frames"])]
            contact = min(contact, len(times) - 1)
        n = len(times)
        whole = np.array_equal(times, np.arange(n))      # the clip's own rate: one animation render, as before
        if self.racket is not None:
            self.racket.hide_render = False
        pts, feet, lowest = [], [], []
        objs = [*self.body, *([self.racket] if self.racket is not None else [])]
        for t in times:                 # every output frame's pose: framing, floor and ground track per output frame
            self._at(float(t))
            v = self._vertices(objs)
            pts.append(v)
            lowest.append(float(v[:, 2].min()))
            feet.append([self.head(j) for j in FEET])
        self._at(float(times[contact]))
        yaw = self.facing()
        floors = floor_track(np.asarray(lowest))
        ground = np.asarray(feet)
        ground[:, :, 2] = floors[:, None]
        allpts = np.concatenate([*pts, ground_patch(ground)])
        lo, hi = allpts.min(axis=0), allpts.max(axis=0)
        target = (lo + hi) / 2
        distance = still.DISTANCE_PER_SIZE * float(np.max(hi - lo))
        cam = still.camera_matrix(target, yaw, self.azimuth, self.elevation, distance)
        frame = fit_square(still.project(allpts, cam), self.size, still.MARGINS["full"])
        self._setup_camera(frame, cam, distance)
        self._place_lights(cam, target)
        self.floor.animation_data_clear()
        self.floor.location = (0.0, 0.0, 0.0)
        t1 = time.time()
        if whole:
            self._keys(self.floor, "location", 2, enumerate(floors), "CONSTANT")
            seconds = self._animation(Path(spec["frames_dir"]), n)
        else:
            seconds = self._stills(Path(spec["frames_dir"]), times, floors)
        self.floor.animation_data_clear()

        track = still.to_pixels(still.project(ground.reshape(-1, 3), cam), frame).reshape(n, len(FEET), 2)
        track = track.mean(axis=1).round(1)
        lowest = np.asarray(lowest)
        return {
            "n_frames": n, "contact_frame": contact, "size": [frame["width"], frame["height"]], "fps": self.out_fps,
            "source": {"fps": self.fps, "n_frames": n_src, "contact_frame": source_contact,
                       "times": None if whole else times.round(4).tolist()},
            "px_per_m": round(frame["f_px"] / distance, 3),
            "floor_z": round(float(floors[contact]), 4), "floor_z_track": floors.round(4).tolist(),
            "air_m": round(float(np.max(lowest - floors)), 4),   # highest she gets off the floor
            "ground_y_px": float(track[contact, 1]), "ground_xy_px": track[contact].tolist(),
            "ground_track_px": track.tolist(), "ground_box_px": ground_box(ground, [cam], frame),
            "framed_bbox_px": _bbox_px(still.to_pixels(still.project(allpts, cam), frame)),
            "camera": camera_row(frame, cam, target, yaw, self.azimuth, self.elevation, distance),
            "racket": self.racket is not None,
            "seconds": {"setup": round(t1 - t0, 1), **seconds},
        }

    # ------------------------------------------------------------------------------------- turntable
    def turntable(self, spec: dict) -> dict:
        t0 = time.time()
        n = int(spec["n_frames"])
        self.pose(None, None)
        if self.racket is not None:
            self.racket.hide_render = True
        pts, floor_z, feet = self.framed_points("full", False)
        yaw = self.facing()
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        target = (lo + hi) / 2
        distance = still.DISTANCE_PER_SIZE * float(np.max(hi - lo))
        azimuths = orbit_azimuths(n, float(spec.get("start_azimuth", 0.0)))
        cams = [still.camera_matrix(target, yaw, a, self.elevation, distance) for a in azimuths]
        frame = fit_square(np.concatenate([still.project(pts, c) for c in cams]), self.size, still.MARGINS["full"])
        self._setup_camera(frame, cams[0], distance)
        self._place_lights(cams[0], target)
        self.floor.animation_data_clear()
        self.floor.location = (0.0, 0.0, floor_z)
        # camera + lights ride an empty on the vertical axis through her centre: one linear turn over n frames
        pivot = bpy.data.objects.new("TurntablePivot", None)
        self.scene.collection.objects.link(pivot)
        pivot.location = (float(target[0]), float(target[1]), 0.0)
        bpy.context.view_layer.update()
        P_inv = np.linalg.inv(np.array(pivot.matrix_world))
        for obj in (self.cam, *self.lights.values()):
            world = np.array(obj.matrix_world)
            obj.parent = pivot
            obj.matrix_parent_inverse = Matrix.Identity(4)
            obj.matrix_basis = Matrix(P_inv @ world)
        self._keys(pivot, "rotation_euler", 2, [(0, 0.0), (n, 2 * math.pi)], "LINEAR")
        self.scene.frame_set(0)
        t1 = time.time()
        seconds = self._animation(Path(spec["frames_dir"]), n)

        ground = still.to_pixels(still.project(feet, cams[0]), frame).mean(axis=0)
        return {
            "n_frames": n, "contact_frame": 0, "size": [frame["width"], frame["height"]],
            "px_per_m": round(frame["f_px"] / distance, 3), "floor_z": round(floor_z, 4),
            "ground_y_px": round(float(ground[1]), 1), "ground_xy_px": ground.round(1).tolist(),
            "ground_box_px": ground_box(feet, cams, frame),
            "framed_bbox_px": _bbox_px(np.concatenate([still.to_pixels(still.project(pts, c), frame) for c in cams])),
            "azimuths": [round(float(a), 2) for a in azimuths],
            "camera": camera_row(frame, cams[0], target, yaw, float(azimuths[0]), self.elevation, distance),
            "racket": False,
            "seconds": {"setup": round(t1 - t0, 1), **seconds},
        }


def main(argv: list[str]) -> None:
    job = json.loads(Path(argv[0]).read_text())
    t0 = time.time()
    studio = MotionStudio(job)
    items = job["items"]
    _log(f"cycles on {studio.device['type']} {studio.device['devices']}, {studio.scene.cycles.samples} samples, "
         f"{studio.size[0]}x{studio.size[1]} at {studio.out_fps:g} fps, {len(items)} videos")
    for i, item in enumerate(items):
        res = studio.turntable(item) if item["kind"] == "turntable" else studio.clip(item)
        res |= {"fingerprint": item.get("fingerprint"), "render_key": item.get("render_key"),
                "blender": bpy.app.version_string, "device": studio.device,
                "samples": studio.scene.cycles.samples}
        Path(item["report"]).write_text(json.dumps(res, indent=2))
        s = res["seconds"]
        _log(f"[{i + 1}/{len(items)}] {item['name']}: {res['n_frames']} frames, {s['cycles']:.0f}s cycles "
             f"({s['cycles'] / res['n_frames']:.2f}s/frame), {s['matte']:.0f}s matte, {s['setup']:.1f}s setup, "
             f"facing {res['camera']['facing_deg']:.0f} deg, {res['px_per_m']:.0f} px/m")
        _log(f"done {item['name']}")
    _log(f"{len(items)} videos rendered in {time.time() - t0:.0f}s")


if __name__ == "__main__" and bpy is not None:
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
