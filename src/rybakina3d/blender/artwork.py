"""Character-sheet figures from the moveset GLB, rendered with Cycles (runs INSIDE Blender).

    blender -b --factory-startup --python-exit-code 1 --python artwork.py -- job.json

`rybakina3d.artwork.render` writes the job. Only bpy / mathutils / numpy are used; the numpy half (facing,
camera placement, framing) imports without Blender for the unit tests.

Per figure: the clip's action at its frame, or the idle pose (rest pose, arms brought down beside the body,
elbows slightly bent, no racket). Her facing in that frame is the way her chest faces: +Z x the left -> right
shoulder line, blended CHEST_WEIGHT : 1 with the hip line, which steadies it when a stroke coils the shoulders
far past the hips (from the hips alone a serve's trophy is seen from behind). The camera sits at the slot's
azimuth (0 = in front, 90 = her left) and elevation relative to that, at a fixed relative distance
(DISTANCE_PER_SIZE: a long-lens look), and the lens + shift are solved so the figure's projected vertices
fill the frame with a small margin:
  full  every vertex (racket included) plus a ground patch around the feet for the contact shadow
  bust  head, shoulders, upper chest: vertices above BUST_DROP_M below the neck base
  head  head + neck: vertices above the neck base, within HEAD_RADIUS_M of the neck -> head axis
Lighting follows the camera, like a studio turnaround: a big warm key from the upper front-left, a cool low
fill from the right, a rim from behind and a warm ambient world (see LIGHTS). A shadow-catcher floor at the
lowest vertex (or at z=0 when she is in the air) takes the soft contact shadow of a high light that only
reaches the floor, on a transparent film; Cycles on the GPU (OptiX > CUDA > Metal > HIP, else CPU), OIDN
denoising, AgX view transform.
Each figure is written twice: <name>.png (Cycles, RGBA with the shadow) and <name>_matte.png (Workbench,
flat dark figure, alpha = coverage without the shadow: silhouettes and pose checks).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Matrix
except ImportError:  # imported outside Blender (unit tests of the numpy part)
    bpy = None
    Matrix = None

UP = np.array([0.0, 0.0, 1.0])
DISTANCE_PER_SIZE = 3.8          # camera distance / figure size: ~85 mm portrait-lens perspective
BUST_DROP_M = 0.20               # bust: from the neck base down to the upper chest
HEAD_DROP_M = 0.04               # head: a little of the neck below its base
HEAD_RADIUS_M = 0.16             # head: vertices this close to the neck -> head axis (hair, visor brim)
GROUND_PATCH_M = 0.15            # full: ground around each foot that must fit in the frame (contact shadow)
GROUNDED_M = 0.10                # lowest vertex within this of z=0: the floor moves up/down to touch her
# margins as a fraction of the framed height: (top, sides, bottom)
MARGINS = {"full": (0.035, 0.05, 0.03), "bust": (0.06, 0.07, 0.0), "head": (0.08, 0.09, 0.0)}
# idle stance: arms hang this far out from vertical and this far forward, elbows bent a little
IDLE_ARM_OUT_DEG, IDLE_ARM_FORWARD_DEG, IDLE_ELBOW_DEG = 11.0, 4.0, 14.0
# Lights follow the camera (azimuth deg towards image-left, elevation deg). Light linking splits the rig: the
# figure lights (a warm key low enough to light the eyes under the visor, a cool fill, a rim) only reach her,
# and a high soft "floor" light only reaches the shadow catcher, so the contact shadow stays short and under
# her whatever the key does. Only key and floor cast shadows.
LIGHTS = {
    "key": {"az": 45.0, "el": 38.0, "size": 3.0, "power": 1100.0, "rgb": (1.0, 0.89, 0.76), "shadow": True,
            "lights": "figure"},
    "fill": {"az": -65.0, "el": 10.0, "size": 4.0, "power": 300.0, "rgb": (0.86, 0.92, 1.0), "shadow": False,
             "lights": "figure"},
    "rim": {"az": -150.0, "el": 35.0, "size": 1.5, "power": 800.0, "rgb": (1.0, 0.95, 0.88), "shadow": False,
            "lights": "figure"},
    "floor": {"az": 30.0, "el": 72.0, "size": 5.0, "power": 300.0, "rgb": (1.0, 0.95, 0.9), "shadow": True,
              "lights": "floor"},
}
LIGHT_DISTANCE_M = 6.0
WORLD = ((0.93, 0.89, 0.83), 0.3)    # warm paper-coloured ambient
MATTE_RGB = (0.10, 0.10, 0.11)
CHEST_WEIGHT = 3.0               # facing: shoulder line : hip line (see module docstring)
EXPOSURE, LOOK = -0.75, "AgX - Medium High Contrast"
WELD_M = 1e-5                    # smooth_skin: vertices closer than this are one
SMOOTH_SKIN = (0.5, 10)          # smooth_skin: Corrective Smooth factor, iterations


# ------------------------------------------------------------------------------------------------ numpy
def facing_yaw(left_hip: np.ndarray, right_hip: np.ndarray) -> float:
    """Heading (rad, from +X towards +Y) of the way she faces: +Z x (left -> right hip)."""
    r = np.asarray(right_hip, np.float64) - np.asarray(left_hip, np.float64)
    return math.atan2(r[0], -r[1])


def camera_matrix(target: np.ndarray, yaw: float, azimuth_deg: float, elevation_deg: float,
                  distance: float) -> np.ndarray:
    """World matrix of a camera looking at `target` from `azimuth_deg` around her facing (90 = her left)."""
    a, e = yaw + math.radians(azimuth_deg), math.radians(elevation_deg)
    back = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])  # target -> camera
    up = UP - np.dot(UP, back) * back
    up /= np.linalg.norm(up)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = np.cross(up, back), up, back
    M[:3, 3] = np.asarray(target, np.float64) + back * distance
    return M


def project(points: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Normalised image coordinates (x right, y up; focal length 1) of world points in front of `cam`."""
    pc = (np.asarray(points, np.float64) - cam[:3, 3]) @ cam[:3, :3]
    depth = np.maximum(-pc[:, 2], 1e-6)
    return pc[:, :2] / depth[:, None]


def fit_frame(uv: np.ndarray, height_px: int, margins: tuple[float, float, float]) -> dict:
    """Lens, shift and resolution that frame `uv` (normalised coords) with margins, `height_px` tall."""
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    span = v1 - v0
    top, side, bottom = (m * span for m in margins)
    u0, u1, v0, v1 = u0 - side, u1 + side, v0 - bottom, v1 + top
    f_px = height_px / (v1 - v0)
    width = int(round((u1 - u0) * f_px / 2)) * 2
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    big = max(width, height_px)
    return {"width": width, "height": int(height_px), "f_px": f_px, "lens_mm": 24.0 / (v1 - v0),
            "sensor_height_mm": 24.0, "shift_x": cu * f_px / big, "shift_y": cv * f_px / big, "centre": (cu, cv)}


def moves_with(obj, arm) -> bool:
    """Whether a mesh object is part of the figure: skinned (an armature modifier), or parented to the armature
    or one of its bones, directly or through other objects (a rigid visor, hair piece or earring on the head bone
    moves with her but has no modifier). Duck-typed on .type / .modifiers / .parent, so it tests without bpy."""
    if obj.type != "MESH":
        return False
    if any(m.type == "ARMATURE" for m in obj.modifiers):
        return True
    parent = obj.parent
    while parent is not None:
        if parent == arm:
            return True
        parent = parent.parent
    return False


def to_pixels(uv: np.ndarray, frame: dict) -> np.ndarray:
    cu, cv = frame["centre"]
    x = frame["width"] / 2 + (uv[:, 0] - cu) * frame["f_px"]
    y = frame["height"] / 2 - (uv[:, 1] - cv) * frame["f_px"]
    return np.stack([x, y], axis=1)


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smallest rotation (3x3) taking direction u onto v."""
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    axis, c = np.cross(u, v), float(np.dot(u, v))
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def hanging_direction(arm_dir: np.ndarray, forward: np.ndarray, out_deg: float, forward_deg: float) -> np.ndarray:
    """Where a relaxed arm points: down, `out_deg` out to its own side, `forward_deg` forward."""
    out = np.array([arm_dir[0], arm_dir[1], 0.0])
    out -= np.dot(out, forward) * forward
    out /= max(np.linalg.norm(out), 1e-9)
    d = -UP * math.cos(math.radians(out_deg)) + out * math.sin(math.radians(out_deg))
    d = d + forward * math.tan(math.radians(forward_deg))
    return d / np.linalg.norm(d)


# ============================================================================== Blender side (bpy below)
def _log(msg: str) -> None:
    print(f"[r3d] {msg}", flush=True)


class Studio:
    """The imported moveset, the light rig, the shadow-catcher floor and the camera."""

    def __init__(self, job: dict):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = self.scene = bpy.context.scene
        fps = float(job.get("fps", 25.0))
        scene.render.fps = int(round(fps))
        scene.render.fps_base = scene.render.fps / fps   # imported keys land on whole frames
        bpy.ops.import_scene.gltf(filepath=job["glb"])
        self.arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        self.racket = next((o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Racket")), None)
        self.body = [o for o in bpy.data.objects if o != self.racket and moves_with(o, self.arm)]
        for o in list(bpy.data.objects):
            # whatever else is a mesh does not move with her: the importer's bone-shape icosphere (a unit sphere at
            # the origin); lights a character GLB may carry would change the sheet's lighting
            if (o.type == "MESH" and o != self.racket and o not in self.body) or o.type == "LIGHT":
                bpy.data.objects.remove(o)
        self.bones = job["bones"]
        ad = self.arm.animation_data or self.arm.animation_data_create()
        for track in ad.nla_tracks:  # the importer stacks every clip in the NLA; one action at a time here
            track.mute = True
        settings = job.get("settings", {})   # samples, plus look-dev overrides of the constants below
        self.chest_weight = float(settings.get("chest_weight", CHEST_WEIGHT))
        if settings.get("smooth_skin"):
            self._smooth_skin()
        self.device = self._render_settings(settings)
        self._world(float(settings.get("world", WORLD[1])))
        self.floor = self._floor()
        receivers = {"figure": self._collection("SheetFigure", [*self.body, self.racket]),
                     "floor": self._collection("SheetFloorOnly", [self.floor])}
        self.lights = {name: self._light(name, spec, receivers[spec["lights"]],
                                         float(settings.get("lights", {}).get(name, 1.0)))
                       for name, spec in LIGHTS.items()}
        self.cam = bpy.data.objects.new("SheetCamera", bpy.data.cameras.new("SheetCamera"))
        scene.collection.objects.link(self.cam)
        scene.camera = self.cam

    # ---------------------------------------------------------------------------------------- setup
    def _smooth_skin(self) -> None:
        """Softer joints for close renders: dual-quaternion skinning (no volume lost where a knee or elbow bends)
        and a Corrective Smooth over the deformation only (rest shape from the original coordinates, so modelled
        detail stays). Coincident vertices are welded first: smoothing a mesh whose triangles are separate
        tears it into spikes."""
        import bmesh

        for o in self.body:
            bm = bmesh.new()
            bm.from_mesh(o.data)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD_M)
            bm.to_mesh(o.data)
            bm.free()
            for m in o.modifiers:
                if m.type == "ARMATURE":
                    m.use_deform_preserve_volume = True
            cs = o.modifiers.new("SmoothSkin", "CORRECTIVE_SMOOTH")
            cs.factor, cs.iterations = SMOOTH_SKIN
            cs.smooth_type, cs.rest_source = "LENGTH_WEIGHTED", "ORCO"

    def _render_settings(self, settings: dict) -> dict:
        scene = self.scene
        scene.render.engine = "CYCLES"
        device = _enable_gpu()
        scene.cycles.device = "GPU" if device["type"] != "CPU" else "CPU"
        scene.cycles.samples = int(settings.get("samples", 128))
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
        if hasattr(scene.cycles, "denoising_use_gpu"):
            scene.cycles.denoising_use_gpu = device["type"] != "CPU"
        scene.cycles.max_bounces = 8
        scene.render.film_transparent = True
        scene.render.resolution_percentage = 100
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = settings.get("look", LOOK)
        scene.view_settings.exposure = float(settings.get("exposure", EXPOSURE))
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.color_depth = "8"
        display = scene.display
        display.render_aa = "16"
        display.shading.light = "FLAT"
        display.shading.color_type = "SINGLE"
        display.shading.single_color = MATTE_RGB
        display.shading.show_backface_culling = False
        return device

    def _world(self, strength: float) -> None:
        world = bpy.data.worlds.new("SheetWorld")
        self.scene.world = world
        bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
        bg.inputs["Color"].default_value = (*WORLD[0], 1.0)
        bg.inputs["Strength"].default_value = strength

    @staticmethod
    def _collection(name: str, objs: list):
        coll = bpy.data.collections.new(name)  # a light-linking set, not part of the scene tree
        for o in objs:
            if o is not None:
                coll.objects.link(o)
        return coll

    def _light(self, name: str, spec: dict, receivers, scale: float):
        data = bpy.data.lights.new(f"Sheet_{name}", "AREA")
        data.use_shadow = spec["shadow"]
        data.shape = "DISK"
        data.size = spec["size"]
        data.energy = spec["power"] * scale
        data.color = spec["rgb"]
        obj = bpy.data.objects.new(f"Sheet_{name}", data)
        self.scene.collection.objects.link(obj)
        obj.light_linking.receiver_collection = receivers
        return obj

    def _floor(self):
        me = bpy.data.meshes.new("SheetFloor")
        s = 40.0
        me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
        obj = bpy.data.objects.new("SheetFloor", me)
        obj.is_shadow_catcher = True
        self.scene.collection.objects.link(obj)
        return obj

    # ----------------------------------------------------------------------------------------- pose
    def _clear_pose(self) -> None:
        ad = self.arm.animation_data
        ad.action = None
        for pb in self.arm.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()

    def pose(self, clip: str | None, frame: int | None) -> None:
        self._clear_pose()
        if clip is None:
            self.scene.frame_set(0)
            self._idle()
            return
        action = bpy.data.actions.get(clip)
        if action is None:
            raise KeyError(f"no action {clip!r} in the moveset")
        ad = self.arm.animation_data
        ad.action = action
        if action.slots:
            ad.action_slot = action.slots[0]
        self.scene.frame_set(int(frame))
        bpy.context.view_layer.update()

    def head(self, joint: str) -> np.ndarray:
        """World position of the head of the bone mapped to an SMPL joint."""
        pb = self.arm.pose.bones[self.bones[joint]]
        return (np.array(self.arm.matrix_world) @ np.append(np.array(pb.head), 1.0))[:3]

    def facing(self) -> float:
        def line(side_l: str, side_r: str) -> np.ndarray:
            r = self.head(side_r) - self.head(side_l)
            r[2] = 0
            return r / max(np.linalg.norm(r), 1e-9)

        r = line("left_hip", "right_hip") + self.chest_weight * line("left_shoulder", "right_shoulder")
        return facing_yaw(np.zeros(3), r)

    def _idle(self) -> None:
        """Rest pose with the arms hanging relaxed beside the body (the rig's rest is an A-pose)."""
        yaw = self.facing()
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        W = np.array(self.arm.matrix_world)
        Wr = W[:3, :3] / np.linalg.norm(W[:3, :3], axis=0)
        for side in ("left", "right"):
            upper, lower, hand = (f"{side}_{j}" for j in ("shoulder", "elbow", "wrist"))
            d = self.head(lower) - self.head(upper)
            R = rotation_between(d, hanging_direction(d, forward, IDLE_ARM_OUT_DEG, IDLE_ARM_FORWARD_DEG))
            self._rotate_bone(upper, Wr.T @ R @ Wr)
            d = self.head(hand) - self.head(lower)
            bend = forward - np.dot(forward, d / np.linalg.norm(d)) * d / np.linalg.norm(d)
            t = math.radians(IDLE_ELBOW_DEG)
            target = d / np.linalg.norm(d) * math.cos(t) + bend / np.linalg.norm(bend) * math.sin(t)
            self._rotate_bone(lower, Wr.T @ rotation_between(d, target) @ Wr)

    def _rotate_bone(self, joint: str, R_arm: np.ndarray) -> None:
        """Rotate a pose bone about its head by R (armature space), children following."""
        pb = self.arm.pose.bones[self.bones[joint]]
        M = np.array(pb.matrix)
        h = M[:3, 3].copy()
        N = np.eye(4)
        N[:3, :3] = R_arm @ M[:3, :3]
        N[:3, 3] = h
        pb.matrix = Matrix(N)
        bpy.context.view_layer.update()

    # ------------------------------------------------------------------------------------- geometry
    def _vertices(self, objs) -> np.ndarray:
        dg = bpy.context.evaluated_depsgraph_get()
        out = []
        for o in objs:
            ev = o.evaluated_get(dg)
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3, np.float64)
            me.vertices.foreach_get("co", co)
            W = np.array(ev.matrix_world)
            out.append(co.reshape(-1, 3) @ W[:3, :3].T + W[:3, 3])
            ev.to_mesh_clear()
        return np.concatenate(out) if out else np.zeros((0, 3))

    def framed_points(self, framing: str, show_racket: bool) -> tuple[np.ndarray, float, np.ndarray]:
        """(points that must fit in the frame, floor height, feet ground points) for this framing."""
        body = self._vertices(self.body)
        racket = self._vertices([self.racket]) if (self.racket is not None and show_racket) else np.zeros((0, 3))
        everything = np.concatenate([body, racket])
        lowest = float(everything[:, 2].min())
        floor_z = lowest if lowest < GROUNDED_M else 0.0
        feet = np.array([self.head(j) for j in ("left_ankle", "right_ankle", "left_foot", "right_foot")])
        feet[:, 2] = floor_z
        if framing == "full":
            g = GROUND_PATCH_M
            patch = np.concatenate([feet + [dx, dy, 0] for dx in (-g, g) for dy in (-g, g)])
            return np.concatenate([everything, patch]), floor_z, feet
        neck = self.head("neck")
        if framing == "bust":
            return body[body[:, 2] >= neck[2] - BUST_DROP_M], floor_z, feet
        if framing == "head":
            axis = self.head("head") - neck
            axis /= np.linalg.norm(axis)
            rel = body - neck
            along = rel @ axis
            radial = np.linalg.norm(rel - along[:, None] * axis, axis=1)
            return body[(along >= -HEAD_DROP_M) & (radial <= HEAD_RADIUS_M)], floor_z, feet
        raise ValueError(f"unknown framing {framing!r}")

    # --------------------------------------------------------------------------------------- render
    def _place_lights(self, cam: np.ndarray, target: np.ndarray) -> None:
        back, right = cam[:3, 2].copy(), cam[:3, 0].copy()  # levelled: the rig turns with the camera only
        back[2] = right[2] = 0
        back /= np.linalg.norm(back)
        right /= np.linalg.norm(right)
        for name, obj in self.lights.items():
            az, el = math.radians(LIGHTS[name]["az"]), math.radians(LIGHTS[name]["el"])
            d = math.cos(el) * (math.cos(az) * back - math.sin(az) * right) + math.sin(el) * UP
            z = d / np.linalg.norm(d)                    # an area light shines along its -Z
            x = np.cross(UP, z)
            x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0, 0])
            M = np.eye(4)
            M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, np.cross(z, x), z, target + z * LIGHT_DISTANCE_M
            obj.matrix_world = Matrix(M)

    def render(self, fig: dict) -> dict:
        t0 = time.time()
        self.pose(fig.get("clip"), fig.get("frame"))
        show_racket = bool(fig.get("racket", True))
        if self.racket is not None:
            self.racket.hide_render = not show_racket
        pts, floor_z, feet = self.framed_points(fig["framing"], show_racket)
        yaw = self.facing()
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        target = (lo + hi) / 2
        distance = DISTANCE_PER_SIZE * float(np.max(hi - lo))
        cam = camera_matrix(target, yaw, fig["azimuth"], fig["elevation"], distance)
        frame = fit_frame(project(pts, cam), int(fig["height"]), MARGINS[fig["framing"]])

        data = self.cam.data
        data.type = "PERSP"
        data.sensor_fit = "VERTICAL"
        data.sensor_height = frame["sensor_height_mm"]
        data.lens = frame["lens_mm"]
        data.shift_x, data.shift_y = frame["shift_x"], frame["shift_y"]
        data.clip_start, data.clip_end = 0.05, distance * 4 + 50
        self.cam.matrix_world = Matrix(cam)
        scene = self.scene
        scene.render.resolution_x, scene.render.resolution_y = frame["width"], frame["height"]
        self.floor.location = (0.0, 0.0, floor_z)
        self._place_lights(cam, target)

        Path(fig["png"]).parent.mkdir(parents=True, exist_ok=True)
        self.floor.hide_render = fig["framing"] != "full"
        scene.render.engine = "CYCLES"
        self._still(fig["png"])
        t1 = time.time()
        self.floor.hide_render = True
        scene.render.engine = "BLENDER_WORKBENCH"
        self._still(fig["matte"])
        scene.render.engine = "CYCLES"

        ground = to_pixels(project(feet, cam), frame).mean(axis=0).round(1).tolist()
        full = fig["framing"] == "full"   # close-ups have no floor in the picture
        box = to_pixels(project(pts, cam), frame)
        return {
            "size": [frame["width"], frame["height"]],
            "ground_y": ground[1] if full else None,
            "ground_xy": ground if full else None,
            "framed_bbox": [round(float(v), 1) for v in (*box.min(axis=0), *box.max(axis=0))],
            "floor_z": round(floor_z, 4),
            "camera": {"azimuth": fig["azimuth"], "elevation": fig["elevation"], "framing": fig["framing"],
                       "facing_deg": round(math.degrees(yaw), 2), "distance_m": round(distance, 3),
                       "lens_mm": round(frame["lens_mm"], 3), "sensor_height_mm": frame["sensor_height_mm"],
                       "shift": [round(frame["shift_x"], 5), round(frame["shift_y"], 5)],
                       "matrix_zup": cam.round(5).tolist()},
            "racket": show_racket and self.racket is not None,
            "seconds": {"cycles": round(t1 - t0, 1), "matte": round(time.time() - t1, 1)},
        }

    def _still(self, path: str) -> None:
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)


def _enable_gpu() -> dict:
    """Headless Blender starts on the CPU: switch Cycles to the best GPU backend present."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = kind
        except TypeError:  # backend not built into this Blender
            continue
        prefs.refresh_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            return {"type": kind, "devices": [d.name for d in gpus]}
    prefs.compute_device_type = "NONE"
    return {"type": "CPU", "devices": []}


def main(argv: list[str]) -> None:
    job = json.loads(Path(argv[0]).read_text())
    t0 = time.time()
    studio = Studio(job)
    _log(f"cycles on {studio.device['type']} {studio.device['devices']}, "
         f"{studio.scene.cycles.samples} samples, {len(job['figures'])} figures")
    figures = {}
    for i, fig in enumerate(job["figures"]):
        res = studio.render(fig)
        figures[fig["name"]] = res
        what = f"{fig['clip']} f{fig['frame']}" if fig.get("clip") else "idle"
        _log(f"[{i + 1}/{len(job['figures'])}] {fig['name']}: {what}, {res['size'][0]}x{res['size'][1]}, "
             f"facing {res['camera']['facing_deg']:.0f} deg, {res['seconds']['cycles']:.1f}s")
    report = {"blender": bpy.app.version_string, "device": studio.device, "samples": studio.scene.cycles.samples,
              "figures": figures, "seconds": round(time.time() - t0, 1)}
    Path(job["report"]).write_text(json.dumps(report, indent=2))


if __name__ == "__main__" and bpy is not None:
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
