"""Procedural low-poly tennis racket (runs inside Blender; bpy / bmesh / numpy only).

Local frame: origin at the butt cap, +Y along the shaft to the tip, +Z = string-face normal, +X across the
head. 68.5 cm long, 33 x 26 cm oval head, ~3.2 cm octagonal grip, open throat; red head and throat, black
shaft and grip, semi-transparent string bed (a small packed grid texture, so it survives glTF/FBX export).
"""

from __future__ import annotations

import math

import bmesh
import bpy
import numpy as np

LENGTH = 0.685
HEAD_LENGTH, HEAD_WIDTH = 0.33, 0.26
BEAM_WIDTH, BEAM_DEPTH = 0.012, 0.023    # frame profile: in the string plane x across it
HANDLE_LENGTH, GRIP = 0.19, 0.032
BUTT_LENGTH, BUTT = 0.012, 0.037
SHAFT_WIDTH, SHAFT_DEPTH = 0.020, 0.021
THROAT_Y = 0.255                          # where the shaft splits into the two throat arms
THROAT_ANGLE = math.radians(42)           # the arms meet the head this far round from its bottom
SEGMENTS = 36
MAINS, CROSSES = 16, 19
RED, BLACK = (0.62, 0.02, 0.03, 1.0), (0.02, 0.02, 0.025, 1.0)
STRING_WHITE = (0.96, 0.96, 0.92)          # linear RGB, like every colour here
FRAME, GRIP_MAT, STRINGS = 0, 1, 2


class _Builder:
    def __init__(self) -> None:
        self.verts: list[tuple[float, float, float]] = []
        self.faces: list[tuple[int, ...]] = []
        self.mats: list[int] = []

    def vert(self, x: float, y: float, z: float) -> int:
        self.verts.append((x, y, z))
        return len(self.verts) - 1

    def face(self, idx, mat: int) -> None:
        self.faces.append(tuple(idx))
        self.mats.append(mat)

    def profile(self, x: float, y: float, nx: float, ny: float, w: float, d: float) -> list[int]:
        """Rectangular section centred on (x, y, 0): half-width w along the in-plane normal, half-depth d."""
        return [self.vert(x + nx * sw, y + ny * sw, sz) for sw, sz in ((w, d), (-w, d), (-w, -d), (w, -d))]

    def skin(self, sections: list[list[int]], mat: int, closed: bool = False, caps: bool = True) -> None:
        n = len(sections)
        for i in range(n if closed else n - 1):
            a, b = sections[i], sections[(i + 1) % n]
            k = len(a)
            for j in range(k):
                self.face((a[j], a[(j + 1) % k], b[(j + 1) % k], b[j]), mat)
        if caps and not closed:
            self.face(sections[0][::-1], mat)
            self.face(sections[-1], mat)

    def bar(self, p0, p1, w: float, d: float, mat: int) -> None:
        """Straight rectangular beam in the string plane from p0 to p1."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        L = math.hypot(dx, dy)
        nx, ny = dy / L, -dx / L
        self.skin([self.profile(p[0], p[1], nx, ny, w, d) for p in (p0, p1)], mat)

    def octagon_prism(self, y0: float, y1: float, across: float, mat: int) -> None:
        r = across / 2 / math.cos(math.pi / 8)
        rings = [[self.vert(r * math.cos(a), y, r * math.sin(a))
                  for a in (math.pi / 8 + k * math.pi / 4 for k in range(8))] for y in (y0, y1)]
        self.skin(rings, mat)


def _head_point(phi: float, inset: float = 0.0) -> tuple[float, float, float, float]:
    """Point on the head's centre line (phi = 0 at the throat) and its outward in-plane normal."""
    ax, ay = HEAD_WIDTH / 2 - BEAM_WIDTH / 2 - inset, HEAD_LENGTH / 2 - BEAM_WIDTH / 2 - inset
    yc = LENGTH - HEAD_LENGTH / 2
    nx, ny = math.sin(phi) / ax, -math.cos(phi) / ay
    n = math.hypot(nx, ny)
    return ax * math.sin(phi), yc - ay * math.cos(phi), nx / n, ny / n


def _geometry() -> tuple[_Builder, list[int]]:
    g = _Builder()
    w, d = BEAM_WIDTH / 2, BEAM_DEPTH / 2
    head = []
    for i in range(SEGMENTS):
        x, y, nx, ny = _head_point(2 * math.pi * i / SEGMENTS)
        head.append(g.profile(x, y, nx, ny, w, d))
    g.skin(head, FRAME, closed=True)
    for side in (-1, 1):
        x, y, nx, ny = _head_point(side * THROAT_ANGLE)
        g.bar((0.0, THROAT_Y), (x - nx * w, y - ny * w), w * 1.1, d * 0.95, FRAME)
    g.bar((0.0, HANDLE_LENGTH - 0.005), (0.0, THROAT_Y + 0.004), SHAFT_WIDTH / 2, SHAFT_DEPTH / 2, GRIP_MAT)
    g.octagon_prism(BUTT_LENGTH, HANDLE_LENGTH, GRIP, GRIP_MAT)
    g.octagon_prism(0.0, BUTT_LENGTH, BUTT, FRAME)

    yc = LENGTH - HEAD_LENGTH / 2
    centre = g.vert(0.0, yc, 0.0)
    rim = []
    for i in range(SEGMENTS):
        x, y, _, _ = _head_point(2 * math.pi * i / SEGMENTS, inset=w * 0.8)
        rim.append(g.vert(x, y, 0.0))
    string_faces = []
    for i in range(SEGMENTS):
        string_faces.append(len(g.faces))
        g.face((centre, rim[i], rim[(i + 1) % SEGMENTS]), STRINGS)
    return g, string_faces


def _strings_image(size: int = 256, color=STRING_WHITE):
    u = (np.arange(size) + 0.5) / size
    line = 2.2 / size
    mains = ((u * MAINS) % 1.0) < line * MAINS
    crosses = ((u * CROSSES) % 1.0) < line * CROSSES
    on = mains[None, :] | crosses[:, None]
    px = np.empty((size, size, 4), np.float32)
    px[...] = (1.0, 1.0, 1.0, 0.10)
    px[on] = (*color[:3], 0.9)
    img = bpy.data.images.new("RacketStrings", size, size, alpha=True)
    img.pixels.foreach_set(px.ravel())
    img.pack()
    return img


def _material(name: str, color, roughness: float, image=None):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_backface_culling = False
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if image is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        mat.node_tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.surface_render_method = "BLENDED"
    return mat


def build(name: str = "Racket", collection=None, colors: dict | None = None):
    """Create the racket object (one mesh, three materials) in its local frame; returns the object.
    `colors` (linear RGB) may override "frame", "grip" and "strings"."""
    c = colors or {}
    frame, grip = (*c.get("frame", RED)[:3], 1.0), (*c.get("grip", BLACK)[:3], 1.0)
    g, string_faces = _geometry()
    me = bpy.data.meshes.new(name)
    me.from_pydata(g.verts, [], g.faces)
    me.polygons.foreach_set("material_index", g.mats)
    for mat in (_material("Racket_Frame", frame, 0.35), _material("Racket_Grip", grip, 0.7),
                _material("Racket_Strings", (1, 1, 1, 1), 0.6, _strings_image(color=c.get("strings", STRING_WHITE)))):
        me.materials.append(mat)

    uv = me.uv_layers.new(name="UVMap")
    sx, sy = HEAD_WIDTH / 2, HEAD_LENGTH / 2
    yc = LENGTH - HEAD_LENGTH / 2
    co = np.array(g.verts)
    for f in string_faces:
        poly = me.polygons[f]
        for li in poly.loop_indices:
            x, y, _ = co[me.loops[li].vertex_index]
            uv.data[li].uv = (0.5 + x / (2 * sx), 0.5 + (y - yc) / (2 * sy))

    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=[f for f in bm.faces if f.material_index != STRINGS])
    bm.to_mesh(me)
    bm.free()
    me.validate()
    me.update()
    obj = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(obj)
    return obj


def triangle_count(obj) -> int:
    return sum(len(p.vertices) - 2 for p in obj.data.polygons)
