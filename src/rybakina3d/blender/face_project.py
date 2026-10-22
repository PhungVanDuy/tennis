"""Face projection for an art-team character: render the rig's face, then bake an aligned image onto it
(runs INSIDE Blender).

    blender -b --factory-startup --python-exit-code 1 --python face_project.py -- job.json

`rybakina3d.character.face` writes the job and does the 2D half (landmarks, alignment, mask) between the two
modes:
  render  the rest-pose face seen from the front by an orthographic camera, unlit (Workbench, flat light, texture
          colour, Standard view transform), so the texture's own pixels are what the landmark detector sees;
          writes the PNG and the camera (location, ortho scale) to the job's "out" JSON
  bake    the same camera projects an image onto the mesh: a "proj" UV layer from each vertex's position in that
          view and a per-corner "facing" weight (normal towards the camera); every material's base colour
          becomes mix(base colour, projected image, mask x facing) and Cycles bakes it as emission into a new
          texture on the original UVs, which the driver then puts into the GLB in place of the base colour
The character faces -Y (Meshy's glTF front), so the camera looks along +Y from in front of the head.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:  # imported outside Blender
    bpy = None

HEAD_FRACTION = 0.14      # of the figure height, from the top: the head's vertices
FRAME_PAD = 1.25          # ortho scale over the head's height
FACING_FROM, FACING_TO = 0.25, 0.6   # cos(normal, towards camera): no projection below, full above


def _load(glb: str):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def _head_camera(meshes, size: int):
    pts = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
    top = max(p.z for p in pts)
    bottom = min(p.z for p in pts)
    head = [p for p in pts if p.z > top - HEAD_FRACTION * (top - bottom)]
    lo = Vector((min(p.x for p in head), min(p.y for p in head), min(p.z for p in head)))
    hi = Vector((max(p.x for p in head), max(p.y for p in head), max(p.z for p in head)))
    scene = bpy.context.scene
    cam = bpy.data.objects.new("FaceCam", bpy.data.cameras.new("FaceCam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = (hi.z - lo.z) * FRAME_PAD
    cam.location = Vector(((lo.x + hi.x) / 2, lo.y - 2.0, (lo.z + hi.z) / 2))
    cam.rotation_euler = Vector((0, 1, 0)).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    return cam, {"location": list(cam.location), "ortho_scale": cam.data.ortho_scale,
                 "head_lo": list(lo), "head_hi": list(hi)}


def render(job: dict) -> None:
    meshes = _load(job["glb"])
    _cam, info = _head_camera(meshes, job["size"])
    for mat in {m for o in meshes for m in o.data.materials if m}:
        tex, _bsdf = _base_texture(mat)
        if tex is not None:          # Workbench's texture colour shows the material's active image node
            mat.node_tree.nodes.active = tex
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"
    scene.display.shading.color_type = "TEXTURE"
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = True
    scene.render.filepath = job["png"]
    bpy.ops.render.render(write_still=True)
    Path(job["out"]).write_text(json.dumps(info, indent=1))
    print(f"[r3d] face render {job['png']}")


def _base_texture(mat):
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None or not bsdf.inputs["Base Color"].is_linked:
        return None, None
    node = bsdf.inputs["Base Color"].links[0].from_node
    return (node if node.type == "TEX_IMAGE" else None), bsdf


def bake(job: dict) -> None:
    meshes = _load(job["glb"])
    cam, _info = _head_camera(meshes, job["size"])
    scene = bpy.context.scene
    view = (cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))).normalized()
    art = bpy.data.images.load(job["image"])
    mask = bpy.data.images.load(job["mask"])
    mask.colorspace_settings.name = "Non-Color"
    target = None
    to_cam = np.array(cam.matrix_world.inverted())
    span = cam.data.ortho_scale                # square frame: the ortho scale spans both axes
    for obj in meshes:
        me = obj.data
        orig_uv = me.uv_layers.active.name
        proj = me.uv_layers.new(name="proj")
        facing = me.color_attributes.new("facing", "FLOAT_COLOR", "CORNER")
        world = np.array(obj.matrix_world)
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
        p = co @ to_cam[:3, :3].T + to_cam[:3, 3]  # camera space: x right, y up
        uv_v = np.stack([p[:, 0] / span + 0.5, p[:, 1] / span + 0.5], 1)
        vidx = np.empty(len(me.loops), np.int32)
        me.loops.foreach_get("vertex_index", vidx)
        proj.data.foreach_set("uv", uv_v[vidx].astype(np.float32).ravel())
        normals = np.empty(len(me.loops) * 3, np.float32)
        me.corner_normals.foreach_get("vector", normals)
        n = normals.reshape(-1, 3) @ np.linalg.inv(world[:3, :3])   # (M^-T n) as a row vector
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        f = np.clip((-(n @ np.array(view)) - FACING_FROM) / (FACING_TO - FACING_FROM), 0.0, 1.0)
        facing.data.foreach_set("color", np.repeat(f[:, None], 4, 1).astype(np.float32).ravel())
        me.uv_layers[orig_uv].active = True
        me.uv_layers[orig_uv].active_render = True
        for mat in me.materials:
            tex, bsdf = _base_texture(mat) if mat else (None, None)
            if tex is None:
                continue
            nt = mat.node_tree
            if target is None:
                w, h = tex.image.size
                scale = max(1, job.get("min_size", 0) // max(w, h))   # small maps are baked up to min_size
                target = bpy.data.images.new("BakedBaseColor", w * scale, h * scale, alpha=False)
                target.colorspace_settings.name = tex.image.colorspace_settings.name
            uv_proj = nt.nodes.new("ShaderNodeUVMap")
            uv_proj.uv_map = "proj"
            art_node, mask_node = nt.nodes.new("ShaderNodeTexImage"), nt.nodes.new("ShaderNodeTexImage")
            for node, img in ((art_node, art), (mask_node, mask)):
                node.image, node.extension = img, "CLIP"
                nt.links.new(uv_proj.outputs["UV"], node.inputs["Vector"])
            attr = nt.nodes.new("ShaderNodeAttribute")
            attr.attribute_name = "facing"
            weight = nt.nodes.new("ShaderNodeMath")
            weight.operation = "MULTIPLY"
            nt.links.new(mask_node.outputs["Color"], weight.inputs[0])
            nt.links.new(attr.outputs["Fac"], weight.inputs[1])
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            nt.links.new(weight.outputs["Value"], mix.inputs["Factor"])
            nt.links.new(tex.outputs["Color"], mix.inputs["A"])
            nt.links.new(art_node.outputs["Color"], mix.inputs["B"])
            emit = nt.nodes.new("ShaderNodeEmission")
            nt.links.new(mix.outputs["Result"], emit.inputs["Color"])
            out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
            bake_node = nt.nodes.new("ShaderNodeTexImage")
            bake_node.image = target
            nt.nodes.active = bake_node
    if target is None:
        raise RuntimeError("no material with an image-textured base colour")
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.bake_type = "EMIT"
    scene.render.bake.margin = 8
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.bake(type="EMIT")
    target.filepath_raw = job["png"]
    target.file_format = "PNG"
    target.save()
    print(f"[r3d] face bake {job['png']} {tuple(target.size)}")


def main(argv: list[str]) -> None:
    job = json.loads(Path(argv[0]).read_text())
    {"render": render, "bake": bake}[job["mode"]](job)


if __name__ == "__main__" and bpy is not None:
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
