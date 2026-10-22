"""Stage `character mesh`: turnaround views → textured quad mesh → humanoid rig, through fal.

Two generators run side by side on the same views so they can be compared on the review site:
  * Meshy 7.1 multi-image-to-3D with remesh (quad, game budget) + built-in humanoid auto-rig  (primary)
  * the A/B generator `models.mesh_alt`, rigged afterwards with Meshy's rigging endpoint: Tripo H3.1
    multiview-to-3D (quad, face limit), Hunyuan3D 3.1 Pro or Hitem3D Hi3D v3.0 (front/back/left/right views, PBR,
    high poly)
`character.mesh_primary` picks which rig becomes character/rigged.glb (default: Meshy when it has one); an empty
`models.mesh` skips Meshy's generator (its rigging endpoint still rigs the A/B mesh).
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..fal import client as _fal
from ..fal import download as _download
from ..fal import subscribe as _subscribe
from ..gemini import generate, image_part, text_part
from ..gltf import glb_stats
from ..util import log, read_json, write_json

VIEWS = ("front", "left", "back", "right")  # the order both multi-view generators expect


class Facing(BaseModel):
    nose_points: str  # "image_left" | "image_right" | "camera" | "away"


def _url(obj) -> str | None:
    return obj.get("url") if isinstance(obj, dict) else None


def ordered_views(cfg: Config) -> dict[str, Path]:
    """Name the side views by what they show (the image model does not always follow the requested order)."""
    views_dir = cfg.run.views_dir
    cache = views_dir / "facing.json"
    if cache.exists():
        return {k: views_dir / v for k, v in read_json(cache).items()}
    out = {"front": views_dir / "front.png", "back": views_dir / "back.png"}
    for name in ("left", "right"):  # positions 2 and 4 of the sheet
        res = generate(cfg.models.locate, [image_part(_jpeg(views_dir / f"{name}.png")),
                                           text_part("Side view of a standing character. Which way does her nose "
                                                     "point: image_left, image_right, camera or away?")], Facing)
        # facing image-right shows her RIGHT side to the camera
        shows = "right" if res.nose_points == "image_right" else "left"
        out[shows] = views_dir / f"{name}.png"
    if set(out) != set(VIEWS):  # both classified the same way: keep sheet order
        out = {n: views_dir / f"{n}.png" for n in VIEWS}
    write_json(cache, {k: v.name for k, v in out.items()})
    log("views: " + ", ".join(f"{k}={v.name}" for k, v in out.items()))
    return out


def _jpeg(path: Path) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _meshy(cfg: Config, urls: list[str], out: Path) -> dict:
    extra = {"geometry_resolution": cfg.character.geometry_resolution} if cfg.character.geometry_resolution else {}
    if cfg.character.refs_dir:  # an art-team variant: their own front render guides the texture
        extra["texture_image_url"] = urls[0]
    res = _subscribe(cfg.models.mesh, {
        **extra,
        "image_urls": urls,
        "should_remesh": True,
        "topology": "quad",
        "target_polycount": cfg.character.target_tris // 2,  # quads
        "symmetry_mode": "auto",
        "pose_mode": "a-pose",
        "should_texture": True,
        "enable_pbr": True,
        "enable_rigging": True,
        "rigging_height_meters": cfg.character.height_m,
        "enable_safety_checker": False,
    })
    files = {}
    for key in ("rigged_character_glb", "rigged_character_fbx", "model_glb", "thumbnail"):
        if _url(res.get(key)):
            ext = Path(res[key]["url"].split("?")[0]).suffix or ".bin"
            files[key] = _download(res[key]["url"], out / f"{key}{ext}").name
    for anim in res.get("basic_animations") or []:
        for k, v in (anim.items() if isinstance(anim, dict) else []):
            if _url(v):
                ext = Path(v["url"].split("?")[0]).suffix
                files[f"basic_{k}"] = _download(v["url"], out / f"basic_{k}{ext}").name
    return {"endpoint": cfg.models.mesh, "files": files, "rig_task_id": res.get("rig_task_id")}


def _blender_local() -> str:
    for cand in (os.environ.get("R3D_BLENDER_LOCAL"), "/Applications/Blender.app/Contents/MacOS/Blender"):
        if cand and Path(cand).exists():
            return cand
    return "blender"


def _fbx_to_glb(fbx: Path, glb: Path) -> Path:
    """Tripo returns quad meshes as FBX; the rigging endpoint wants GLB."""
    import subprocess

    expr = (
        "import bpy; bpy.ops.wm.read_factory_settings(use_empty=True); "
        f"bpy.ops.import_scene.fbx(filepath={str(fbx)!r}); "
        f"bpy.ops.export_scene.gltf(filepath={str(glb)!r}, export_format='GLB')"
    )
    subprocess.run([_blender_local(), "-b", "--factory-startup", "--python-expr", expr], check=True,
                   capture_output=True)
    return glb


def _rig(cfg: Config, files: dict, out: Path) -> None:
    """Meshy's humanoid auto-rig on files["model_glb"]; adds the rigged GLB/FBX to `files`."""
    glb_url = _fal().upload_file(str(out / files["model_glb"]))
    rig = _subscribe(cfg.models.rig, {"model_url": glb_url, "height_meters": cfg.character.height_m,
                                      "enable_safety_checker": False})
    for key in ("rigged_character_glb", "rigged_character_fbx"):
        if _url(rig.get(key)):
            ext = ".glb" if key.endswith("glb") else ".fbx"
            files[key] = _download(rig[key]["url"], out / f"rigged_character{ext}").name


HUNYUAN_FACES = 200_000   # inside Meshy rigging's limit; ~2x Meshy's budget for close renders
HI3D_FACES = 280_000      # Hi3D's own advice is 2M, but Meshy rigging takes ~300k faces at most


def _hi3d(cfg: Config, urls: list[str], out: Path, files: dict) -> dict:
    if "model_glb" not in files:
        front, left, back, right = urls  # VIEWS order
        res = _subscribe(cfg.models.mesh_alt, {
            "front_image_url": front, "back_image_url": back, "left_image_url": left, "right_image_url": right,
            "resolution": "2048quality", "enable_texture": True, "enable_pbr": True, "face_count": HI3D_FACES,
            "export_format": "glb", "enable_safety_checker": False,
        })
        files["model_glb"] = _download(res["model_mesh"]["url"], out / "model.glb").name
        if _url(res.get("thumbnail")):
            files["thumbnail"] = _download(res["thumbnail"]["url"], out / "thumbnail.png").name
    _rig(cfg, files, out)
    return {"endpoint": cfg.models.mesh_alt, "rig_endpoint": cfg.models.rig, "files": files}


def _hunyuan(cfg: Config, urls: list[str], out: Path, files: dict) -> dict:
    if "model_glb" not in files:
        front, left, back, right = urls  # VIEWS order
        res = _subscribe(cfg.models.mesh_alt, {
            "input_image_url": front, "back_image_url": back, "left_image_url": left, "right_image_url": right,
            "generate_type": "Normal", "enable_pbr": True, "face_count": HUNYUAN_FACES,
            "enable_safety_checker": False,
        })
        files["model_glb"] = _download(res["model_glb"]["url"], out / "model.glb").name
        if _url(res.get("thumbnail")):
            files["thumbnail"] = _download(res["thumbnail"]["url"], out / "thumbnail.png").name
    _rig(cfg, files, out)
    return {"endpoint": cfg.models.mesh_alt, "rig_endpoint": cfg.models.rig, "files": files}


def _tripo(cfg: Config, urls: list[str], out: Path, files: dict) -> dict:
    if "model" not in files:
        res = _subscribe(cfg.models.mesh_alt, {
            "image_urls": urls,  # front, left, back, right
            "quad": True,
            "face_limit": cfg.character.target_tris // 2,
            "texture": True,
            "pbr": True,
            "texture_quality": "detailed",
            "geometry_quality": "detailed",
            "texture_alignment": "original_image",
            "orientation": "default",
            "enable_safety_checker": False,
        })
        mesh_url = _url(res.get("model_mesh"))
        if not mesh_url:
            raise RuntimeError("tripo returned no mesh")
        files["model"] = _download(mesh_url, out / f"model{Path(mesh_url.split('?')[0]).suffix}").name
        if _url(res.get("rendered_image")):
            files["thumbnail"] = _download(res["rendered_image"]["url"], out / "thumbnail.webp").name
    model = out / files["model"]
    if model.suffix == ".fbx":
        files["model_glb"] = _fbx_to_glb(model, out / "model.glb").name
    else:
        files["model_glb"] = model.name
    _rig(cfg, files, out)
    return {"endpoint": cfg.models.mesh_alt, "rig_endpoint": cfg.models.rig, "files": files}


def run(cfg: Config, force: bool = False) -> dict:
    """Generate (or complete) every mesh generator; a generator that already has a rig is kept unless --force."""
    run_ = cfg.run
    report = read_json(run_.mesh_report) if run_.mesh_report.exists() and not force else {"generators": {}}
    views = ordered_views(cfg)
    report["views"] = {v: str(views[v].relative_to(run_.root)) for v in VIEWS}

    alt = cfg.models.mesh_alt
    generators = ((("meshy", _meshy),) if cfg.models.mesh else ()) + (
        (("hunyuan", _hunyuan),) if "hunyuan" in alt else (("hi3d", _hi3d),) if "hitem3d" in alt else
        (("tripo", _tripo),) if alt else ())
    todo = {name: fn for name, fn in generators
            if force or not report["generators"].get(name, {}).get("files", {}).get("rigged_character_glb")}
    if todo:
        fal = _fal()
        urls = [fal.upload_file(str(views[v])) for v in VIEWS]
        log(f"uploaded {len(urls)} views; running {', '.join(todo)}")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {}
            for name, fn in todo.items():
                prev = {} if force else report["generators"].get(name, {}).get("files", {})
                args = (cfg, urls, run_.mesh_dir / name) + ((dict(prev),) if fn is not _meshy else ())
                futures[name] = pool.submit(fn, *args)
            for name, fut in futures.items():
                try:
                    report["generators"][name] = fut.result()
                except Exception as e:  # noqa: BLE001 - keep the other generator's result
                    log(f"[red]{name} failed: {e}[/]")
                    report["generators"][name] = {"error": str(e)}

    for name, info in report["generators"].items():
        info["stats"] = {key: glb_stats(run_.mesh_dir / name / f) for key, f in info.get("files", {}).items()
                         if f.endswith(".glb")}

    order = [cfg.character.mesh_primary] if cfg.character.mesh_primary else []
    primary = next((n for n in [*order, "meshy", "tripo", "hunyuan", "hi3d"] if report["generators"].get(n, {})
                    .get("files", {}).get("rigged_character_glb")), None)
    if primary is None:
        raise RuntimeError("no generator produced a rigged character")
    src = run_.mesh_dir / primary / report["generators"][primary]["files"]["rigged_character_glb"]
    if not run_.rig.exists() or run_.rig.read_bytes() != src.read_bytes():
        shutil.copy(src, run_.rig)
    report["primary"] = primary
    write_json(run_.mesh_report, report)
    for name, info in report["generators"].items():
        st = info.get("stats", {}).get("rigged_character_glb") or {}
        log(f"{name}: {st.get('triangles', '?')} tris, {len(st.get('joints', []))} joints")
    log(f"primary rig: {primary} → {run_.rig}")
    return report
