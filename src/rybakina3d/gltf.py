"""Dependency-free GLB inspection (triangles, skins, joints, animations, textures)."""

from __future__ import annotations

import json
import struct
from pathlib import Path


def read_glb_json(path: Path) -> dict:
    data = Path(path).read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a GLB")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(data[20 : 20 + chunk_len])


def glb_stats(path: Path) -> dict:
    g = read_glb_json(path)
    acc = g.get("accessors", [])
    tris = 0
    verts = 0
    for mesh in g.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                continue
            if "indices" in prim:
                tris += acc[prim["indices"]]["count"] // 3
            else:
                tris += acc[prim["attributes"]["POSITION"]]["count"] // 3
            verts += acc[prim["attributes"]["POSITION"]]["count"]
    nodes = g.get("nodes", [])
    skins = g.get("skins", [])
    joints = [nodes[j].get("name", f"node{j}") for s in skins for j in s.get("joints", [])]
    return {
        "triangles": tris,
        "vertices": verts,
        "meshes": len(g.get("meshes", [])),
        "materials": len(g.get("materials", [])),
        "textures": len(g.get("images", [])),
        "skins": len(skins),
        "joints": joints,
        "animations": [a.get("name", f"anim{i}") for i, a in enumerate(g.get("animations", []))],
        "size_mb": round(Path(path).stat().st_size / 1e6, 2),
    }
