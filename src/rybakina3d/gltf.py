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


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    g = json.loads(data[20 : 20 + json_len])
    rest = 20 + json_len
    if rest + 8 > len(data):
        return g, b""
    bin_len, bin_type = struct.unpack_from("<II", data, rest)
    if bin_type != 0x004E4942:
        raise ValueError("second GLB chunk is not BIN")
    return g, data[rest + 8 : rest + 8 + bin_len]


def _write_glb(path: Path, g: dict, binary: bytes) -> None:
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + (8 + len(binary) if binary else 0)
    out = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
    if binary:
        out += struct.pack("<II", len(binary), 0x004E4942) + binary
    Path(path).write_bytes(out)


def base_color_images(path: Path) -> list[int]:
    """Indices of the images that materials use as base colour."""
    g = read_glb_json(path)
    tex = g.get("textures", [])
    found = []
    for m in g.get("materials", []):
        info = m.get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if info is not None and tex[info["index"]].get("source") is not None:
            src = tex[info["index"]]["source"]
            if src not in found:
                found.append(src)
    return found


def replace_image(path: Path, out: Path, image: int, data: bytes, mime: str = "image/png") -> None:
    """Copy a GLB with one embedded image swapped for `data`; every buffer view is repacked 4-byte aligned."""
    g, binary = _read_glb(path)
    target = g["images"][image]["bufferView"]
    views = g["bufferViews"]
    packed = bytearray()
    for i in sorted(range(len(views)), key=lambda i: views[i].get("byteOffset", 0)):
        v = views[i]
        chunk = data if i == target else binary[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]]
        packed += b"\0" * (-len(packed) % 4)
        v["byteOffset"], v["byteLength"] = len(packed), len(chunk)
        packed += chunk
    g["buffers"][0]["byteLength"] = len(packed)
    g["images"][image]["mimeType"] = mime
    _write_glb(out, g, bytes(packed))
