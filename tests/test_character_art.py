"""The art-team variant: view normalisation, skin evening, face detail transfer and the GLB image swap."""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image

from rybakina3d.character.art import _cutout, _request_size, _square_view
from rybakina3d.character.face import delight, even_skin, oval
from rybakina3d.gltf import _read_glb, _write_glb, base_color_images, replace_image


def test_request_size_fits_budget_and_multiple_of_16():
    w, h = _request_size(1715, 2515)
    assert (w, h) == (1648, 2416)
    assert w % 16 == 0 and h % 16 == 0 and w * h <= 4_000_000


def test_square_view_scales_to_height_and_keeps_feet_on_line(tmp_path):
    fig = Image.new("RGBA", (40, 100), (200, 50, 50, 255))
    out = _square_view(fig, 200, tmp_path / "v.png")
    im = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert im.shape[0] == im.shape[1]
    red = np.nonzero((im[..., 0] > 150) & (im[..., 1] < 100))
    assert red[0].max() - red[0].min() + 1 == 200                    # figure height
    assert im.shape[0] - 1 - red[0].max() == round(200 * 0.06)       # feet MARGIN above the bottom


def test_cutout_crops_to_alpha(tmp_path):
    a = np.zeros((50, 60, 4), np.uint8)
    a[10:30, 20:25] = 255
    Image.fromarray(a).save(tmp_path / "a.png")
    assert _cutout(tmp_path / "a.png").size == (5, 20)


def test_oval_is_centred_between_eyes_and_mouth():
    pts = np.array([[80, 100], [120, 100], [100, 120], [85, 140], [115, 140]], float)
    m = oval(pts, (240, 200))
    ys, xs = np.nonzero(m)
    assert abs(xs.mean() - 100) < 1.5
    assert ys.min() < 100 < 140 < ys.max()


def test_even_skin_darkens_pale_skin_only():
    tex = np.zeros((64, 64, 3), np.uint8)
    tex[:32] = (176, 120, 90)       # body skin
    tex[32:48] = (235, 200, 180)    # pale face skin, same hue family
    tex[48:] = (245, 245, 245)      # white cloth
    out, stats = even_skin(tex)
    lum = lambda x: x.astype(float).mean(-1)  # noqa: E731
    assert lum(out[40, 0]) < lum(tex[40, 0]) - 10                    # pale skin pulled down
    assert np.abs(out[:32].astype(int) - tex[:32]).max() <= 2         # body skin kept
    assert np.abs(out[56].astype(int) - tex[56]).max() <= 2           # cloth kept
    assert stats["skin_share"] > 0.4


def test_delight_takes_detail_from_art_and_tone_from_base():
    art = np.full((128, 128, 3), 60, np.uint8)
    art[60:68, 40:88] = 20                                           # a dark feature (a brow)
    base = np.full((128, 128, 3), 170, np.uint8)
    out = delight(art, base, np.ones((128, 128)), 4.0, 12.0).astype(int)
    assert abs(out[10, 10].mean() - 170) < 3                          # tone = base
    assert out[64, 64].mean() < out[10, 10].mean() - 8                # the feature survives


def _glb(tmp_path, images: list[bytes]):
    binary, views = b"", []
    for data in images:
        views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)})
        binary += data + b"\0" * (-len(data) % 4)
    g = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}], "bufferViews": views,
         "images": [{"bufferView": i, "mimeType": "image/png"} for i in range(len(images))],
         "textures": [{"source": 1}],
         "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]}
    path = tmp_path / "m.glb"
    _write_glb(path, g, binary)
    return path


def test_replace_image_swaps_bytes_and_repacks(tmp_path):
    src = _glb(tmp_path, [b"abc", b"0123456789", b"xy"])
    assert base_color_images(src) == [1]
    out = tmp_path / "out.glb"
    replace_image(src, out, 1, b"NEW-IMAGE-BYTES!!")
    g, binary = _read_glb(out)
    got = [binary[v["byteOffset"]:v["byteOffset"] + v["byteLength"]] for v in g["bufferViews"]]
    assert got == [b"abc", b"NEW-IMAGE-BYTES!!", b"xy"]
    assert all(v["byteOffset"] % 4 == 0 for v in g["bufferViews"])
    assert struct.unpack_from("<I", out.read_bytes(), 8)[0] == out.stat().st_size
    assert g["buffers"][0]["byteLength"] <= len(binary)
