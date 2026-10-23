"""The review site's character variants (site._variants): a fake variant's Quest 3 / draft motion in a tmp run."""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from rybakina3d import site
from rybakina3d.config import Config, RunLayout
from rybakina3d.media import _bin

CLIPS = [("serve_00m01s", "serve", True), ("serve_00m02s", "serve", False), ("forehand_00m03s", "forehand", True),
         ("backhand_00m04s", "backhand", True)]


def _write(path, data: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def _age(path, seconds: float) -> None:
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


@pytest.fixture
def run_cfg(tmp_path):
    """A run with four moveset clips; serve_00m02s fails its label check (flagged by curate)."""
    cfg = Config.load()
    cfg.run_dir = tmp_path / "run"
    rows = [{"id": cid, "stroke": stroke, "ok": ok, "kept": True, "score": 1.0} for cid, stroke, ok in CLIPS]
    _write(cfg.run.curation, json.dumps({"clips": rows, "kept": [c for c, _, _ in CLIPS], "keep_all": True}))
    for cid, stroke, _ in CLIPS:                     # cut clips: the moveset is the kept ones among them
        _write(cfg.run.clip(cid).meta, json.dumps({"id": cid, "stroke": stroke, "contact_index": 30}))
    return cfg


def _tiny_mp4(path, size=(64, 68), fps=60, seconds=0.2) -> bool:
    """A real H.264 file for the reel (the site probes it); False when ffmpeg is missing."""
    try:
        ffmpeg = _bin("ffmpeg")
    except RuntimeError:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=white:s={size[0]}x{size[1]}:r={fps}:d={seconds}", "-pix_fmt", "yuv420p", str(path)],
                   check=True)
    return True


def _quest3(cfg: Config) -> RunLayout:
    """variants/naomi: composed with serve_00m01s + forehand_00m03s; serve_00m02s rendered after the compose (no
    tile yet); backhand_00m04s not rendered; the moveset GLB, no FBX."""
    v = RunLayout(cfg.run.root, "naomi")
    rdir, tdir = v.motion_renders_for("quest3"), v.motion_dir / "naomi_moveset_tiles_quest3"
    rel = lambda p: p.relative_to(v.root).as_posix()  # noqa: E731
    rows = {}
    for cid in ("serve_00m01s", "forehand_00m03s", "serve_00m02s"):
        _write(rdir / f"{cid}.mp4", b"\0" * 2_000_000)
        _write(rdir / f"{cid}.jpg")
        rows[cid] = {"id": cid, "video": rel(rdir / f"{cid}.mp4"), "poster": rel(rdir / f"{cid}.jpg"),
                     "n_frames": 195, "fps": 60.0, "contact_frame": 180, "size": [2064, 2208]}
    _write(rdir / "renders.json", json.dumps({"profile": "quest3", "fps": 60.0, "size": [2064, 2208], "clips": rows}))
    reel = v.motion_dir / "naomi_moveset_reel_quest3.mp4"
    if not _tiny_mp4(reel):
        _write(reel)
    _write(reel.with_suffix(".jpg"))
    composed = []
    for cid, stroke in (("serve_00m01s", "serve"), ("forehand_00m03s", "forehand")):
        _write(tdir / f"{cid}.mp4")
        _write(tdir / f"{cid}.jpg")
        composed.append({"id": cid, "stroke": stroke, "flagged": False, "mp4": rows[cid]["video"],
                         "poster": rows[cid]["poster"], "tile": rel(tdir / f"{cid}.mp4"),
                         "tile_poster": rel(tdir / f"{cid}.jpg"), "size": [2064, 2208], "fps": 60.0, "frames": 195,
                         "contact_frame": 180, "mb": 2.0})
    _write(v.motion_dir / "motion_quest3.json", json.dumps({
        "profile": "quest3", "name": "Naomi Osaka", "motion_from": "Elena Rybakina", "size": [2064, 2208],
        "fps": 60.0, "reel": {"mp4": rel(reel), "poster": rel(reel.with_suffix(".jpg")), "frames": 312, "loop_s": 5.2,
                              "mb": 0.01},
        "groups": [{"stroke": "serve", "clips": ["serve_00m01s"]},
                   {"stroke": "forehand", "clips": ["forehand_00m03s"]}],
        "clips": composed, "made": "2026-10-22T12:00:00"}))
    # the compose ran after the first two renders, serve_00m02s was rendered after it
    for p in (rdir / "serve_00m01s.mp4", rdir / "forehand_00m03s.mp4"):
        _age(p, 100)
    for p in (reel, *tdir.iterdir()):
        _age(p, 50)
    _write(v.moveset_glb)
    _age(v.moveset_glb, 200)                       # every render above was made from this moveset
    return v


def test_no_variants_folder_means_no_section(run_cfg, tmp_path):
    link = site._Linker(tmp_path / "site")
    assert site._variants(run_cfg, link) == []
    (run_cfg.run.root / "variants" / "empty").mkdir(parents=True)          # a folder with nothing in it yet
    assert site._variants(run_cfg, link) == [] and link.count == 0


def test_quest3_variant_links_what_exists(run_cfg, tmp_path):
    v = _quest3(run_cfg)
    link = site._Linker(tmp_path / "site")
    [n] = site._variants(run_cfg, link)
    assert (n["variant"], n["name"], n["motion_from"], n["profile"], n["composed"]) == \
        ("naomi", "Naomi Osaka", "Elena Rybakina", "quest3", True)
    assert n["config"] == "configs/naomi.yaml" and n["size"] == [2064, 2208] and n["fps"] == 60.0
    assert n["progress"]["quest3"] == {"rendered": 3, "composed": True, "size": [2064, 2208]}
    assert n["progress"]["draft"]["rendered"] == 0 and not n["progress"]["draft"]["composed"]
    # the reel: probed when ffmpeg made a real one, else the report's numbers; it predates serve_00m02s's render
    reel = n["reel"]
    assert reel["url"] == "media/variants/naomi/artwork/motion/naomi_moveset_reel_quest3.mp4"
    assert reel["poster"] == "media/variants/naomi/artwork/motion/naomi_moveset_reel_quest3.jpg"
    assert reel["loop_s"] == 5.2 and reel["fps"] == 60.0 and reel["size"] in ([64, 68], [2064, 2208])
    assert n["n_in_reel"] == 2 and n["reel_stale"] is True
    # groups: the report's, then the moveset clips it does not hold yet in their stroke's group, atlas order
    assert n["groups"] == [{"stroke": "serve", "clips": ["serve_00m01s", "serve_00m02s"]},
                           {"stroke": "forehand", "clips": ["forehand_00m03s"]},
                           {"stroke": "backhand", "clips": ["backhand_00m04s"]}]
    by = {c["id"]: c for c in n["clips"]}
    s1 = by["serve_00m01s"]
    assert s1["mp4"] == "media/variants/naomi/artwork/motion/render_quest3/serve_00m01s.mp4" and s1["mb"] == 2.0
    assert s1["tile"] == "media/variants/naomi/artwork/motion/naomi_moveset_tiles_quest3/serve_00m01s.mp4"
    assert s1["tile_poster"].endswith("naomi_moveset_tiles_quest3/serve_00m01s.jpg") and s1["tile_stale"] is False
    assert (s1["frames"], s1["contact_frame"], s1["size"], s1["flagged"]) == (195, 180, [2064, 2208], False)
    s2 = by["serve_00m02s"]                          # rendered since the compose: full video + poster, no tile
    assert s2["mp4"] and s2["poster"] and s2["tile"] is None and s2["flagged"] is True
    assert (s2["frames"], s2["contact_frame"]) == (195, 180)
    bh = by["backhand_00m04s"]                       # not rendered yet
    assert bh["mp4"] is None and bh["poster"] is None and bh["tile"] is None
    assert (n["n_clips"], n["n_rendered"], n["n_tiles"], n["n_flagged"]) == (4, 3, 2, 1)
    assert n["downloads"]["glb"]["url"] == "media/variants/naomi/anim/naomi_moveset.glb"
    assert n["downloads"]["fbx"] is None and n["downloads"]["animations"] is None   # a GLB cut short: no stats
    # every URL is a link that resolves inside the site
    urls = [reel["url"], reel["poster"], n["downloads"]["glb"]["url"]]
    urls += [c[k] for c in n["clips"] for k in ("mp4", "poster", "tile", "tile_poster") if c[k]]
    for u in urls:
        assert (tmp_path / "site" / u).is_symlink() and (tmp_path / "site" / u).exists(), u
    # a tile older than a re-render of its clip is still played, but counted as stale
    os.utime(v.motion_renders_for("quest3") / "forehand_00m03s.mp4")
    n = site._variants(run_cfg, site._Linker(tmp_path / "site"))[0]
    assert n["n_tiles_stale"] == 1 and next(c for c in n["clips"] if c["id"] == "forehand_00m03s")["tile_stale"]


def test_renders_of_an_earlier_moveset_wait_for_their_re_render(run_cfg, tmp_path):
    v = _quest3(run_cfg)
    _write(v.moveset_glb)                          # the rig changed: renders older than the moveset show the old one
    _age(v.motion_renders_for("quest3") / "serve_00m02s.mp4", -10)     # this one was re-rendered since
    [n] = site._variants(run_cfg, site._Linker(tmp_path / "site"))
    by = {c["id"]: c for c in n["clips"]}
    old = by["serve_00m01s"]
    assert old["outdated"] is True and old["mp4"] is None and old["tile"] is None and old["poster"] is None
    assert by["serve_00m02s"]["outdated"] is False and by["serve_00m02s"]["mp4"]
    assert (n["n_rendered"], n["n_outdated"]) == (1, 2)


def test_draft_is_live_until_quest3_is_composed(run_cfg, tmp_path):
    v = RunLayout(run_cfg.run.root, "naomi")
    ddir = v.motion_renders_for("draft")
    _write(ddir / "serve_00m01s.mp4")
    _write(ddir / "renders.json", json.dumps({"profile": "draft", "fps": 60.0, "size": [516, 552], "clips": {
        "serve_00m01s": {"video": "serve_00m01s.mp4", "n_frames": 195, "contact_frame": 180, "size": [516, 552]}}}))
    _write(v.motion_dir / "naomi_moveset_reel_draft.mp4")
    # the draft's reel is on the same fixed Quest 3 canvas as the quest3 one; its clips are quarter size
    _write(v.motion_dir / "motion_draft.json", json.dumps({
        "profile": "draft", "name": "Naomi Osaka", "size": [2064, 2208], "fps": 60.0,
        "reel": {"mp4": "variants/naomi/artwork/motion/naomi_moveset_reel_draft.mp4", "frames": 312, "loop_s": 5.2},
        "clips": [{"id": "serve_00m01s", "stroke": "serve", "flagged": False,
                   "mp4": "variants/naomi/artwork/motion/render_draft/serve_00m01s.mp4"}]}))
    # Quest 3 renders are coming in, but nothing is composed at that profile yet: the draft stays live
    qdir = v.motion_renders_for("quest3")
    _write(qdir / "serve_00m01s.mp4")
    _write(qdir / "renders.json", json.dumps({"clips": {"serve_00m01s": {"video": "serve_00m01s.mp4"}}}))
    _write(v.motion_dir / "motion_quest3.json", "{not json")          # half-written report: ignored
    [n] = site._variants(run_cfg, site._Linker(tmp_path / "site"))
    assert n["profile"] == "draft" and n["composed"] and n["size"] == [516, 552]      # the clips', not the reel's
    assert n["progress"]["quest3"]["rendered"] == 1 and not n["progress"]["quest3"]["composed"]
    assert n["reel"]["url"].endswith("naomi_moveset_reel_draft.mp4") and n["reel"]["poster"] is None
    assert n["reel"]["size"] == [2064, 2208] and n["reel"]["frames"] == 312     # not a real video: the report's
    # ...but the clip whose Quest 3 video exists already links that one, with its own facts
    c = next(c for c in n["clips"] if c["id"] == "serve_00m01s")
    assert c["mp4"] == "media/variants/naomi/artwork/motion/render_quest3/serve_00m01s.mp4" and c["tile"] is None
    assert c["profile"] == "quest3" and n["n_hires"] == 1
    assert (c["size"], c["fps"], c["frames"]) == ([2064, 2208], 60.0, None)


def test_a_partial_quest3_reel_does_not_replace_the_complete_draft(run_cfg, tmp_path):
    v = RunLayout(run_cfg.run.root, "naomi")
    moveset = v.kept_clip_ids()
    for profile, ids in (("draft", moveset), ("quest3", moveset[:1])):
        rdir = v.motion_renders_for(profile)
        for cid in ids:
            _write(rdir / f"{cid}.mp4")
        _write(v.motion_dir / f"motion_{profile}.json", json.dumps({"profile": profile, "clips": [
            {"id": cid, "mp4": (rdir / f"{cid}.mp4").relative_to(v.root).as_posix()} for cid in ids]}))
    [n] = site._variants(run_cfg, site._Linker(tmp_path / "site"))
    assert n["profile"] == "draft" and n["n_hires"] == 1
    rdir = v.motion_renders_for("quest3")
    for cid in moveset:
        _write(rdir / f"{cid}.mp4")
    _write(v.motion_dir / "motion_quest3.json", json.dumps({"profile": "quest3", "clips": [
        {"id": cid, "mp4": (rdir / f"{cid}.mp4").relative_to(v.root).as_posix()} for cid in moveset]}))
    [n] = site._variants(run_cfg, site._Linker(tmp_path / "site2"))
    assert n["profile"] == "quest3" and n["n_hires"] == len(moveset)


def test_clips_rendered_after_a_partial_compose_keep_the_atlas_order(run_cfg, tmp_path):
    """The reel was composed while only forehand_00m03s had a render: serve clips rendered since must not trail it."""
    v = RunLayout(run_cfg.run.root, "naomi")
    rdir = v.motion_renders_for("quest3")
    rel = lambda p: p.relative_to(v.root).as_posix()  # noqa: E731
    for cid in ("serve_00m02s", "forehand_00m03s", "serve_00m01s"):
        _write(rdir / f"{cid}.mp4")
    _write(rdir / "renders.json", json.dumps({"profile": "quest3", "clips": {
        cid: {"video": rel(rdir / f"{cid}.mp4")} for cid in ("serve_00m02s", "forehand_00m03s", "serve_00m01s")}}))
    _write(v.motion_dir / "motion_quest3.json", json.dumps({
        "profile": "quest3", "groups": [{"stroke": "forehand", "clips": ["forehand_00m03s"]}],
        "clips": [{"id": "forehand_00m03s", "stroke": "forehand", "mp4": rel(rdir / "forehand_00m03s.mp4")}]}))
    [n] = site._variants(run_cfg, site._Linker(tmp_path / "site"))
    assert n["groups"] == [{"stroke": "serve", "clips": ["serve_00m01s", "serve_00m02s"]},
                           {"stroke": "forehand", "clips": ["forehand_00m03s"]},
                           {"stroke": "backhand", "clips": ["backhand_00m04s"]}]
    assert [c["id"] for c in n["clips"]] == ["serve_00m01s", "serve_00m02s", "forehand_00m03s", "backhand_00m04s"]
    assert n["n_in_reel"] is None and n["n_rendered"] == 3          # no reel file: nothing to call stale


def test_renders_only_and_moveset_only(run_cfg, tmp_path):
    v = RunLayout(run_cfg.run.root, "osaka_b")
    qdir = v.motion_renders_for("quest3")
    _write(qdir / "forehand_00m03s.mp4")
    _write(qdir / "forehand_00m03s.jpg")
    _write(qdir / "renders.json", json.dumps({"fps": 60.0, "size": [2064, 2208], "clips": [
        {"id": "forehand_00m03s", "video": "forehand_00m03s.mp4", "n_frames": 195, "contact_frame": 180}]}))
    w = RunLayout(run_cfg.run.root, "zz_moveset_only")
    _write(w.moveset_fbx)
    _write(RunLayout(run_cfg.run.root, "rig_only").rig)           # character still being built: not listed
    a, b = site._variants(run_cfg, site._Linker(tmp_path / "site"))
    # no reel report: every moveset clip in the atlas order, the one rendered with its video and poster
    assert (a["variant"], a["name"], a["config"], a["profile"], a["composed"]) == \
        ("osaka_b", "Osaka B", None, "quest3", False)
    assert a["reel"] is None and a["n_in_reel"] is None and a["reel_stale"] is False
    assert [g["stroke"] for g in a["groups"]] == ["serve", "forehand", "backhand"]
    assert (a["n_clips"], a["n_rendered"], a["n_tiles"]) == (4, 1, 0)
    fh = next(c for c in a["clips"] if c["id"] == "forehand_00m03s")
    assert fh["mp4"].endswith("osaka_b/artwork/motion/render_quest3/forehand_00m03s.mp4") and fh["poster"]
    assert (fh["size"], fh["fps"], fh["frames"]) == ([2064, 2208], 60.0, 195)
    # a variant with only its moveset: listed (downloads), nothing in motion yet
    assert b["profile"] is None and b["clips"] == [] and b["reel"] is None
    assert b["downloads"]["fbx"]["url"] == "media/variants/zz_moveset_only/anim/zz_moveset_only_moveset.fbx"


def test_build_writes_variants_into_site_json(run_cfg, tmp_path):
    _quest3(run_cfg)
    out = site.build(run_cfg)
    d = json.loads((out / "data" / "site.json").read_text())
    [n] = d["variants"]
    assert n["name"] == "Naomi Osaka" and n["n_rendered"] == 3
    assert (out / n["reel"]["url"]).exists() and (out / "app.js").exists()
    # the variant's links survive the prune of a rebuild, and go when the variant does
    site.build(run_cfg)
    assert (out / n["clips"][0]["mp4"]).exists()
    shutil.rmtree(run_cfg.run.root / "variants")
    d = json.loads((site.build(run_cfg) / "data" / "site.json").read_text())
    assert d["variants"] == [] and not (out / "media" / "variants").exists()


def _notes_of(retarget: dict, joints: int = 24) -> dict:
    """site._notes on a run with no clips yet: only the moveset's retarget.json and the rig's joints."""
    from rybakina3d.config import PROJECT_ROOT

    cfg = Config.load(PROJECT_ROOT / "configs" / "naomi.yaml")
    d = {"clips": [], "character": {"rig": {"stats": {"joints": [f"Bone{i}" for i in range(joints)]}}},
         "moveset": {"retarget": retarget, "url": "moveset.glb"}, "footage": {"shots": []}, "curation": None,
         "qa_summary": None, "pipeline": [], "signature": None}
    return site._notes(cfg, d)


def test_notes_read_the_hands_off_retarget_json():
    racket = {"bone": "RightHand", "length_m": 0.685, "grip_angle_deg": 25.0}
    plain = _notes_of({"settings": {}, "racket": racket})
    text = " ".join(plain["limitations"] + plain["next"])
    assert "no fingers" in text and "Pin the left hand" in text and "25° grip" in text
    hands = {"settings": {"grip": "palm", "fingers": True, "left_hand_ik": True}, "finger_bones": 30}
    on = _notes_of({"settings": {}, "racket": {**racket, "grip": "palm", "grip_angle_deg": 41.0}, "hands": hands},
                   joints=54)
    text = " ".join(on["limitations"] + on["next"])
    assert "The 30 finger bones" in text and "IK puts the left hand on the handle" in text
    assert "seated on the palm, 41° across it" in text
    assert "no fingers" not in text and "Pin the left hand" not in text and "not constrained" not in text
