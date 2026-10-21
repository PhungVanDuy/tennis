"""Stage `ingest` + `shots`: probe the compilation, extract audio, build the upload proxy, split into shots."""

from __future__ import annotations

from .config import Config
from .media import extract_audio, make_proxy, probe
from .util import log, read_json, step, write_json


def ingest(cfg: Config) -> None:
    run = cfg.run
    run.ensure(run.source_dir)
    with step("probe"):
        p = probe(cfg.source.video)
        write_json(run.probe, p.to_dict())
        log(f"{p.width}x{p.height} @ {p.fps_fraction} fps, {p.frames} frames, {p.duration / 60:.1f} min")
    if not run.audio.exists():
        with step("audio 16 kHz mono"):
            extract_audio(cfg.source.video, run.audio)
    if not run.proxy.exists():
        with step(f"proxy {cfg.index.proxy_height}p"):
            make_proxy(cfg.source.video, run.proxy, cfg.index.proxy_height)


def detect_shots(cfg: Config) -> list[dict]:
    """Hard cuts via PySceneDetect's adaptive detector (run on the proxy: same frames, cheap decode)."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector

    run = cfg.run
    pr = read_json(run.probe)
    fps = pr["fps"]
    with step("shot detection"):
        video = open_video(str(run.proxy))
        sm = SceneManager()
        sm.auto_downscale = False
        sm.add_detector(AdaptiveDetector(adaptive_threshold=cfg.shots.adaptive_threshold,
                                         min_scene_len=max(2, round(cfg.shots.min_shot_s * fps))))
        sm.detect_scenes(video, show_progress=False)
        scenes = sm.get_scene_list(start_in_scene=True)
    shots = []
    for i, (a, b) in enumerate(scenes):
        f0, f1 = a.get_frames(), b.get_frames()
        shots.append({"id": i, "start_frame": f0, "end_frame": f1,
                      "start": round(f0 / fps, 3), "end": round(f1 / fps, 3)})
    write_json(run.shots, {"fps": fps, "frames": pr["frames"], "shots": shots})
    lens = sorted(s["end"] - s["start"] for s in shots)
    log(f"{len(shots)} shots, median {lens[len(lens) // 2]:.1f}s, longest {lens[-1]:.1f}s")
    return shots
