"""Stage `clips`: cut every selected window out of the source, frame-exact.

clip.mp4     every source frame once, re-stamped at the model rate (GVHMR is trained at 30 fps; re-stamping
             instead of resampling avoids duplicated frames, the motion is played back at the source rate).
preview.mp4  the same frames at the true source rate, 720p with audio, for review.
"""

from __future__ import annotations

from .config import Config
from .media import cut_frames, probe
from .util import log, read_json, write_json


def run(cfg: Config, only: list[str] | None = None, force: bool = False) -> None:
    run_ = cfg.run
    sel = read_json(run_.selection)
    fps = sel["fps"]
    src = probe(cfg.source.video)
    for c in sel["clips"]:
        if only and c["id"] not in only:
            continue
        clip = run_.clip(c["id"])
        if clip.meta.exists() and clip.video.exists() and not force:
            continue
        n = c["end_frame"] - c["start_frame"]
        out_fps = cfg.clips.model_fps if cfg.clips.time_mode == "retime" else None
        cut_frames(cfg.source.video, clip.video, c["start_frame"], n, fps, out_fps=out_fps)
        cut_frames(cfg.source.video, clip.preview, c["start_frame"], n, fps, height=cfg.clips.preview_height,
                   audio=True, crf=22)
        got = probe(clip.video).frames
        if got != n:
            log(f"[yellow]{c['id']}: expected {n} frames, got {got}[/]")
        write_json(clip.meta, {
            **c,
            "n_frames": got,
            "src_fps": fps,
            "play_fps": fps,
            "model_fps": out_fps or fps,
            "time_mode": cfg.clips.time_mode,
            "contact_index": c["contact_frame"] - c["start_frame"],
            "size": [src.width, src.height],
            "t_start": c["start_frame"] / fps,
        })
        log(f"{c['id']}: {got} frames ({got / fps:.2f}s) from {c['start_frame'] / fps:.2f}s")
