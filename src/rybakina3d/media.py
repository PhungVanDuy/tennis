"""ffmpeg / ffprobe wrappers. Frame-exact: every cut is addressed by source frame index."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT


def _bin(name: str) -> str:
    local = PROJECT_ROOT / "bin" / name  # static build on the GPU host
    if local.exists():
        return str(local)
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"{name} not found (install ffmpeg or put a static build in bin/)")
    return found


def run_ffmpeg(*args: str) -> None:
    """Run ffmpeg; the last argument is the output, written to a temp name and renamed on success."""
    *head, out = args
    out_path = Path(out)
    tmp = out_path.with_name(f".{out_path.stem}.part{out_path.suffix}")
    try:
        subprocess.run([_bin("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *head, str(tmp)], check=True)
        tmp.replace(out_path)
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class Probe:
    width: int
    height: int
    fps: float
    fps_fraction: str
    frames: int
    duration: float
    has_audio: bool

    def to_dict(self) -> dict:
        return asdict(self)


def probe(video: Path) -> Probe:
    out = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    fps = Fraction(v["avg_frame_rate"])
    duration = float(info["format"]["duration"])
    frames = int(v.get("nb_frames") or round(duration * fps))
    return Probe(
        width=int(v["width"]), height=int(v["height"]), fps=float(fps), fps_fraction=str(fps),
        frames=frames, duration=duration,
        has_audio=any(s["codec_type"] == "audio" for s in info["streams"]),
    )


def extract_audio(video: Path, out: Path, sr: int = 16000) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg("-i", str(video), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out))
    return out


def make_proxy(video: Path, out: Path, height: int = 360) -> Path:
    """Small upload copy for the video-understanding model (keeps audio: ball strikes help)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg("-i", str(video), "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "30", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out))
    return out


def cut_frames(video: Path, out: Path, start_frame: int, n_frames: int, src_fps: float, *,
               out_fps: float | None = None, height: int | None = None, crf: int = 16,
               audio: bool = False) -> Path:
    """Copy source frames [start_frame, start_frame + n_frames) one-to-one into ``out``.

    ``out_fps`` re-stamps the frames (no frames dropped or duplicated); ``None`` keeps the source rate.
    Seeking half a frame early makes the accurate seek land exactly on ``start_frame``.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    ss = max(0.0, (start_frame - 0.5) / src_fps)
    vf = []
    if out_fps is not None:
        vf.append(f"setpts=N/({out_fps}*TB)")
    if height is not None:
        vf.append(f"scale=-2:{height}")
    args = ["-ss", f"{ss:.6f}", "-i", str(video), "-frames:v", str(n_frames)]
    if vf:
        args += ["-vf", ",".join(vf)]
    if out_fps is not None:
        args += ["-r", f"{out_fps}"]
    args += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if audio and out_fps is None:
        # -frames:v caps the video; the audio limit gets one frame of slack so it never trims the last frame
        args += ["-t", f"{(n_frames + 1) / src_fps:.6f}", "-c:a", "aac", "-b:a", "96k"]
    else:
        args += ["-an"]
    args += ["-movflags", "+faststart", str(out)]
    run_ffmpeg(*args)
    return out


def retime(video: Path, out: Path, fps: float, height: int | None = None) -> Path:
    """Re-stamp every frame of ``video`` at ``fps`` (frame-for-frame, used to align overlays to the source)."""
    vf = [f"setpts=N/({fps}*TB)"] + ([f"scale=-2:{height}"] if height else [])
    run_ffmpeg("-i", str(video), "-vf", ",".join(vf), "-r", f"{fps}", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out))
    return out


def read_frame(video: Path, frame: int, fps: float) -> np.ndarray:
    """One RGB frame by index (accurate seek)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, (frame + 0.25) / fps * 1000.0)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame} of {video}")
    return img[..., ::-1].copy()


def read_frames(video: Path) -> list[np.ndarray]:
    """All frames (BGR) of a short clip."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames
