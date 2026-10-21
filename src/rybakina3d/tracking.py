"""Stage `track`: follow Rybakina through each clip.

YOLO + ByteTrack give person tracks; the track that overlaps Gemini's hitter box on the contact frame is
hers. ByteTrack can switch ids under fast motion, so the chosen track is then extended frame by frame
with the best-overlapping detection, gaps are interpolated and the boxes lightly smoothed (GVHMR crops
around them).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .config import PROJECT_ROOT, Config
from .util import log, read_json


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _detect(model, video, conf: float) -> tuple[dict[int, dict[int, np.ndarray]], list[list[np.ndarray]]]:
    """Per-frame detections and ByteTrack tracks: tracks[id][frame] = xyxy, dets[frame] = [xyxy…]."""
    tracks: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    dets: list[list[np.ndarray]] = []
    stream = model.track(source=str(video), stream=True, persist=False, classes=[0], conf=conf, imgsz=1280,
                         tracker="bytetrack.yaml", verbose=False, half=True)
    for f, r in enumerate(stream):
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        dets.append(list(xyxy))
        if boxes.id is not None:
            for tid, b in zip(boxes.id.int().cpu().tolist(), xyxy, strict=True):
                tracks[tid][f] = b
    return tracks, dets


def _pick(tracks: dict[int, dict[int, np.ndarray]], target: np.ndarray, k: int, n: int) -> int | None:
    best, best_score = None, 0.0
    for tid, boxes in tracks.items():
        for f in sorted(boxes, key=lambda f: abs(f - k)):
            if abs(f - k) > 6:
                break
            score = iou(boxes[f], target) * (1 - abs(f - k) / 10) + 0.05 * len(boxes) / n
            if score > best_score:
                best, best_score = tid, score
            break
    return best if best_score > 0.15 else None


def _extend(seed: dict[int, np.ndarray], dets: list[list[np.ndarray]], n: int) -> dict[int, np.ndarray]:
    """Grow the track forwards/backwards with the best-overlapping detection (fixes id switches)."""
    boxes = dict(seed)
    frames = sorted(boxes)
    for direction, start in ((1, frames[-1]), (-1, frames[0])):
        last = boxes[start]
        f = start + direction
        miss = 0
        while 0 <= f < n and miss < 8:
            if f in boxes:
                last, miss = boxes[f], 0
            else:
                cands = [(iou(d, last), d) for d in dets[f]]
                cands = [c for c in cands if c[0] > 0.3]
                if cands:
                    last = max(cands, key=lambda c: c[0])[1]
                    boxes[f], miss = last, 0
                else:
                    miss += 1
            f += direction
    return boxes


def _fill_and_smooth(boxes: dict[int, np.ndarray], n: int, win: int = 5) -> tuple[np.ndarray, np.ndarray]:
    valid = np.zeros(n, bool)
    out = np.zeros((n, 4), np.float32)
    known = sorted(boxes)
    valid[known] = True
    arr = np.stack([boxes[f] for f in known])
    idx = np.arange(n)
    for j in range(4):
        out[:, j] = np.interp(idx, known, arr[:, j])
    if win > 1 and n >= win:
        pad = win // 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(win) / win
        out = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(4)], axis=1)
    return out.astype(np.float32), valid


def _preview(video, frame: int, box: np.ndarray, out_path) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if ok:
        x0, y0, x1, y1 = box.astype(int)
        cv2.rectangle(img, (x0, y0), (x1, y1), (60, 220, 60), 3)
        cv2.imwrite(str(out_path), cv2.resize(img, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 85])


def run(cfg: Config, only: list[str] | None = None, force: bool = False) -> None:
    from ultralytics import YOLO

    weights = PROJECT_ROOT / "weights" / cfg.track.detector
    weights.parent.mkdir(exist_ok=True)
    model = YOLO(str(weights) if weights.exists() else cfg.track.detector)
    for cid in cfg.run.clip_ids():
        if only and cid not in only:
            continue
        clip = cfg.run.clip(cid)
        if clip.track.exists() and not force:
            continue
        meta = read_json(clip.meta)
        n, k = meta["n_frames"], meta["contact_index"]
        tracks, dets = _detect(model, clip.video, cfg.track.conf)
        target = np.array(meta["hitter_box"], np.float32)
        tid = _pick(tracks, target, k, n)
        if tid is None:
            log(f"[red]{cid}: no track matches the hitter box — skipped[/]")
            continue
        boxes = _extend(tracks[tid], dets, n)
        bbx, valid = _fill_and_smooth(boxes, n)
        coverage = float(valid.mean())
        np.savez_compressed(clip.track, bbx_xyxy=bbx, valid=valid, track_id=tid, coverage=coverage)
        _preview(clip.video, k, bbx[k], clip.track_preview)
        h = float(np.median(bbx[:, 3] - bbx[:, 1]))
        flag = "" if coverage >= cfg.track.min_coverage else "  [yellow]low coverage[/]"
        log(f"{cid}: track {tid}, coverage {coverage:.0%}, median height {h:.0f}px{flag}")
