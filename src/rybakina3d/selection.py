"""Stage `select`: turn Gemini's shot labels into a ranked set of stroke clips.

For every stroke Rybakina hits in a usable shot:
  1. refine the contact time to the strongest audio onset near Gemini's estimate (ball strikes are sharp);
  2. ask Gemini for the hitter's box on the contact frame — this identifies *her* among the people on court
     and measures how large she is in the image;
  3. cut a window around contact, clipped to the shot; rank, and keep `oversample` × the quota per stroke
     (`curate` later rejects clips whose recovered motion contradicts the label).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from .config import Config
from .gemini import generate, image_part, text_part
from .media import read_frame
from .util import log, read_json, step, write_json

NEAR_PLAYER_PX = 400.0   # a near-court player's height at 1080p


def size_score(height_px: float) -> float:
    """Ranking credit for how large the hitter is in the frame: 1 at a near-court player's height, capped at 1.5."""
    return min(height_px / NEAR_PLAYER_PX, 1.5)


class HitterBox(BaseModel):
    found: bool
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0-1000")
    full_body_visible: bool
    confidence: float


LOCATE_PROMPT = """This is a frame from a tennis broadcast at the moment of a {stroke} hit by {athlete}
({handedness}-handed, {backhand} backhand, outfit: {outfit}). She is on the {side} side of the court.
Return the bounding box of HER whole body (head to feet, including the racket arm) as box_2d
[ymin, xmin, ymax, xmax] normalized to 0-1000. found=false if she is not visible."""


@dataclass
class Candidate:
    shot_id: int
    stroke: str
    t_gemini: float
    confidence: float
    view: str
    side: str
    quality: int
    outfit: str
    shot_start_frame: int
    shot_end_frame: int
    contact_frame: int = 0
    contact_source: str = "gemini"
    onset_strength: float = 0.0
    hitter_box: list[float] = field(default_factory=list)   # pixels xyxy on the contact frame
    hitter_height_px: float = 0.0
    start_frame: int = 0
    end_frame: int = 0
    score: float = 0.0
    id: str = ""


def _candidates(cfg: Config, index: dict) -> list[Candidate]:
    sel = cfg.select
    out = []
    for s in index["shots"]:
        if s["view"] not in sel.allowed_views or s["replay"]:
            continue
        if s["slow_motion"] and not sel.allow_slow_motion:
            continue
        if s["rybakina_side"] not in ("near", "solo") or not s["full_body"] or s["quality"] < sel.min_quality:
            continue
        for e in s["events"]:
            if e["hitter"] != "rybakina" or e["type"] not in sel.strokes:
                continue
            out.append(Candidate(s["shot_id"], e["type"], e["t"], e["confidence"], s["view"], s["rybakina_side"],
                                 s["quality"], s["outfit"], s["start_frame"], s["end_frame"]))
    return out


class Onsets:
    """Spectral-flux onset envelope of the soundtrack (10 ms hops), z-scored locally."""

    def __init__(self, wav_path):
        import librosa
        from scipy.io import wavfile

        sr, y = wavfile.read(wav_path)
        y = y.astype(np.float32) / 32768.0
        # ball strikes live above the commentary band
        self.env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=160, n_fft=512, fmin=1500, n_mels=40)
        self.hop_s = 160 / sr

    def refine(self, t: float, radius: float) -> tuple[float, float]:
        i0, i1 = int((t - radius) / self.hop_s), int((t + radius) / self.hop_s)
        i0, i1 = max(0, i0), min(len(self.env) - 1, i1)
        if i1 <= i0:
            return t, 0.0
        seg = self.env[i0:i1]
        ctx = self.env[max(0, i0 - 300): i1 + 300]  # ±3 s context for the z-score
        k = int(np.argmax(seg))
        z = float((seg[k] - np.median(ctx)) / (np.std(ctx) + 1e-6))
        return (i0 + k) * self.hop_s, z


def _locate(cfg: Config, c: Candidate, fps: float, cache: dict) -> None:
    import cv2

    key = f"{c.shot_id}:{c.contact_frame}"
    if key in cache:
        c.hitter_box, c.hitter_height_px = cache[key]
        return
    frame = read_frame(cfg.source.video, c.contact_frame, fps)
    h, w = frame.shape[:2]
    ok, jpg = cv2.imencode(".jpg", frame[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    prompt = LOCATE_PROMPT.format(stroke=c.stroke, athlete=cfg.athlete.name, outfit=c.outfit,
                                  handedness=cfg.athlete.handedness, backhand=cfg.athlete.backhand.replace("_", "-"),
                                  side="near (closest to the camera)" if c.side == "near" else "visible")
    res = generate(cfg.models.locate, [image_part(jpg.tobytes()), text_part(prompt)], HitterBox)
    if res.found and len(res.box_2d) == 4:
        y0, x0, y1, x1 = res.box_2d
        c.hitter_box = [x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h]
        c.hitter_height_px = (y1 - y0) / 1000 * h
        if not res.full_body_visible:
            c.hitter_height_px *= 0.5  # partly visible: rank down
    cache[key] = (c.hitter_box, c.hitter_height_px)


def run(cfg: Config, force: bool = False) -> list[dict]:
    run_ = cfg.run
    if run_.selection.exists() and not force:
        log(f"selection exists ({run_.selection}); --force to redo")
        return read_json(run_.selection)["clips"]
    fps = read_json(run_.probe)["fps"]
    cands = _candidates(cfg, read_json(run_.index))
    log(f"{len(cands)} candidate strokes after label filters")

    with step("audio onset refinement"):
        onsets = Onsets(run_.audio)
        for c in cands:
            t, z = onsets.refine(c.t_gemini, cfg.select.onset_search_s)
            if c.stroke != "celebration" and z > 2.5:
                c.contact_frame, c.contact_source, c.onset_strength = round(t * fps), "audio", z
            else:
                c.contact_frame = round(c.t_gemini * fps)
            c.contact_frame = int(np.clip(c.contact_frame, c.shot_start_frame + 1, c.shot_end_frame - 2))

    cache_path = run_.index_dir / "locate.json"
    cache = {k: tuple(v) for k, v in read_json(cache_path).items()} if cache_path.exists() else {}
    with step(f"locating the hitter on {len(cands)} contact frames ({cfg.models.locate})"):
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda c: _locate(cfg, c, fps, cache), cands))
    write_json(cache_path, cache)
    cands = [c for c in cands if c.hitter_box]

    for c in cands:
        pre, post = cfg.select.window(c.stroke)
        want = round((pre + post) * fps)
        c.start_frame = max(c.shot_start_frame + 1, c.contact_frame - round(pre * fps))
        c.end_frame = min(c.shot_end_frame - 1, c.contact_frame + round(post * fps) + 1)
        coverage = (c.end_frame - c.start_frame) / want
        size = size_score(c.hitter_height_px)
        evidence = c.quality / 5 + size + 0.5 * c.confidence + 0.3 * (c.contact_source == "audio")
        c.score = evidence * min(1.0, coverage)
        if coverage < 0.6:
            c.score = 0.0

    # over-generate: `curate` validates every clip on its recovered motion and keeps the quota
    chosen: list[Candidate] = []
    for stroke, keep in cfg.select.strokes.items():
        quota = round(keep * cfg.select.oversample)
        pool_ = sorted((c for c in cands if c.stroke == stroke and c.score > 0), key=lambda c: -c.score)
        picked, used_shots = [], set()
        for c in pool_:
            if len(picked) >= quota:
                break
            if c.shot_id in used_shots:  # one clip per shot and stroke: more variety
                continue
            if any(abs(c.contact_frame - p.contact_frame) < fps for p in chosen + picked):
                continue
            used_shots.add(c.shot_id)
            picked.append(c)
        for c in picked:  # stable, readable id: stroke + source timestamp of the contact
            t = c.contact_frame / fps
            c.id = f"{stroke}_{int(t // 60):02d}m{int(t % 60):02d}s"
        chosen += picked
        log(f"{stroke:12s} {len(picked)}/{quota} (from {len(pool_)})")

    clips = [asdict(c) for c in chosen]
    write_json(run_.selection, {"fps": fps, "clips": clips, "n_candidates": len(cands)})
    return clips
