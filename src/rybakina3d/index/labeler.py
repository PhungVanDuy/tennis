"""Stage `index`: Gemini labels every detected shot (view, replay, Rybakina's side, strokes + contact times).

The compilation is uploaded once; each request asks about one time window and lists the shots inside it,
so the model labels known intervals instead of inventing its own segmentation.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ..config import Config
from ..gemini import generate, text_part, upload_video, video_part
from ..util import log, read_json, step, write_json
from .schema import ChunkLabels

PROMPT = """You are labelling a highlights compilation of the tennis player {athlete} for a motion-capture pipeline.
{athlete} is a tall, right-handed WTA player with a two-handed backhand; she is the subject of almost every rally.

The clip you see covers {t0:.1f}s–{t1:.1f}s of the full video. It contains these shots (hard cuts already detected),
times in seconds from the start of the FULL video:
{table}

For EVERY shot id above return one entry:
- view: main_broadcast (high camera behind a baseline, whole court), behind_low (low camera behind the baseline,
  player large), side (courtside / side-on), closeup (face, torso, bench, crowd reaction), other (graphics, crowd, ceremony).
- replay / slow_motion: is this a replay, is it slowed down.
- rybakina_side: near (on the half closest to the camera), far, solo (only she is visible), absent.
- full_body: her whole body from head to feet is visible for most of the shot.
- quality 1-5: how useful the shot is for capturing HER body motion in 3D (large, sharp, unoccluded, steady camera = 5).
- outfit: her outfit colours.
- events: every stroke hit by EITHER player in the shot: serve, forehand, backhand (two-handed drive), slice
  (one-handed backhand slice), volley, overhead; plus "celebration" when she celebrates (fist pump, arms up).
  t = the exact moment the racket hits the ball, in seconds from the start of the FULL video, to 0.1 s (use the
  sound of the strike). hitter = rybakina / opponent / unknown. confidence 0-1.
Be precise; do not invent strokes that are not visible."""


def _chunks(shots: list[dict], chunk_s: float) -> list[list[dict]]:
    out, cur, t_start = [], [], None
    for s in shots:
        if t_start is None:
            t_start = s["start"]
        cur.append(s)
        if s["end"] - t_start >= chunk_s:
            out.append(cur)
            cur, t_start = [], None
    if cur:
        out.append(cur)
    return out


def _label_chunk(cfg: Config, file, chunk: list[dict], k: int) -> dict:
    cache = cfg.run.index_dir / f"chunk_{k:02d}.json"
    if cache.exists():
        return read_json(cache)
    t0, t1 = chunk[0]["start"], chunk[-1]["end"]
    table = "\n".join(f"  shot {s['id']}: {s['start']:.2f} – {s['end']:.2f}" for s in chunk)
    prompt = PROMPT.format(athlete=cfg.athlete.name, t0=t0, t1=t1, table=table)
    res = generate(cfg.models.index, [video_part(file, t0, t1, cfg.index.fps), text_part(prompt)], ChunkLabels,
                   media_resolution="MEDIA_RESOLUTION_LOW")
    data = {"t0": t0, "t1": t1, "shot_ids": [s["id"] for s in chunk], "labels": res.model_dump(mode="json")}
    write_json(cache, data)
    log(f"chunk {k:02d} {t0:7.1f}–{t1:7.1f}s: {len(res.shots)} shots labelled")
    return data


def _fix_time(t: float, shot: dict, t0: float) -> float | None:
    """Gemini sometimes reports times relative to the window start; map them back to absolute."""
    lo, hi = shot["start"] - 0.6, shot["end"] + 0.6
    if lo <= t <= hi:
        return t
    if lo <= t + t0 <= hi:
        return t + t0
    return None


def run(cfg: Config, force: bool = False) -> None:
    run_ = cfg.run
    run_.ensure(run_.index_dir)
    if force:
        for p in run_.index_dir.glob("chunk_*.json"):
            p.unlink()
    shots = read_json(run_.shots)["shots"]
    by_id = {s["id"]: s for s in shots}
    file = upload_video(run_.proxy, run_.index_dir / "upload.json")
    chunks = _chunks(shots, cfg.index.chunk_s)
    with step(f"labelling {len(shots)} shots in {len(chunks)} windows with {cfg.models.index}"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda kc: _label_chunk(cfg, file, kc[1], kc[0]), enumerate(chunks)))

    labels: dict[int, dict] = {}
    dropped = 0
    for res in results:
        for lab in res["labels"]["shots"]:
            sid = lab["shot_id"]
            if sid not in by_id or sid not in res["shot_ids"]:
                continue
            events = []
            for ev in lab["events"]:
                t = _fix_time(ev["t"], by_id[sid], res["t0"])
                if t is None:
                    dropped += 1
                    continue
                events.append({**ev, "t": round(t, 3)})
            labels[sid] = {**lab, "events": events, **{k: by_id[sid][k] for k in ("start", "end", "start_frame", "end_frame")}}
    missing = [s["id"] for s in shots if s["id"] not in labels]
    n_events = sum(len(v["events"]) for v in labels.values())
    write_json(run_.index, {"model": cfg.models.index, "shots": [labels[k] for k in sorted(labels)],
                            "missing": missing, "dropped_events": dropped})
    log(f"{len(labels)}/{len(shots)} shots labelled, {n_events} events, {dropped} events outside their shot, "
        f"{len(missing)} shots unlabelled")
