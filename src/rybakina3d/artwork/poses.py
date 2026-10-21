"""Stage `artwork poses`: every sheet slot → a concrete clip, frame and camera (pure numpy, runs anywhere).

The figures are driven by the retargeted moveset, so poses are read from the character's own joint tracks
(anim/joints/<clip>.npz, Z-up metres, what `render` will show), not from the SMPL motion.

  idle                 no clip: `render` uses the rest pose with the arms brought down
  {stroke, at, offset} the best clip of that stroke at a moment of it, offset in frames, clamped to the clip:
                       contact = curation's contact_index; start = frame 0;
                       trophy  = the toss hand (left wrist for a right-hander) at its highest in the
                                 TROPHY_WINDOW_S before contact; it hangs there for ~10 frames while the
                                 racket arm cocks, so the last frame within TROPHY_APEX_M of the apex
                                 (racket up behind the head) is taken

Clip choice per stroke slot: label-valid clips first (curation label_ok), then the ones whose motion reads as
the stroke at contact (`reads_as`: on a two-handed backhand both hands are on the racket, on a forehand the
off hand has let go: broadcast labels swap the two often enough to matter on a sheet), then clips not yet on
the sheet, QA PASS before WARN before FAIL, then the curation score. The hero instead ranks by `dynamism`
(hand speed, reach, stance width, how low she sits) — the most dynamic readable stroke — among the moments
where her face shows (`head_tilt` within HERO_MAX_HEAD_TILT_DEG: a head bowed over the ball hides the face
under the visor). Slots are resolved
biggest box first, so the big figures get the best clips, and no clip appears twice at the same moment
(two figures of one clip must be MIN_MOMENT_GAP frames apart).

The atlas is every moveset clip in curation order at its contact frame, with the layout's atlas camera.
Output: artwork/poses.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..util import log, read_json, read_npz, write_json
from .layout import ATLAS, SLOTS, Slot

TROPHY_WINDOW_S = 2.0
TROPHY_APEX_M = 0.05          # toss hand this close to its highest point still counts as the apex
MIN_MOMENT_GAP = 10           # frames between two figures taken from the same clip
TWO_HANDED_GAP_M = 0.3        # wrists closer than this at contact = both hands on the racket
QA_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}
DYNAMIC_SLOTS = ("hero",)
HERO_MAX_HEAD_TILT_DEG = 35.0  # neck -> head line from vertical; beyond it the visor brim covers her eyes
# dynamism terms are divided by what a committed groundstroke shows, so each counts about 1
DYNAMISM_REF = {"speed": 5.0, "reach": 0.4, "stance": 0.7, "drop": 0.15}


@dataclass
class ClipPose:
    """One moveset clip: curation verdicts + the character's joint track."""

    id: str
    stroke: str
    label_ok: bool
    qa: str
    score: float
    contact: int
    positions: np.ndarray        # (L, J, 3) Z-up metres
    joints: dict[str, int]       # SMPL joint name -> column of `positions`
    fps: float

    @property
    def n(self) -> int:
        return len(self.positions)

    def joint(self, name: str) -> np.ndarray:
        return self.positions[:, self.joints[name]]


def trophy_frame(toss_z: np.ndarray, contact: int, fps: float) -> int:
    """The serve's trophy position: the end of the toss hand's apex in the window before contact."""
    lo = max(0, contact - int(round(TROPHY_WINDOW_S * fps)))
    if contact <= lo:
        return int(contact)
    window = toss_z[lo:contact]
    k = int(np.argmax(window))
    near = window >= window[k] - TROPHY_APEX_M
    while k + 1 < len(window) and near[k + 1]:
        k += 1
    return lo + k


def moment_frame(clip: ClipPose, at: str, offset: int = 0, handedness: str = "right") -> int:
    if at == "contact":
        base = clip.contact
    elif at == "start":
        base = 0
    elif at == "trophy":
        toss = "left_wrist" if handedness == "right" else "right_wrist"
        base = trophy_frame(clip.joint(toss)[:, 2], clip.contact, clip.fps)
    else:
        raise ValueError(f"unknown moment {at!r} (contact | trophy | start)")
    return int(np.clip(base + int(offset), 0, clip.n - 1))


def reads_as(clip: ClipPose, backhand: str = "two_handed") -> bool:
    """Whether the motion looks like its stroke at contact (labels from broadcast can swap fh / bh)."""
    k = clip.contact
    gap = float(np.linalg.norm(clip.joint("left_wrist")[k] - clip.joint("right_wrist")[k]))
    if clip.stroke == "backhand" and backhand == "two_handed":
        return gap < TWO_HANDED_GAP_M
    if clip.stroke == "forehand":
        return gap >= TWO_HANDED_GAP_M
    return True


def head_tilt(clip: ClipPose, frame: int) -> float:
    """Degrees the neck -> head line leans from vertical at `frame` (how far she bows her head)."""
    v = clip.joint("head")[frame] - clip.joint("neck")[frame]
    return float(np.degrees(np.arccos(np.clip(v[2] / max(np.linalg.norm(v), 1e-9), -1.0, 1.0))))


def dynamism(clip: ClipPose, frame: int) -> dict[str, float]:
    """How dynamic the pose at `frame` looks: hand speed, reach in front of the chest, stance, sit depth."""
    P = clip.positions
    rel = P - P[:, clip.joints["pelvis"]][:, None]   # the swing, without GVHMR's root noise
    lo, hi = max(0, frame - 1), min(clip.n - 1, frame + 1)
    hands = [clip.joints["left_wrist"], clip.joints["right_wrist"]]
    speed = float(np.mean([np.linalg.norm(rel[hi, h] - rel[lo, h]) for h in hands]) * clip.fps / max(hi - lo, 1))
    mid = P[frame, hands].mean(axis=0)
    reach = float(np.linalg.norm((mid - clip.joint("spine3")[frame])[:2]))
    stance = float(np.linalg.norm((clip.joint("left_ankle")[frame] - clip.joint("right_ankle")[frame])[:2]))
    pelvis_z = clip.joint("pelvis")[:, 2]
    drop = float(pelvis_z.max() - pelvis_z[frame])
    terms = {"speed": speed, "reach": reach, "stance": stance, "drop": drop}
    score = float(np.mean([terms[k] / DYNAMISM_REF[k] for k in terms]))
    return {**{k: round(v, 3) for k, v in terms.items()}, "score": round(score, 3)}


def load_clips(cfg: Config) -> list[ClipPose]:
    """Moveset clips in curation order (the ones with a retargeted joint track)."""
    run_ = cfg.run
    cur = read_json(run_.curation)
    rows = {r["id"]: r for r in cur["clips"]}
    clips = []
    for cid in cur["kept"]:
        path = run_.joints_dir / f"{cid}.npz"
        if cid not in rows or not path.exists():
            log(f"[yellow]{cid}: not in the moveset (no {path.name}), left off the sheet[/]")
            continue
        z = read_npz(path)
        r = rows[cid]
        positions = np.asarray(z["positions"], np.float64)
        clips.append(ClipPose(
            id=cid, stroke=r["stroke"], label_ok=bool(r.get("label_ok", r.get("ok"))), qa=r.get("qa", "PASS"),
            score=float(r.get("score", 0.0)), contact=int(np.clip(r["contact_index"], 0, len(positions) - 1)),
            positions=positions, joints={str(n): i for i, n in enumerate(z["smpl_joints"])},
            fps=float(z.get("fps", 25.0)),
        ))
    return clips


def _free(used: dict[str, list[int]], cid: str, frame: int) -> bool:
    return all(abs(frame - f) >= MIN_MOMENT_GAP for f in used.get(cid, []))


def resolve_slot(slot: Slot, clips: list[ClipPose], used: dict[str, list[int]], handedness: str = "right",
                 backhand: str = "two_handed") -> dict:
    """One slot → {slot, pose, clip, frame, camera, ...}; records the moment it takes in `used`."""
    out = {"slot": slot.name, "pose": "idle", "clip": None, "frame": None, "azimuth": slot.azimuth,
           "elevation": slot.elevation, "framing": slot.framing, "silhouette": slot.silhouette,
           "paint": slot.paint, "box": list(slot.box)}
    if slot.pose == "idle":
        return out
    spec = dict(slot.pose)
    stroke, at, offset = spec["stroke"], spec.get("at", "contact"), int(spec.get("offset", 0))
    out.update(pose="stroke", stroke=stroke, at=at, offset=offset)
    pool = [c for c in clips if c.stroke == stroke]
    if not any(c.label_ok for c in pool):
        if pool:
            log(f"[yellow]{slot.name}: no label-valid {stroke} clip, using a flagged one[/]")
    else:
        pool = [c for c in pool if c.label_ok]
    options = [(c, moment_frame(c, at, offset, handedness)) for c in pool]
    options = [(c, f) for c, f in options if _free(used, c.id, f)]
    if not options:
        log(f"[yellow]{slot.name}: no {stroke} clip left for this moment; slot left empty[/]")
        return {**out, "pose": None, "problem": f"no {stroke} clip available"}

    dynamic = slot.name in DYNAMIC_SLOTS
    dyn = {c.id: {**dynamism(c, f), "head_tilt": round(head_tilt(c, f), 1)} for c, f in options} if dynamic else {}

    def rank(opt: tuple[ClipPose, int]) -> tuple:
        c, _ = opt
        if dynamic:
            return (not reads_as(c, backhand), dyn[c.id]["head_tilt"] > HERO_MAX_HEAD_TILT_DEG, c.qa == "FAIL",
                    c.id in used, -dyn[c.id]["score"])
        return not reads_as(c, backhand), c.id in used, QA_RANK.get(c.qa, 1), -c.score

    clip, frame = min(options, key=rank)
    used.setdefault(clip.id, []).append(frame)
    out.update(clip=clip.id, frame=frame, contact_index=clip.contact, n_frames=clip.n,
               choice={"label_ok": clip.label_ok, "qa": clip.qa, "score": clip.score,
                       "reads_as_stroke": reads_as(clip, backhand), "candidates": len(options),
                       **({"dynamism": dyn[clip.id]} if dynamic else {})})
    return out


def resolve(slots: list[Slot], clips: list[ClipPose], handedness: str = "right",
            backhand: str = "two_handed") -> list[dict]:
    """Every slot, resolved biggest box first; returned in layout order."""
    def area(s: Slot) -> float:
        x0, y0, x1, y1 = s.box
        return (x1 - x0) * (y1 - y0)

    used: dict[str, list[int]] = {}
    done = {s.name: resolve_slot(s, clips, used, handedness, backhand) for s in sorted(slots, key=area, reverse=True)}
    return [done[s.name] for s in slots]


def atlas(clips: list[ClipPose]) -> list[dict]:
    return [{"clip": c.id, "stroke": c.stroke, "frame": c.contact, "contact_index": c.contact, "n_frames": c.n,
             "label_ok": c.label_ok, "qa": c.qa, "azimuth": ATLAS.azimuth, "elevation": ATLAS.elevation,
             "framing": ATLAS.framing} for c in clips]


def run(cfg: Config) -> dict:
    run_ = cfg.run
    if not run_.curation.exists():
        raise FileNotFoundError(f"no {run_.curation} (run `r3d curate`)")
    clips = load_clips(cfg)
    if not clips:
        raise RuntimeError(f"no retargeted clips in {run_.joints_dir} (run `r3d retarget`)")
    slots = resolve(SLOTS, clips, cfg.athlete.handedness, cfg.athlete.backhand)
    out = {"glb": str(run_.moveset_glb.relative_to(run_.root)), "fps": clips[0].fps,
           "handedness": cfg.athlete.handedness, "slots": slots, "atlas": atlas(clips)}
    write_json(run_.artwork_poses, out)
    for s in slots:
        if s["pose"] == "stroke":
            log(f"{s['slot']:9s} {s['clip']} f{s['frame']} ({s['at']}{s['offset']:+d}, {s['choice']['qa']}"
                f"{', dynamism ' + str(s['choice']['dynamism']['score']) if 'dynamism' in s['choice'] else ''})")
    log(f"{len(slots)} slots ({sum(s['pose'] == 'idle' for s in slots)} idle), {len(out['atlas'])} atlas figures "
        f"→ {run_.artwork_poses}")
    return out
