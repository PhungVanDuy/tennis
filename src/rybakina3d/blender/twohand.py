"""Left hand onto the racket in two-handed strokes: two-bone IK on the retargeted pose (runs INSIDE Blender).

numpy only: retarget.py imports this file, so it imports neither bpy, retarget.py nor the rybakina3d package, and
the unit tests import it without Blender. `left_hand_on_racket` takes the arrays retarget.retarget() has after
smoothing and FK and returns new pose-bone rotations for the left arm only (collar, upper arm, forearm, hand).

Spaces: armature space throughout (Blender's bone.matrix_local / pose_bone.matrix), Z up, metres. The pose of bone
b is pose[t, b] = pose[t, parent] @ rest[parent]^-1 @ rest[b] @ basis[t, b] (4x4), as in
retarget.forward_kinematics. The racket frame is racket.py's: origin at the butt cap, +Y along the shaft to the
tip, +Z the string normal on the right palm's side (attach_racket's eastern grip: the right palm presses on the
handle's -Z face, so the left palm takes the +Z face).

Per frame t:
  weight  w(t) from the distance between the wrists (running median of median_frames), with hysteresis: on from
          where they come within on_m until they are off_m apart, off from there until they come within on_m
          again (GVHMR's wrists stay 20-30 cm apart through some two-handed strokes: a smoothstep between the two
          left the hand half on for whole strokes); dips shorter than fill_frames filled (GVHMR's wrists drift
          apart for a few frames mid-stroke while both hands hold the racket), the edges carried ramp_out_s
          further out. Then lowered, never raised, into ramps (ramp_limit): each step costs frames by how far it
          moves the palm beyond its own path (at most max_blend_step_m a frame), the fingertips and the elbow (at
          most max_blend_swing_m) and how fast it closes the fingers as the hand comes down (at most
          close_ramp_step a frame), measured level by level along the blend, so a ramp slows only where it has far
          to go; at least blend_frames. The weight stays low where the arms fly apart (a serve's toss lets go of
          the throat: the hand keeps to its own path there); where the full IK falls short of the handle
          (short_on_m .. short_off_m) the hand only comes as far as the hover (see blend). No stroke label or
          contact frame is needed: one clip can hold several strokes.
  target  the left palm on the handle just above the right hand (handle_s from the butt) while the racket head
          moves fast relative to the chest, at throat_s while it is slow (ready position), sliding between the
          two; the palm on the face opposite the right palm, the handle crossing it grip_angle_deg from the hand's
          long axis towards the thumb (or along a measured handle direction). The hand is rigid on the racket
          there, but free to roll round the handle (the fingers wrap it the same way). By default throat_s is
          handle_s: a hand further up reaches the racket's open throat (racket.py splits it at 0.255 m), where its
          palm and fingers enter the throat arms.
  collar  where the target is out of the arm's reach, the collar turns the shoulder towards it, up to
          max_collar_deg, smoothed over time so it starts before it is needed and never jumps.
  IK      analytic two-bone IK; the elbow's included angle is capped at max_elbow_deg with a soft approach
          (soft_m), so the arm never locks straight. The bend plane starts from the motion's own elbow (the
          pole) and swivels about the shoulder-wrist line, together with the palm's roll round the handle, to
          the most comfortable arm over the clip (comfortable_arm: wrist bend, twist at shoulder / elbow / wrist,
          elbow clear of the torso, reach, smooth in time). GVHMR never saw the left hand on the racket, so
          its elbow plane alone bends the wrist 80-110 deg on real clips. The forearm then rolls about its own
          axis so the elbow and the wrist share the arm's twist (twist_balance).
  blend   in the racket's frame, not by slerping joint rotations (far-apart solutions swept the hand through
          the body, and flipped where they were half a turn apart): at weight w the hand bone is put on an
          approach path and the arm solved for it as above, the collar, swivel, elbow cap and twist taken w of the
          way (w = 0 is the motion, w = 1 the full IK). Over the first turn_share of w the palm goes straight from
          the motion's palm to hover_m out from its grip spot, the hand turning into the grip on the way (its wrist
          bent at most max_wrist_bend_deg off the forearm; the cap fades out as the hand comes down); over the rest
          it comes straight down onto the handle (with even_share the turn's share per stretch of frames is its
          share of the palm's path). The hand's quaternions are kept continuous over each stretch of frames near
          the racket, so the path never flips between frames. The straight path may cross the racket: the push
          takes the hand round it (a path round the handle at the hover's distance was two to three times longer
          than the straight one, and its ramps as much slower: the hand rarely reached the handle in a stroke).
  push    with the hand's skin points (hand_points), the hand is moved off the racket square to the shaft
          wherever its skin would go in (by more than push_margin_m with the palm on its grip spot), the motion's
          own hand too (GVHMR's left hand passes through the throat in some ready positions); push_off picks the
          push per frame and its direction over the clip, so it ramps in and out and does not swap sides. It fades
          out as the hand comes down from the hover (turned into the grip, the hand clears the handle there: a push
          would lift the closing fingers into it).
  repair  where the palm still steps more than max_pop_m (the elbow max_elbow_pop_m) beyond the motion's and the
          full IK's steps (the blend's speed varies along the path, the push moves the hand too), those ramps slow
          down and the hand is placed again, repair_rounds times at most.
  grip    how far the left fingers close (0 relaxed, 1 the grip solved on the handle), from the palm's lift off
          its grip spot on the handle (its grip point's distance from the axis; the relaxed fingers curl into the
          handle within ~4 cm of it, the grip's poke through it from ~1 cm out): closed up to close_near_m[0],
          close_hold from close_near_m[1] to close_open_m[0], relaxed from close_open_m[1]; times how well the hand's
          handle direction matches the shaft (fit_on_deg / fit_off_deg), whether the palm is low enough on the
          shaft (grip_top_s, open over grip_open_m: the closed hand reaches ~10 cm up the shaft from the palm
          centre) and how far the approach has got (w over the turn's share): the fingers close as the hand comes
          down. Then it changes at most max_close_step a frame, closing early and opening late (where that leaves
          skin in the racket, the push takes the hand out).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Left palm when the character stage measured none (hand_rig.json): this far along the hand bone (its long axis)
# from the wrist, and this far out along the palm normal (Naomi's palm skin sits 2.8-3.1 cm off the bone axis).
PALM_OFFSET_M = 0.075
PALM_SKIN_M = 0.03
HEAD_FROM_BUTT_M = 0.52   # racket-head centre (racket.py: 0.685 m long, 0.33 m head)
FINGERTIP_PAST_PALM_M = 0.08   # the blend's speed is also measured at a point this far past the palm centre


@dataclass(frozen=True)
class LeftArm:
    """Bone indices (into the rig's bone list) of the chain the IK turns, and of the hand holding the racket."""

    upper: int
    fore: int
    hand: int
    right_hand: int
    collar: int | None = None   # without one, targets out of the arm's reach stay out of reach


@dataclass(frozen=True)
class Params:
    """Tunables of left_hand_on_racket (see the module docstring); the defaults were tuned on the 5 s clip
    (configs/naomi_5s_grip.yaml) and the 39-clip moveset (configs/naomi.yaml)."""

    on_m: float = 0.20            # wrists this close (m): the left hand goes on the racket ...
    off_m: float = 0.30           # ... and stays on until they are this far apart
    median_frames: int = 5        # running median of the wrist distance
    fill_frames: int = 9          # dips of the weight shorter than this are filled (closing: running max, then min)
    blend_frames: int = 4         # shortest on / off ramp
    ramp_out_s: float = 0.2       # the ramps may reach this far past the frames where the wrists are close
    max_blend_step_m: float = 0.02    # most a ramp may move the palm per frame beyond its own path ...
    max_blend_swing_m: float = 0.025  # ... and the fingertips or the elbow
    close_ramp_step: float = 0.6  # ... and close or open the fingers as the hand comes down onto the handle
    blend_samples: int = 11       # weights along the blend its speed is measured at
    hover_m: float = 0.05         # the palm turns into the grip this far out from its spot on the handle ...
    turn_share: float = 0.6       # ... over this share of the blend, then comes down onto it; with even_share,
    even_share: bool = True       # per stretch of frames, the share of the palm's path the turn takes (in range)
    share_range: tuple[float, float] = (0.3, 0.9)
    max_wrist_bend_deg: float = 70.0   # on the way to the grip the wrist bends at most this far off the forearm
    max_pop_m: float = 0.014      # the palm's step may exceed the motion's and the full IK's by this much
    max_elbow_pop_m: float = 0.025   # ... and the elbow's by this much; beyond either,
    repair_rounds: int = 6        # those ramps slow down (x repair_factor) and the hand is placed again, this many
    repair_factor: float = 0.6    # times at most
    # the push off the racket (with the hand's skin points given; push_off)
    push_margin_m: float = 0.003  # skin this deep in the racket is let be with the palm on its grip spot (the palm
                                  # pads sink ~3 mm into the handle there), none from close_near_m[1] off it
    max_push_m: float = 0.10      # pushes tried in push_step_m steps up to this far
    push_step_m: float = 0.005
    push_hold_s: float = 0.12     # each push held and averaged over this long either side (its turns too)
    push_rounds: int = 2          # the push measured this many times, each with the fingers the last one left
    push_directions: int = 16     # directions square to the shaft it may take, chosen by cost (Viterbi):
    push_inside_cost: float = 20.0     # per skin point left inside ...
    push_cost: float = 0.05            # ... per cm^2 of push ...
    push_jump_cost: float = 20.0       # ... per cm^2 the push moves the hand in a frame beyond what the blend
    push_free_m: float = 0.003         # leaves of max_pop_m there (at least push_free_m)
    handle_s: float = 0.14        # left palm centre from the butt in a stroke: its little finger on the right
                                  # hand's index (m)
    throat_s: float = 0.14        # ... in the ready position (0.27: on the throat; see the module docstring)
    slow_mps: float = 1.5         # racket-head speed relative to the chest: at or below, slow (throat) ...
    fast_mps: float = 3.0         # ... at or above, fast (handle)
    hold_s: float = 0.25          # a fast phase keeps the handle target this long either side of it ...
    slide_s: float = 0.3          # ... and the palm slides between the two targets over this long
    grip_angle_deg: float = 45.0  # handle across the left palm, from the hand's long axis towards the thumb
    handle_above_palm_m: float = 0.015   # handle axis above the palm skin: 17 mm radius with overgrip, 2 mm sunk
    max_elbow_deg: float = 165.0  # cap on the elbow's included angle (180 = straight)
    soft_m: float = 0.01          # the reach eases into that cap over this distance
    twist_balance: float = 0.5    # share of the wrist-minus-elbow twist the forearm's roll takes (0.5: even)
    # comfortable_arm: the elbow swivel and the palm's roll about the handle, chosen by cost (deg^2; wrist bend: 1)
    max_roll_deg: float = 60.0    # the palm may roll this far either way round the handle from the opposite face
    twist_cost: float = 0.35      # per deg of twist the elbow and wrist carry together
    shoulder_twist_cost: float = 1.0   # per deg of the upper arm's twist (0.5 swung it 50 deg in 4 frames)
    swivel_prior: float = 0.15    # per deg of swivel off the motion's own elbow plane
    roll_prior: float = 0.3       # per deg of roll
    body_cost: float = 10.0       # per cm the elbow comes nearer the torso axis than min_torso_m (or the motion)
    min_torso_m: float = 0.18
    reach_cost: float = 10.0      # per cm the wrist stays short of its target
    swivel_smooth: float = 1.5    # per deg of swivel change from one frame to the next
    roll_smooth: float = 1.5      # per deg of roll change
    swivel_step_deg: float = 10.0
    roll_step_deg: float = 15.0
    max_collar_deg: float = 16.0  # most the collar turns to extend the reach (12 left a take-back 2.4 cm short)
    short_on_m: float = 0.02      # the full IK this far short of the handle (after the collar): the hand still goes
    short_off_m: float = 0.04     # ... this far: it hovers in the grip hover_m out (half-closed fingers short of the
                                  # handle would hit it)
    collar_hold_s: float = 0.2    # the collar turn is held and averaged over this long either side
    # the left fingers' closing (the returned grip track), from the palm's lift off its grip spot (m)
    close_near_m: tuple[float, float] = (0.0, 0.005)    # lifted up to the first: fully closed, from the second ...
    close_hold: float = 0.75      # ... this far closed (the fingertips clear the handle) ...
    close_open_m: tuple[float, float] = (0.025, 0.045)   # ... out to the first lift, relaxed from the second
    fit_on_deg: float = 5.0       # the hand's handle direction this close to the shaft: closing as above ...
    fit_off_deg: float = 20.0     # ... this far off: relaxed
    grip_top_s: float = 0.16      # palm centre at most this far up the shaft: closed (the hand stays off the throat)
    grip_open_m: float = 0.06     # ... relaxed this much further up
    max_close_step: float = 0.35  # most the closing changes from one frame to the next (closing early, opening late)


# --------------------------------------------------------------------------------------------- rotations
def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def axis_angle(axis: np.ndarray, angle: np.ndarray | float) -> np.ndarray:
    """(...,3) axes, (...) angles (rad) -> (...,3,3) rotations (Rodrigues)."""
    k = _unit(axis)
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    o = np.zeros_like(x)
    K = np.stack([o, -z, y, z, o, -x, -y, x, o], -1).reshape(k.shape[:-1] + (3, 3))
    a = np.asarray(angle, np.float64)[..., None, None]
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(...,3) -> (...,3,3): the minimal rotation taking direction u onto direction v."""
    u, v = _unit(u), _unit(v)
    axis = np.cross(u, v)
    s, c = np.linalg.norm(axis, axis=-1), np.sum(u * v, -1)
    helper = np.where((np.abs(u[..., :1]) < 0.9), [1.0, 0, 0], [0, 1.0, 0])
    axis = np.where((s < 1e-9)[..., None], np.cross(u, helper), axis)   # opposite directions: any perpendicular
    return axis_angle(axis, np.arctan2(s, c))


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,4) unit quaternions (w, x, y, z), w >= 0."""
    R = np.asarray(R, np.float64)
    m = {(i, j): R[..., i, j] for i in range(3) for j in range(3)}
    tr = np.stack([1 + m[0, 0] + m[1, 1] + m[2, 2], 1 + m[0, 0] - m[1, 1] - m[2, 2],
                   1 - m[0, 0] + m[1, 1] - m[2, 2], 1 - m[0, 0] - m[1, 1] + m[2, 2]], -1)
    case = np.argmax(tr, -1)
    r = 0.5 * np.sqrt(np.maximum(np.take_along_axis(tr, case[..., None], -1)[..., 0], 1e-12))
    f = 0.25 / r
    cands = [
        (r, (m[2, 1] - m[1, 2]) * f, (m[0, 2] - m[2, 0]) * f, (m[1, 0] - m[0, 1]) * f),
        ((m[2, 1] - m[1, 2]) * f, r, (m[0, 1] + m[1, 0]) * f, (m[0, 2] + m[2, 0]) * f),
        ((m[0, 2] - m[2, 0]) * f, (m[0, 1] + m[1, 0]) * f, r, (m[1, 2] + m[2, 1]) * f),
        ((m[1, 0] - m[0, 1]) * f, (m[0, 2] + m[2, 0]) * f, (m[1, 2] + m[2, 1]) * f, r),
    ]
    q = np.zeros(R.shape[:-2] + (4,))
    for i, cand in enumerate(cands):
        sel = case == i
        q[sel] = np.stack(cand, -1)[sel]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    q = _unit(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(q.shape[:-1] + (3, 3))


def slerp(R0: np.ndarray, R1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """(...,3,3) rotations R0 -> R1 along the shorter arc, t (...) in [0, 1]."""
    q0, q1 = quat_from_matrix(R0), quat_from_matrix(R1)
    dot = np.sum(q0 * q1, -1)
    q1 = np.where(dot[..., None] < 0, -q1, q1)
    theta = np.arccos(np.clip(np.abs(dot), -1.0, 1.0))
    t = np.broadcast_to(np.asarray(t, np.float64), theta.shape)
    sin = np.sin(theta)
    small = sin < 1e-6
    a = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1.0, sin))
    b = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin))
    return matrix_from_quat(a[..., None] * q0 + b[..., None] * q1)


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """(...,4) unit quaternions q0 -> q1 as given (no shorter-arc flip: over a time series whose signs are kept
    continuous the path then never jumps to the other arc), t (...) in [0, 1]."""
    dot = np.clip(np.sum(q0 * q1, -1), -1.0, 1.0)
    theta = np.arccos(dot)
    t = np.broadcast_to(np.asarray(t, np.float64), theta.shape)
    sin = np.sin(theta)
    small = sin < 1e-6
    a = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1.0, sin))
    b = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin))
    return _unit(a[..., None] * q0 + b[..., None] * q1)


def twist_bend_deg(Rb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(L,3,3) pose-bone rotations -> (twist about the bone's own Y axis in (-180, 180], bend of its Y axis off the
    rest direction), in degrees: for the hand's basis, the wrist's roll and its flexion / deviation together."""
    q = quat_from_matrix(Rb)
    twist = (np.degrees(2 * np.arctan2(q[..., 2], q[..., 0])) + 180.0) % 360.0 - 180.0
    return twist, np.degrees(np.arccos(np.clip(Rb[..., 1, 1], -1, 1)))


def _frames(M: np.ndarray) -> np.ndarray:
    """4x4 matrices with their rotation columns normalized (no scale)."""
    M = np.array(M, np.float64)
    M[..., :3, :3] /= np.linalg.norm(M[..., :3, :3], axis=-2, keepdims=True)
    return M


# ------------------------------------------------------------------------------------------ time series
def smoothstep(x: np.ndarray) -> np.ndarray:
    """0 at x <= 0, 1 at x >= 1, 3x^2 - 2x^3 between (fingers.smoothstep_pm1 spans -1 .. 1 instead)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _window(x: np.ndarray, half: int) -> np.ndarray:
    """(L, 2*half+1) sliding windows over x, the ends padded with the end values."""
    return np.lib.stride_tricks.sliding_window_view(np.pad(x, half, mode="edge"), 2 * half + 1)


def running(x: np.ndarray, frames: int, how: str) -> np.ndarray:
    """Centred running median / max / min / mean of a (L,) series over `frames` (rounded up to odd)."""
    half = max(int(frames), 1) // 2
    if half == 0 or len(x) == 0:
        return np.asarray(x, np.float64).copy()
    fn = {"median": np.median, "max": np.max, "min": np.min, "mean": np.mean}[how]
    return fn(_window(np.asarray(x, np.float64), half), -1)


def fill_dips(w: np.ndarray, frames: int) -> np.ndarray:
    """(L,) w with every dip narrower than `frames` filled up to its edges (running max, then running min over the
    same window: a morphological closing). Wider dips, the ramps and the ends keep their values."""
    return running(running(w, frames, "max"), frames, "min")


def two_hand_weight(wrist_dist: np.ndarray, on_m: float = 0.20, off_m: float = 0.30,
                    median_frames: int = 5) -> np.ndarray:
    """(L,) raw left-hand-on-racket weight: 1 with the wrists on_m or closer, 0 at off_m or farther (smoothstep
    between), on the running median of the distance (single-frame GVHMR pops do not toggle it)."""
    d = running(np.asarray(wrist_dist, np.float64), median_frames, "median")
    return smoothstep((off_m - d) / (off_m - on_m))


def two_hand_state(wrist_dist: np.ndarray, on_m: float = 0.20, off_m: float = 0.30,
                   median_frames: int = 5) -> np.ndarray:
    """(L,) 1 where the left hand belongs on the racket, else 0, from the running median of the wrist distance with
    hysteresis: on from where it comes within on_m until it reaches off_m, off from there until it comes within on_m
    again. A clip starts on where its first distance is nearer on_m than off_m."""
    d = running(np.asarray(wrist_dist, np.float64), median_frames, "median")
    out = np.zeros(len(d))
    on = bool(len(d)) and d[0] <= (on_m + off_m) / 2
    for t, x in enumerate(d):
        on = x < off_m if on else x <= on_m
        out[t] = on
    return out


def limit_ramps(w: np.ndarray, rate: np.ndarray) -> np.ndarray:
    """w (L,) in [0, 1]; rate (L-1,) the largest change allowed from frame t to t+1. Returns the largest series
    at or below w that respects the rates: every ramp lies inside the frames where w is up (it starts rising
    where w does and has come down where w has), never steeper than allowed; a window too short for its ramps
    keeps a lower peak."""
    lo = np.array(w, np.float64)
    for t in range(1, len(w)):
        lo[t] = min(lo[t], lo[t - 1] + rate[t - 1])
    for t in range(len(w) - 2, -1, -1):
        lo[t] = min(lo[t], lo[t + 1] + rate[t])
    return lo


def advance(x: float, cost: np.ndarray, levels: np.ndarray) -> float:
    """The weight one frame takes a ramp to from x, where going from levels[k] to levels[k+1] costs cost[k] frames
    per unit of weight (levels ascending, its last the top)."""
    left = 1.0
    k = int(np.clip(np.searchsorted(levels, x, "right") - 1, 0, len(cost) - 1))
    while k < len(cost):
        need = (levels[k + 1] - x) * cost[k]
        if need >= left:
            return x + left / cost[k]
        left -= need
        x = levels[k + 1]
        k += 1
    return float(levels[-1])


def ramp_limit(w: np.ndarray, cost: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """limit_ramps with the allowed change set level by level: cost (L-1, K) frames per unit of weight from
    levels[k] to levels[k+1] ((K+1,) ascending) between frames t and t+1, so a ramp slows only over the stretch of
    the blend that has far to go. The largest series at or below w (L,) that keeps to it."""
    lo = np.array(w, np.float64)
    for t in range(1, len(w)):
        lo[t] = min(lo[t], advance(lo[t - 1], cost[t - 1], levels))
    for t in range(len(w) - 2, -1, -1):
        lo[t] = min(lo[t], advance(lo[t + 1], cost[t], levels))
    return lo


def widen(w: np.ndarray, frames: int) -> np.ndarray:
    """(L,) w with every edge carried `frames` further out, falling linearly to 0 (w(t + d) x (1 - |d| / (frames
    + 1)) at most): room for ramps outside the frames where w is up; inside them w is kept."""
    out = np.array(w, np.float64)
    for d in range(1, max(int(frames), 0) + 1):
        k = 1 - d / (frames + 1)
        out[d:] = np.maximum(out[d:], k * w[:-d])
        out[:-d] = np.maximum(out[:-d], k * w[d:])
    return out


def runs(on: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of True in a (L,) mask."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(on, np.int8), 0]))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def unwrap_runs(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    """(L,) angles (rad) unwrapped over each run of `on`, each run then shifted by whole turns so that its mean
    lies in [-pi, pi): continuous within a run, as near zero as it can be. Frames off the runs keep x."""
    x = np.array(x, np.float64)
    for a, b in runs(on):
        u = np.unwrap(x[a:b])
        x[a:b] = u - 2 * np.pi * np.round(u.mean() / (2 * np.pi))
    return x


def continuous_quats(q: np.ndarray, on: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """(L,4) quaternions with their signs made continuous over each run of `on`, each run then signed so that it
    lies on ref's (L,4) side on average: slerp_quat from them to ref then takes the same arc on neighbouring
    frames (and the shorter one where it can)."""
    q = np.array(q, np.float64)
    for a, b in runs(on):
        for t in range(a + 1, b):
            if q[t] @ q[t - 1] < 0:
                q[t] = -q[t]
        if np.sum(q[a:b] * ref[a:b]) < 0:
            q[a:b] = -q[a:b]
    return q


def handle_position(racket: np.ndarray, chest: np.ndarray, fps: float, p: Params) -> np.ndarray:
    """(L,) left palm target, metres from the butt along the shaft: handle_s while the racket head moves fast
    relative to the chest (a stroke: take-back to finish), throat_s while it is slow (ready position)."""
    head = racket[:, :3, 3] + racket[:, :3, 1] * HEAD_FROM_BUTT_M
    rel = head - chest
    speed = np.linalg.norm(np.gradient(rel, axis=0), axis=-1) * fps if len(rel) > 1 else np.zeros(len(rel))
    fast = smoothstep((speed - p.slow_mps) / (p.fast_mps - p.slow_mps))
    fast = running(fast, 2 * round(p.hold_s * fps) + 1, "max")       # the backswing's turn-round stays "fast"
    fast = running(fast, 2 * round(p.slide_s * fps / 2) + 1, "mean")
    return p.throat_s + fast * (p.handle_s - p.throat_s)


# -------------------------------------------------------------------------------------------- grip + IK
def default_palm(rest: np.ndarray, hand: int, other_hand: int) -> tuple[np.ndarray, np.ndarray]:
    """(palm centre on the skin, outward palm normal) in the hand bone's frame, for a rig whose palm was not
    measured: PALM_OFFSET_M along the bone (its long axis) and PALM_SKIN_M out along a normal taken as the
    horizontal direction to the other hand in the rest pose, orthogonal to the bone (palms face the body at rest;
    attach_racket's heuristic: 16-18 deg off Naomi's mesh)."""
    R = rest[hand, :3, :3]
    to = rest[other_hand, :3, 3] - rest[hand, :3, 3]
    to[2] = 0.0
    n = R.T @ to
    n[1] = 0.0                                   # orthogonal to the bone's long axis (Y)
    n = _unit(n)
    return np.array([0.0, PALM_OFFSET_M, 0.0]) + PALM_SKIN_M * n, n


def left_grip_frame(palm_centre: np.ndarray, palm_normal: np.ndarray, handle_dir: np.ndarray | None = None,
                    grip_angle_deg: float = 45.0, above_palm_m: float = 0.015) -> np.ndarray:
    """(4,4) the racket frame at the left palm, in the left-hand bone's frame: origin on the handle axis above_palm_m
    out from the palm centre, +Z = -palm normal (the left palm on the face opposite the right one), +Y along the
    handle to the head: handle_dir projected onto the palm plane when given (e.g. a solved grip), else
    grip_angle_deg from the hand's long axis (bone Y) towards the thumb (left hand: thumb = normal x long axis)."""
    n = _unit(palm_normal)
    if handle_dir is None:
        y = np.array([0.0, 1.0, 0.0])
        a = _unit(y - (y @ n) * n)
        th = math.radians(grip_angle_deg)
        d = math.cos(th) * a + math.sin(th) * np.cross(n, a)
    else:
        hd = np.asarray(handle_dir, np.float64)
        d = _unit(hd - (hd @ n) * n)
    z = -n
    G = np.eye(4)
    G[:3, 0], G[:3, 1], G[:3, 2] = np.cross(d, z), d, z
    G[:3, 3] = np.asarray(palm_centre, np.float64) + n * above_palm_m
    return G


def _elbow_circle(S: np.ndarray, E: np.ndarray, W: np.ndarray, target: np.ndarray, pole: np.ndarray,
                  max_elbow_deg: float | np.ndarray, soft_m: float | np.ndarray) -> tuple[np.ndarray, ...]:
    """The circle the elbow may sit on to put the wrist at (or towards) the target: shoulder->target unit u, the
    reached distance d (capped, soft; the cap and softness scalars or per frame), the circle's centre offset x
    along u and radius h, and the unit directions `ref` (towards the pole, in the plane normal to u) and
    `side` = u x ref. The pole is taken off the arm's own shoulder-wrist line first: with the motion's elbow as the
    pole, ref is then the motion's bend direction, which stays put as the target moves off the motion's wrist (the
    pole's offset from S->target alone is mostly the arm's lean off that line, which turns over as the line
    crosses the arm)."""
    l1, l2 = np.linalg.norm(E - S, axis=-1), np.linalg.norm(W - E, axis=-1)
    v = target - S
    dist = np.linalg.norm(v, axis=-1)
    u = _unit(v)
    cap = np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * np.cos(np.radians(max_elbow_deg)))
    soft = np.broadcast_to(np.asarray(soft_m, np.float64), dist.shape)
    knee = cap - soft
    eased = knee + soft * (1 - np.exp(-np.maximum(dist - knee, 0.0) / np.maximum(soft, 1e-12)))
    d = np.where(soft > 0, np.where(dist > knee, eased, dist), np.minimum(dist, cap))
    d = np.maximum(d, np.abs(l1 - l2) + 1e-6)
    x = (l1 * l1 - l2 * l2 + d * d) / (2 * d)
    h = np.sqrt(np.maximum(l1 * l1 - x * x, 0.0))
    a = _unit(W - S)
    bend = (pole - S) - np.sum((pole - S) * a, -1, keepdims=True) * a
    ref = bend - np.sum(bend * u, -1, keepdims=True) * u
    ok = np.linalg.norm(ref, axis=-1) > 1e-6
    if not ok.all():
        if ok.any():   # carry the last good bend plane forward (and the first one back)
            idx = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
            idx[idx < 0] = int(np.argmax(ok))
            ref = ref[idx]
        else:
            ref = np.cross(u, [0.0, 0.0, 1.0])
            ref = np.where(np.linalg.norm(ref, axis=-1, keepdims=True) > 1e-6, ref, np.cross(u, [1.0, 0, 0]))
        ref = ref - np.sum(ref * u, -1, keepdims=True) * u
    ref = _unit(ref)
    return u, d, x, h, ref, np.cross(u, ref)


def _arm_turns(S: np.ndarray, E: np.ndarray, W: np.ndarray, E2: np.ndarray, W2: np.ndarray) -> tuple[np.ndarray, ...]:
    """World rotations (L,3,3) taking the upper arm S->E onto S->E2 and the forearm E->W onto E2->W2. Each bone
    turns from (its direction, the arm plane's normal) to the new ones, so the elbow keeps bending about the same
    hinge axis of the forearm (no twist enters the elbow) and the upper arm rolls with the swivel. A nearly
    straight arm has no plane: there the turn fades to the minimal one."""
    l1, l2 = np.linalg.norm(E - S, axis=-1), np.linalg.norm(W - E, axis=-1)
    n0, n2 = np.cross(E - S, W - E), np.cross(E2 - S, W2 - E2)
    bent = np.minimum(np.linalg.norm(n0, axis=-1), np.linalg.norm(n2, axis=-1)) / np.maximum(l1 * l2, 1e-12)
    keep = smoothstep((bent - 0.02) / 0.08)                       # sin(elbow bend): 0 below ~1 deg, 1 from ~6 deg
    R1 = slerp(rotation_between(E - S, E2 - S), _turn(E - S, n0, E2 - S, n2), keep)
    minimal = rotation_between(np.einsum("lij,lj->li", R1, W - E), W2 - E2) @ R1
    return R1, slerp(minimal, _turn(W - E, n0, W2 - E2, n2), keep)


def _turn(d0: np.ndarray, n0: np.ndarray, d1: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """(L,3,3) rotation taking direction d0 onto d1 and the plane normal n0 onto n1 (both orthogonalised)."""
    def frame(d: np.ndarray, n: np.ndarray) -> np.ndarray:
        d = _unit(d)
        n = _unit(n - np.sum(n * d, -1, keepdims=True) * d)
        return np.stack([d, n, np.cross(d, n)], -1)
    return frame(d1, n1) @ np.swapaxes(frame(d0, n0), -1, -2)


def two_bone_ik(S: np.ndarray, E: np.ndarray, W: np.ndarray, target: np.ndarray, pole: np.ndarray,
                max_elbow_deg: float | np.ndarray = 180.0, soft_m: float | np.ndarray = 0.0,
                swivel: np.ndarray | None = None) -> tuple[np.ndarray, ...]:
    """Analytic two-bone IK per frame, (L,3) shoulder / elbow / wrist heads and wrist target.

    The elbow stays in the plane through the shoulder and the target on the pole's side (pole: a point, e.g. the
    motion's own elbow; where it is on the shoulder-target line, the previous frame's bend plane), turned by
    `swivel` (L,) rad about the shoulder->target axis when given. The reach is capped at the elbow's included
    angle max_elbow_deg, which it approaches softly over the last soft_m (both scalars or (L,)).
    Returns (E2, W2, R_upper, R_fore, short): new heads, the world rotations that pre-multiply the upper arm's and
    the forearm's current ones (see _arm_turns), and how far (m) each wrist stays from its target."""
    S, E, W, target, pole = (np.atleast_2d(np.asarray(x, np.float64)) for x in (S, E, W, target, pole))
    u, d, x, h, ref, side = _elbow_circle(S, E, W, target, pole, max_elbow_deg, soft_m)
    if swivel is not None:
        ref = np.cos(swivel)[:, None] * ref + np.sin(swivel)[:, None] * side
    E2 = S + u * x[:, None] + ref * h[:, None]
    W2 = S + u * d[:, None]
    R1, R12 = _arm_turns(S, E, W, E2, W2)
    return E2, W2, R1, R12, np.linalg.norm(W2 - target, axis=-1)


def viterbi(cost: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Index (L,) of the cheapest path through per-frame state costs (L,K), a step from state j to state k
    adding trans[j, k] (K,K), or trans[t - 1, j, k] into frame t with per-frame costs (L-1,K,K)."""
    acc, back = cost[0].copy(), np.zeros(cost.shape, int)
    for t in range(1, len(cost)):
        tot = acc[:, None] + (trans[t - 1] if trans.ndim == 3 else trans)
        back[t] = np.argmin(tot, 0)
        acc = tot[back[t], np.arange(cost.shape[1])] + cost[t]
    path = np.zeros(len(cost), int)
    path[-1] = int(np.argmin(acc))
    for t in range(len(cost) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


@dataclass
class Arm:
    """The left arm before the IK, as comfortable_arm needs it ((L,...) arrays, armature space)."""

    S: np.ndarray            # shoulder (upper-arm head), elbow and wrist heads, after the collar turn
    E: np.ndarray
    W: np.ndarray
    R_collar: np.ndarray     # world rotations of the collar (after its turn), upper arm and forearm
    R_upper: np.ndarray
    R_fore: np.ndarray
    rel: dict                # "upper" / "fore" / "hand": rest rotation relative to the parent bone
    hips: np.ndarray         # torso axis: the root bone's head ...
    chest: np.ndarray        # ... to the head of the bone the arm hangs from


def _torso_distance(point: np.ndarray, hips: np.ndarray, chest: np.ndarray) -> np.ndarray:
    """Distance from points (L,K,3) to the torso axis hips -> chest (L,3), carried 30% past the chest."""
    axis = chest - hips
    t = np.sum((point - hips[:, None]) * axis[:, None], -1) / np.maximum(np.sum(axis * axis, -1), 1e-12)[:, None]
    return np.linalg.norm(point - (hips[:, None] + axis[:, None] * np.clip(t, 0.0, 1.3)[..., None]), axis=-1)


def comfortable_arm(arm: Arm, handle: np.ndarray, grip_inv: np.ndarray, p: Params) -> tuple[np.ndarray, ...]:
    """Elbow swivel (L,) rad (about the shoulder->wrist-target axis; 0 = the motion's own elbow plane) and palm roll
    about the handle (L,) rad (0 = the face opposite the right palm) that hold the handle most comfortably, and the
    per-frame cost of that choice. handle (L,4,4): the racket frame moved to the palm's point on the shaft;
    grip_inv: the left-hand bone in the grip frame (left_grip_frame inverted).

    Cost per frame and candidate, in degrees squared: the wrist's bend off its rest direction, (twist_cost x the
    twist the elbow and wrist carry together; it is shared between them afterwards), (shoulder_twist_cost x the
    upper arm's twist), (swivel_prior x the swivel), (roll_prior x the roll), (body_cost x every cm the elbow comes
    nearer the torso axis than min(min_torso_m, where the motion has it)) and (reach_cost x every cm the wrist
    stays short of its target). Over the clip, (swivel_smooth x the swivel's change per frame)^2 and the same for
    the roll are added; the cheapest path is found exactly (Viterbi) on a swivel_step_deg x roll_step_deg grid,
    then the grid's steps are smoothed out."""
    L = len(arm.S)
    sw = np.radians(np.arange(-180.0, 180.0, p.swivel_step_deg))
    rolls = np.radians(np.arange(-p.max_roll_deg, p.max_roll_deg + 1e-9, p.roll_step_deg))
    Ks = len(sw)

    def rep(X: np.ndarray) -> np.ndarray:
        return np.repeat(X, Ks, 0)

    floor = np.minimum(p.min_torso_m, _torso_distance(arm.E[:, None], arm.hips, arm.chest)[:, 0])
    costs = []
    for roll in rolls:
        spin = np.eye(4)
        spin[:3, :3] = axis_angle(np.array([0.0, 1.0, 0.0]), roll)
        target = handle @ spin @ grip_inv
        u, d, x, h, ref, side = _elbow_circle(arm.S, arm.E, arm.W, target[:, :3, 3], arm.E, p.max_elbow_deg,
                                              p.soft_m)
        pv = np.cos(sw)[None, :, None] * ref[:, None] + np.sin(sw)[None, :, None] * side[:, None]
        E2 = (arm.S + u * x[:, None])[:, None] + pv * h[:, None, None]                # (L, Ks, 3)
        W2 = arm.S + u * d[:, None]
        R1, R12 = _arm_turns(rep(arm.S), rep(arm.E), rep(arm.W), E2.reshape(L * Ks, 3), rep(W2))
        up, fo = R1 @ rep(arm.R_upper), R12 @ rep(arm.R_fore)
        shoulder = twist_bend_deg(arm.rel["upper"].T @ np.swapaxes(rep(arm.R_collar), -1, -2) @ up)[0]
        elbow = twist_bend_deg(arm.rel["fore"].T @ np.swapaxes(up, -1, -2) @ fo)[0]
        wrist, bend = twist_bend_deg(arm.rel["hand"].T @ np.swapaxes(fo, -1, -2) @ rep(target[:, :3, :3]))
        inside = np.maximum(floor[:, None] - _torso_distance(E2, arm.hips, arm.chest), 0.0) * 100     # cm
        short = np.linalg.norm(W2 - target[:, :3, 3], axis=-1) * 100
        c = (bend ** 2 + (p.twist_cost * _wrap_deg(elbow + wrist)) ** 2
             + (p.shoulder_twist_cost * shoulder) ** 2).reshape(L, Ks)
        c = c + (p.body_cost * inside) ** 2 + (p.reach_cost * short)[:, None] ** 2
        costs.append(c + (p.swivel_prior * np.degrees(sw)) ** 2 + (p.roll_prior * math.degrees(roll)) ** 2)
    cost = np.concatenate(costs, 1)                                               # state = roll index * Ks + swivel
    d_sw = np.degrees(np.abs((sw[:, None] - sw[None, :] + np.pi) % (2 * np.pi) - np.pi))
    d_roll = np.degrees(np.abs(rolls[:, None] - rolls[None, :]))
    trans = (p.swivel_smooth * d_sw)[None, :, None, :] ** 2 + (p.roll_smooth * d_roll)[:, None, :, None] ** 2
    path = viterbi(cost, trans.reshape(len(rolls) * Ks, len(rolls) * Ks))
    width = 2 * int(round(max(p.swivel_step_deg, p.roll_step_deg) / 10.0)) + 1   # ~a frame per 5 deg of grid
    swivel = running(np.unwrap(sw[path % Ks]), width, "mean")
    roll = running(rolls[path // Ks], width, "mean")
    return swivel, roll, cost[np.arange(L), path]


def collar_turn(C: np.ndarray, S: np.ndarray, target: np.ndarray, reach: np.ndarray, max_deg: float,
                hold_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """(L,3,3) world rotations about the collar head C that bring the shoulder S within `reach` of the target, as
    far as max_deg allows, and their angles (deg). The angle is held over the neighbouring hold_frames and averaged
    over as many, so the collar starts turning before the arm runs out of reach and never jumps."""
    s, t = S - C, target - C
    ns, nt = np.linalg.norm(s, axis=-1), np.linalg.norm(t, axis=-1)
    alpha = np.arccos(np.clip(np.sum(s * t, -1) / np.maximum(ns * nt, 1e-12), -1, 1))
    c = (ns * ns + nt * nt - reach * reach) / np.maximum(2 * ns * nt, 1e-12)
    need = np.where(c >= 1, alpha, alpha - np.arccos(np.clip(c, -1, 1)))
    need = np.where(np.linalg.norm(target - S, axis=-1) <= reach, 0.0, np.clip(need, 0.0, math.radians(max_deg)))
    need = running(running(need, 2 * hold_frames + 1, "max"), 2 * hold_frames + 1, "mean")
    axis = _unit(np.cross(s, t))
    smooth = np.stack([running(axis[:, k], 2 * hold_frames + 1, "mean") for k in range(3)], -1)
    axis = np.where(np.linalg.norm(smooth, axis=-1, keepdims=True) > 1e-3, _unit(smooth), axis)
    return axis_angle(axis, need), np.degrees(need)


def _rel_rot(rest: np.ndarray, parents: list[int], b: int) -> np.ndarray:
    p = parents[b]
    return rest[p, :3, :3].T @ rest[b, :3, :3]


def chain_fk(pose: np.ndarray, rest: np.ndarray, parents: list[int], basis: dict[int, np.ndarray]) -> dict:
    """Armature-space 4x4 poses (L,4,4) of the bones in `basis` (parent first, no location keys) from their new
    local rotations; every other bone keeps its pose in `pose`."""
    out: dict[int, np.ndarray] = {}
    for b, Rb in basis.items():
        p = parents[b]
        M = np.zeros(Rb.shape[:-2] + (4, 4))
        M[..., :3, :3] = Rb
        M[..., 3, 3] = 1.0
        out[b] = out.get(p, pose[:, p]) @ (np.linalg.inv(rest[p]) @ rest[b]) @ M
    return out


def closing_for_lift(lift: np.ndarray, p: Params) -> np.ndarray:
    """(L,) finger closing for the palm lifted `lift` (m) off its grip spot on the handle: 1 up to close_near_m[0],
    close_hold from close_near_m[1] to close_open_m[0], 0 from close_open_m[1] (smoothstep between). Measured on
    the character's hand turned into the grip and lifted along the palm normal: the relaxed fingers curl into the
    handle within ~4 cm of it, the grip's poke through it from ~1 cm out, three-quarters closed clears it from 0.5
    to 3 cm."""
    lift = np.asarray(lift, np.float64)
    (a, b), (c, d) = p.close_near_m, p.close_open_m
    return (1 - (1 - p.close_hold) * smoothstep((lift - a) / (b - a))) * (1 - smoothstep((lift - c) / (d - c)))


def grip_lift(hand: np.ndarray, racket: np.ndarray, grip: np.ndarray) -> tuple[np.ndarray, ...]:
    """Per frame, for the left-hand bone's pose hand (L,4,4) and the racket frame (L,4,4), both armature space, and
    grip the racket frame at the palm in the hand bone's frame (left_grip_frame): how far the grip point is off the
    racket's axis (m; 0 with the palm on its grip spot), how far up the shaft it is (m) and the angle (deg) between
    the hand's handle direction and the shaft."""
    g = np.einsum("lij,j->li", hand, grip[:, 3])[:, :3]
    d = np.einsum("lij,j->li", hand[:, :3, :3], grip[:3, 1])
    butt, y = racket[:, :3, 3], _unit(racket[:, :3, 1])
    s = np.sum((g - butt) * y, -1)
    radial = np.linalg.norm(g - butt - s[:, None] * y, axis=-1)
    return radial, s, np.degrees(np.arccos(np.clip(np.sum(_unit(d) * y, -1), -1.0, 1.0)))


def grip_amount(hand: np.ndarray, racket: np.ndarray, grip: np.ndarray, near: np.ndarray, p: Params) -> np.ndarray:
    """(L,) how far the left fingers close, from the left-hand bone's pose hand (L,4,4) and the racket frame racket
    (L,4,4), both armature space, grip the racket frame at the palm in the hand bone's frame (left_grip_frame), near
    (L,) in [0, 1] how far the approach has got (0 on the motion's own path, 1 once the hand has turned into the
    grip): closing_for_lift of the grip point's distance from the axis x how well the hand's handle direction
    matches the shaft (the roll round the axis is free) x whether the palm sits low enough on the shaft x near
    (see the module docstring)."""
    radial, s, angle = grip_lift(hand, racket, grip)
    aligned = 1 - smoothstep((angle - p.fit_on_deg) / (p.fit_off_deg - p.fit_on_deg))
    low = 1 - smoothstep((s - p.grip_top_s) / p.grip_open_m)
    return closing_for_lift(radial, p) * aligned * low * np.clip(np.asarray(near, np.float64), 0.0, 1.0)


def palm_on_handle(pose: np.ndarray, racket_in_hand: np.ndarray, bones: LeftArm, palm_centre: np.ndarray,
                   palm_normal: np.ndarray, s_max: float = 0.30) -> tuple[np.ndarray, np.ndarray]:
    """Per frame: distance (m) from the left palm centre to the racket's axis between the butt and s_max (handle,
    shaft, throat), and how far the palm turns away from that axis (deg, 0 = facing it)."""
    racket = _frames(pose[:, bones.right_hand] @ racket_in_hand)
    H = pose[:, bones.hand]
    c = np.einsum("lij,j->li", H[:, :3, :3], palm_centre) + H[:, :3, 3]
    n = np.einsum("lij,j->li", H[:, :3, :3], _unit(palm_normal))
    butt, y = racket[:, :3, 3], racket[:, :3, 1]
    q = butt + y * np.clip(np.sum((c - butt) * y, -1), 0.0, s_max)[:, None]
    facing = np.degrees(np.arccos(np.clip(np.sum(n * _unit(q - c), -1), -1, 1)))
    return np.linalg.norm(c - q, axis=-1), facing


def _elbow_deg(S: np.ndarray, E: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(_unit(S - E) * _unit(W - E), -1), -1, 1)))


def _summary(x: np.ndarray, scale: float = 100.0, digits: int = 1) -> dict | None:
    """p50 / p90 / max of x times scale, rounded to digits (None for no values); retarget's grip metrics use it too."""
    x = np.asarray(x, np.float64)
    if not len(x):
        return None
    return {k: round(float(np.percentile(x, q)) * scale, digits) for k, q in (("p50", 50), ("p90", 90), ("max", 100))}


# ------------------------------------------------------------------------------- the push off the racket
@dataclass(frozen=True)
class RacketShape:
    """The lower racket the left hand stays out of, in the racket frame (racket.py's shapes, as
    fingers.inside_racket tests them): butt cap and grip as cylinders to handle_length, the shaft as a box to
    throat_y, the throat's two arms as boxes from there to arm_end (x, y)."""

    handle_radius: float = 0.016
    handle_length: float = 0.19
    butt_radius: float = 0.0185
    butt_length: float = 0.012
    throat_y: float = 0.255
    shaft_half: tuple[float, float] = (0.010, 0.0105)    # across (x), along the string normal (z)
    arm_end: tuple[float, float] = (0.0784, 0.4058)
    arm_half: tuple[float, float] = (0.0066, 0.0109)     # in the string plane, along the normal


def inside_racket(q: np.ndarray, shape: RacketShape, margin: float = 0.0) -> np.ndarray:
    """(...,3) points in the racket frame -> (...) whether each is inside the lower racket by more than margin."""
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    radius = np.where(y < shape.butt_length, shape.butt_radius, shape.handle_radius)
    inside = (y >= 0) & (y <= shape.handle_length) & (np.hypot(x, z) < radius - margin)
    inside |= ((y > shape.handle_length) & (y <= shape.throat_y) & (np.abs(x) < shape.shaft_half[0] - margin)
               & (np.abs(z) < shape.shaft_half[1] - margin))
    ex, ey = shape.arm_end[0], shape.arm_end[1] - shape.throat_y
    length = math.hypot(ex, ey)
    for sign in (-1.0, 1.0):
        ux, uy = sign * ex / length, ey / length
        along = x * ux + (y - shape.throat_y) * uy
        across = np.abs(x * uy - (y - shape.throat_y) * ux)
        inside |= ((along >= 0) & (along <= length) & (across < shape.arm_half[0] - margin)
                   & (np.abs(z) < shape.arm_half[1] - margin))
    return inside


def hand_skin(hand_points: np.ndarray, closed: np.ndarray) -> np.ndarray:
    """(L,K,3) the hand's skin points in its bone frame at each frame's finger closing (L,), linear between the
    poses hand_points (N,K,3) holds for closings linspace(0, 1, N)."""
    n = len(hand_points)
    if n == 1:
        return np.broadcast_to(hand_points[0], (len(closed),) + hand_points.shape[1:])
    x = np.clip(np.asarray(closed, np.float64), 0.0, 1.0) * (n - 1)
    lo = np.minimum(np.floor(x).astype(int), n - 2)
    f = (x - lo)[:, None, None]
    return (1 - f) * hand_points[lo] + f * hand_points[lo + 1]


def _inside_count(q: np.ndarray, frame: np.ndarray, frames: int, offset: np.ndarray, shape: RacketShape,
                  margin: np.ndarray) -> np.ndarray:
    """(frames,) how many of the points q (M,3, racket frame; frame (M,) the frame of each) are inside the racket
    by more than margin (M,) once moved by offset ((3,) or (M,3))."""
    return np.bincount(frame, weights=inside_racket(q + offset, shape, margin), minlength=frames)


def push_off(points: np.ndarray, racket: np.ndarray, shape: RacketShape, margin: np.ndarray, lifts: np.ndarray,
             hold: int, p: Params, free: np.ndarray | None = None) -> np.ndarray:
    """(L,3) armature-space offsets that move the hand's points (L,K,3) out of the racket (none inside by more than
    margin (L,)), square to the shaft and turning with the racket.

    For each of p.push_directions round the shaft, the least of `lifts` (ascending, m) that clears each frame (where
    none does, the one leaving the fewest points inside), held and averaged over `hold` frames (running max, then
    mean: the push starts before it is needed, is never short of it and never jumps). The direction is chosen
    over the clip by Viterbi: per frame p.push_inside_cost per point still inside plus p.push_cost x (cm of
    push)^2, and p.push_jump_cost x (cm the push moves the hand from one frame to the next beyond `free` (L-1,) m,
    default p.push_free_m)^2, so it changes only where the push is ~0 or where it really pays (the hand does not
    swap sides of the racket where it crosses its axis); a change then turns it gradually over `hold` frames."""
    L, n = len(points), int(p.push_directions)
    angles = np.arange(n) * 2 * np.pi / n
    dirs = np.stack([np.sin(angles), np.zeros(n), np.cos(angles)], -1)          # +Z (the left palm's face) first
    q = np.einsum("lji,lkj->lki", racket[:, :3, :3], points - racket[:, None, :3, 3])
    reach = float(np.max(lifts)) + 1e-6           # only points this close to the racket can be inside after a push
    near = np.nonzero((np.abs(q[..., 0]) < shape.arm_end[0] + shape.arm_half[0] + reach)
                      & (q[..., 1] >= 0.0) & (q[..., 1] <= shape.arm_end[1])
                      & (np.abs(q[..., 2]) < max(shape.butt_radius, shape.arm_half[1]) + reach))
    if not len(near[0]):
        return np.zeros((L, 3))
    f, qn, mn = near[0], q[near], np.asarray(margin, np.float64)[near[0]]
    lift, left = np.zeros((n, L)), np.zeros((n, L))
    for k, d in enumerate(dirs):
        need, fewest, best = np.full(L, np.nan), np.full(L, np.inf), np.zeros(L)
        todo = np.ones(len(f), bool)
        for h in lifts:   # the least lift that clears each frame, else the one leaving the fewest points inside
            count = _inside_count(qn[todo], f[todo], L, h * d, shape, mn[todo])
            open_ = np.isnan(need)
            better = open_ & (count < fewest)
            fewest[better], best[better] = count[better], h
            need[open_ & (count == 0)] = h
            todo &= np.isnan(need)[f]
            if not todo.any():
                break
        need = np.where(np.isnan(need), best, need)
        lift[k] = running(running(need, hold, "max"), hold, "mean")
        left[k] = _inside_count(qn, f, L, lift[k][f, None] * d, shape, mn)
    if not lift.any():
        return np.zeros((L, 3))
    vec = lift.T[..., None] * np.einsum("lij,dj->ldi", racket[:, :3, :3], dirs) * 100   # (L, D, 3) the pushes, cm
    jump = np.linalg.norm(vec[1:, None] - vec[:-1, :, None], axis=-1)            # (L-1, D from, D to)
    free_cm = (np.full(L - 1, p.push_free_m) if free is None else np.asarray(free, np.float64)) * 100
    path = viterbi((p.push_inside_cost * left + p.push_cost * (lift * 100) ** 2).T,
                   p.push_jump_cost * np.maximum(jump - free_cm[:, None, None], 0.0) ** 2)
    size = lift[path, np.arange(L)]
    turn = _unit(np.stack([running(dirs[path, k] * size, hold, "mean") for k in range(3)], -1))
    return size[:, None] * np.einsum("lij,lj->li", racket[:, :3, :3], turn)


# ---------------------------------------------------------------------------------------------- the pass
def left_hand_on_racket(rest: np.ndarray, parents: list[int], basis: np.ndarray, pose: np.ndarray,
                        racket_in_hand: np.ndarray, bones: LeftArm, palm_centre: np.ndarray | None = None,
                        palm_normal: np.ndarray | None = None, handle_dir: np.ndarray | None = None,
                        wrist_dist: np.ndarray | None = None, fps: float = 25.0, params: Params | None = None,
                        hand_points: np.ndarray | None = None, racket_shape: RacketShape | None = None,
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Put the left hand on the racket wherever the wrists are close (see the module docstring).

    rest (B,4,4) bone rest matrices (Rig.rest), parents (B,) parent indices (-1 = root), basis (L,B,3,3) pose-bone
    rotations and pose (L,B,4,4) armature-space pose matrices of the same frames (forward_kinematics of basis),
    racket_in_hand (4,4) the racket frame in the right-hand bone's frame (the handle at frame t is
    pose[t, right_hand] @ racket_in_hand). palm_centre / palm_normal: the left palm (a point on the palm skin, the
    outward normal) in the left-hand bone's frame, default default_palm(); handle_dir: the handle's direction across
    the left palm in that frame (default: grip_angle_deg). wrist_dist (L,): the distance the weight is computed
    from, default the hand bones' heads (pass the SMPL wrists to use GVHMR's own scale). hand_points (N,K,3): the
    left hand's skin points in its bone's frame for finger closings linspace(0, 1, N) (fingers.hand_points), to
    push the hand off the racket (RacketShape, default racket.py's); without them there is no push.

    Returns (basis with only the chain's bones changed, the blend weight w (L,) in [0, 1], how far the left fingers
    close (L,) in [0, 1] (grip_amount), a JSON-ready summary). Run forward_kinematics again afterwards."""
    p = params or Params()
    basis = np.asarray(basis, np.float64)
    pose = np.asarray(pose, np.float64)
    rest = np.asarray(rest, np.float64)
    L = len(basis)
    chain = [b for b in (bones.collar, bones.upper, bones.fore, bones.hand) if b is not None]
    for child, parent in zip(chain[1:], chain[:-1], strict=True):
        if parents[child] != parent:
            raise ValueError(f"bone {child} is not a child of {parent}: the left arm must be one chain")
    if parents[chain[0]] < 0:
        raise ValueError("the left arm chain cannot start at a root bone")
    if palm_centre is None or palm_normal is None:
        palm_centre, palm_normal = default_palm(rest, bones.hand, bones.right_hand)
    palm_centre, palm_normal = np.asarray(palm_centre, np.float64), _unit(palm_normal)
    grip = left_grip_frame(palm_centre, palm_normal, handle_dir, p.grip_angle_deg, p.handle_above_palm_m)

    heads = pose[..., :3, 3]
    if wrist_dist is None:
        wrist_dist = np.linalg.norm(heads[:, bones.hand] - heads[:, bones.right_hand], axis=-1)
    w_raw = fill_dips(two_hand_state(wrist_dist, p.on_m, p.off_m, p.median_frames), p.fill_frames)
    racket = _frames(pose[:, bones.right_hand] @ racket_in_hand)
    s = handle_position(racket, heads[:, parents[chain[0]]], fps, p)
    along = np.tile(np.eye(4), (L, 1, 1))
    along[:, 1, 3] = s
    handle = racket @ along                                  # the racket frame moved to the palm's point on the shaft
    grip_inv = np.linalg.inv(grip)

    # the full IK's choices (weight 1) on every frame: the collar turn, then the swivel and palm roll
    R = pose[..., :3, :3]
    S0, E0, W0 = heads[:, bones.upper], heads[:, bones.fore], heads[:, bones.hand]
    S, E, W = S0, E0, W0
    C = heads[:, bones.collar] if bones.collar is not None else None
    Rc, collar_deg = np.tile(np.eye(3), (L, 1, 1)), np.zeros(L)
    if C is not None:
        l1, l2 = np.linalg.norm(E - S, axis=-1), np.linalg.norm(W - E, axis=-1)
        cap = np.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * math.cos(math.radians(p.max_elbow_deg)))
        wrist0 = (handle @ grip_inv)[:, :3, 3]                # the wrist's target with the palm on the opposite face
        Rc, collar_deg = collar_turn(C, S, wrist0, cap - p.soft_m, p.max_collar_deg, round(p.collar_hold_s * fps))
        S, E, W = (C + np.einsum("lij,lj->li", Rc, X - C) for X in (S, E, W))
    rel = {k: _rel_rot(rest, parents, b) for k, b in (("upper", bones.upper), ("fore", bones.fore),
                                                       ("hand", bones.hand))}
    R_col = Rc @ R[:, bones.collar] if bones.collar is not None else R[:, parents[bones.upper]]
    root = chain[0]
    while parents[root] >= 0:
        root = parents[root]
    arm = Arm(S, E, W, R_col, Rc @ R[:, bones.upper], Rc @ R[:, bones.fore], rel, heads[:, root],
              heads[:, parents[chain[0]]])
    swivel, roll, _ = comfortable_arm(arm, handle, grip_inv, p)
    spin = np.tile(np.eye(4), (L, 1, 1))
    spin[:, :3, :3] = axis_angle(np.array([0.0, 1.0, 0.0]), roll)
    target = handle @ spin @ grip_inv                         # where the left-hand bone holds the handle

    # the blend path, weight 0 (the motion) to 1 (the full IK), in the racket's frame (butt origin, +Y the shaft,
    # +Z the left palm's face): the palm straight from the motion's to hover_m out from its grip spot, then down
    # onto it; the hand's rotation from the motion's to the grip's, its quaternion signs continuous over each
    # stretch where the wrists are close (or the ramps may reach)
    w_raw = widen(w_raw, round(p.ramp_out_s * fps))
    near = w_raw > 0
    Rk, butt = racket[:, :3, :3], racket[:, :3, 3]
    q_f = np.einsum("lji,lj->li", Rk, np.einsum("lij,j->li", R[:, bones.hand], palm_centre) + W0 - butt)
    q_g = np.einsum("lji,lj->li", Rk, np.einsum("lij,j->li", target[:, :3, :3], palm_centre) + target[:, :3, 3]
                    - butt)
    hover = p.hover_m * _unit(q_g * [1.0, 0.0, 1.0])         # off the axis at the grip spot
    rot_g = continuous_quats(quat_from_matrix(np.swapaxes(Rk, -1, -2) @ target[:, :3, :3]), near,
                             np.tile([1.0, 0, 0, 0], (L, 1)))
    rot_f = continuous_quats(quat_from_matrix(np.swapaxes(Rk, -1, -2) @ R[:, bones.hand]), near, rot_g)
    sw = unwrap_runs(swivel, near)
    fk_elbow = _elbow_deg(S0, E0, W0)
    tip = palm_centre + np.array([0.0, FINGERTIP_PAST_PALM_M, 0.0])
    share = np.full(L, p.turn_share)

    def stages(lam: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """How far weight lam has got towards the hover (turning into the grip) and down from it onto the handle."""
        return np.clip(lam / share, 0.0, 1.0), np.clip((lam - share) / (1.0 - share), 0.0, 1.0)

    def solve(lam: np.ndarray, offset: np.ndarray | None = None) -> tuple[dict, list[np.ndarray], np.ndarray]:
        """The chain's local rotations at blend weight lam (L,) with the palm's target moved by offset (L,3), the
        palm, fingertip and elbow (L,3) and how far the wrist stays short of its target."""
        a, down = stages(lam)
        q = q_f + (q_g + hover - q_f) * a[:, None] - hover * down[:, None]
        palm = butt + np.einsum("lij,lj->li", Rk, q) + (0.0 if offset is None else offset)
        Rh = Rk @ matrix_from_quat(slerp_quat(rot_f, rot_g, a))
        Rc_l = slerp(np.tile(np.eye(3), (L, 1, 1)), Rc, lam)
        Sa, Ea, Wa = (S0, E0, W0) if C is None else (C + np.einsum("lij,lj->li", Rc_l, X - C) for X in (S0, E0, W0))
        cap_deg = p.max_elbow_deg + (1 - lam) * np.maximum(fk_elbow - p.max_elbow_deg, 0.0)   # 0: its own reach
        wrist = palm - np.einsum("lij,j->li", Rh, palm_centre)
        E2, W2, R1, R12, short = two_bone_ik(Sa, Ea, Wa, wrist, Ea, cap_deg, p.soft_m * lam, sw * lam)
        world = {bones.upper: R1 @ Rc_l @ R[:, bones.upper], bones.fore: R12 @ Rc_l @ R[:, bones.fore]}
        if p.twist_balance:   # the forearm rolls about its own axis (the wrist stays put) and evens out the twist
            up, fo = world[bones.upper], world[bones.fore]
            elbow_twist = twist_bend_deg(rel["fore"].T @ np.swapaxes(up, -1, -2) @ fo)[0]
            wrist_twist = twist_bend_deg(rel["hand"].T @ np.swapaxes(fo, -1, -2) @ Rh)[0]
            gap = unwrap_runs(np.radians(wrist_twist - elbow_twist), near)
            world[bones.fore] = axis_angle(W2 - E2, lam * p.twist_balance * gap) @ fo
        # on the way the wrist bends at most max_wrist_bend_deg off the forearm (the hand turns about the wrist; the
        # cap fades out as it comes down onto the handle)
        fo = world[bones.fore]
        Lh = rel["hand"].T @ np.swapaxes(fo, -1, -2) @ Rh
        y = Lh[:, :, 1]
        cap = np.radians(p.max_wrist_bend_deg + (180.0 - p.max_wrist_bend_deg) * down)
        over = np.maximum(np.arccos(np.clip(y[:, 1], -1.0, 1.0)) - cap, 0.0)
        Rh = fo @ rel["hand"] @ axis_angle(np.cross(y, [0.0, 1.0, 0.0]), over) @ Lh
        world[bones.hand] = Rh
        if bones.collar is not None:
            world[bones.collar] = Rc_l @ R[:, bones.collar]
        local = {}
        for bone in chain:
            parent_rot = world.get(parents[bone], R[:, parents[bone]])
            local[bone] = _rel_rot(rest, parents, bone).T @ np.swapaxes(parent_rot, -1, -2) @ world[bone]
        points = [W2 + np.einsum("lij,j->li", Rh, palm_centre), W2 + np.einsum("lij,j->li", Rh, tip), E2]
        return local, points, short

    # the blend's cost level by level (frames per unit of weight): it moves the palm at most max_blend_step_m and
    # the fingertips and the elbow at most max_blend_swing_m per frame beyond their own paths, and the fingers'
    # closing at most close_ramp_step per frame (on the hand at each level, and on closing_for_lift as the palm
    # comes down from the hover, finer than the levels)
    lams = np.linspace(0.0, 1.0, max(int(p.blend_samples), 2))
    if p.even_share:   # the turn's share of the blend per stretch, so that the palm moves at an even speed along it
        pts = [solve(np.full(L, lam))[1][0] for lam in lams]
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)                        # (N-1, L)
        first = np.sum(seg * (lams[1:, None] <= p.turn_share + 1e-9), 0)
        even = first / np.maximum(seg.sum(0), 1e-9)
        for a, b in runs(near):
            share[a:b] = np.clip(np.median(even[a:b]), *p.share_range)
    solved = [solve(np.full(L, lam)) for lam in lams]
    limits = (p.max_blend_step_m, p.max_blend_swing_m, p.max_blend_swing_m)
    closing = [grip_amount(chain_fk(pose, rest, parents, local)[bones.hand], racket, grip, stages(lam)[0], p)
               for lam, (local, _, _) in zip(lams, solved, strict=True)]
    dl = np.diff(lams)
    sub = np.linspace(0.0, 1.0, 17)
    cost = np.zeros((L, len(dl)))
    for k in range(len(dl)):
        for b0, b1, lim in zip(solved[k][1], solved[k + 1][1], limits, strict=True):
            cost[:, k] = np.maximum(cost[:, k], np.linalg.norm(b1 - b0, axis=-1) / (lim * dl[k]))
        cost[:, k] = np.maximum(cost[:, k], np.abs(closing[k + 1] - closing[k]) / (p.close_ramp_step * dl[k]))
        down = stages(lams[k] + dl[k] * sub[None, :])[1] if False else \
            np.clip((lams[k] + dl[k] * sub[None, :] - share[:, None]) / (1.0 - share[:, None]), 0.0, 1.0)
        slope = np.abs(np.diff(closing_for_lift(p.hover_m * (1 - down), p), axis=1)).max(1) * (len(sub) - 1) / dl[k]
        cost[:, k] = np.maximum(cost[:, k], slope / p.close_ramp_step)
    cost = np.maximum(cost, float(max(p.blend_frames, 1)))
    cost[~near] = float(max(p.blend_frames, 1))              # off the stretches the weight is 0 anyway
    edge = np.maximum(cost[1:], cost[:-1])
    p0, p1, short = solved[0][1][0], solved[-1][1][0], solved[-1][2]
    own = np.maximum(np.linalg.norm(np.diff(p0, axis=0), axis=-1), np.linalg.norm(np.diff(p1, axis=0), axis=-1))
    reach = 1 - smoothstep((running(short, p.median_frames, "max") - p.short_on_m) / (p.short_off_m - p.short_on_m))
    goal = w_raw * (share + (1 - share) * reach)              # short of the handle: the hand hovers in the grip

    def weight(budget: np.ndarray) -> np.ndarray:
        """The weight with each frame-to-frame step costing edge / budget (L-1,) frames per unit."""
        return np.clip(ramp_limit(goal, edge / budget[:, None], lams), 0.0, 1.0)

    def gentle(closed: np.ndarray) -> np.ndarray:
        """The closing changing at most max_close_step per frame: the least such series at or above it."""
        return 1 - limit_ramps(1 - closed, np.full(max(L - 1, 0), p.max_close_step))

    def place(w: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """The chain's local rotations at weight w, pushed off the racket where the hand's skin would go in (with
        hand_points), the hand bone's pose, the finger closing and the push (L,3)."""
        a, down = stages(w)

        def closing_of(local: dict) -> tuple[np.ndarray, np.ndarray]:
            H = chain_fk(pose, rest, parents, local)[bones.hand]
            return H, grip_amount(H, racket, grip, a, p)

        local, pts, _ = solve(w)
        H, closed = closing_of(local)
        push = np.zeros((L, 3))
        if hand_points is None:
            return local, H, gentle(closed), push
        # the push may move the hand by what the blend leaves of max_pop_m in each frame (push_free_m at least)
        blend = np.linalg.norm(np.diff(pts[0], axis=0), axis=-1) - own
        free = np.clip(p.max_pop_m - np.maximum(blend, 0.0), p.push_free_m, p.max_pop_m)
        shape = racket_shape or RacketShape()
        hold = 2 * round(p.push_hold_s * fps) + 1
        lifts = np.arange(0.0, p.max_push_m + 1e-9, p.push_step_m)
        base = H
        on = 1 - smoothstep(grip_lift(base, racket, grip)[0] / p.close_near_m[1])   # the palm on its grip spot

        def push_for(closed: np.ndarray) -> np.ndarray:
            """The push off the racket of the unpushed hand with the fingers closed so far; none while the hand comes
            down onto its grip spot (turned into the grip there, it clears the handle)."""
            skin = np.einsum("lij,lkj->lki", base[:, :3, :3], hand_skin(hand_points, closed)) + base[:, None, :3, 3]
            return push_off(skin, racket, shape, p.push_margin_m * on, lifts, hold, p, free) * (1 - down)[:, None]

        for _ in range(p.push_rounds):   # each round with the fingers as the last push left them
            push = push_for(closed)
            if not push.any():
                break
            local = solve(w, push)[0]
            H, closed = closing_of(local)
        if L > 1 and np.abs(np.diff(closed)).max() > p.max_close_step:   # the fingers' closing slowed: pushed again
            closed = gentle(closed)
            push = push_for(closed)
            local = solve(w, push)[0]
            H = chain_fk(pose, rest, parents, local)[bones.hand]
        return local, H, closed, push

    # the palm's step against the motion's and the full IK's: where it still moves more than max_pop_m beyond
    # both (the blend's speed varies along its path, the push moves the hand too), those ramps slow down and the
    # hand is placed again. Where a racket swings fast, the path's own point half-way round it can outrun both at
    # a fixed weight ("calm": how far up the blend it does not, less what the push may add); a weight left half-way
    # there comes down to that
    excess = np.stack([np.linalg.norm(np.diff(pts[0], axis=0), axis=-1) - own for _, pts, _ in solved])   # (N, L-1)
    excess = np.maximum(np.c_[excess, excess[:, -1:]], np.c_[excess[:, :1], excess])           # (N, L)
    bad = excess > p.max_pop_m - (p.push_free_m if hand_points is not None else 0.0)
    calm = np.where(bad.any(0), lams[np.maximum(np.argmax(bad, 0) - 1, 0)], 1.0)
    e0, e1 = solved[0][1][2], solved[-1][1][2]
    own_elbow = np.maximum(np.linalg.norm(np.diff(e0, axis=0), axis=-1), np.linalg.norm(np.diff(e1, axis=0), axis=-1))
    budget = np.ones(L - 1)
    for _ in range(p.repair_rounds + 1):
        w = weight(budget)
        local, H, closed, push = place(w)
        palm = np.einsum("lij,j->li", H[:, :3, :3], palm_centre) + H[:, :3, 3]
        elbow = chain_fk(pose, rest, parents, local)[bones.fore][:, :3, 3]
        over = ((np.linalg.norm(np.diff(palm, axis=0), axis=-1) - own > p.max_pop_m)
                | (np.linalg.norm(np.diff(elbow, axis=0), axis=-1) - own_elbow > p.max_elbow_pop_m))
        if not over.any():
            break
        around = over | np.r_[over[1:], False] | np.r_[False, over[:-1]]
        budget = np.where(around, budget * p.repair_factor, budget)
        partial = (np.r_[over, False] | np.r_[False, over]) & (w < 1.0) & (w > calm)
        goal = np.where(partial, np.minimum(goal, calm), goal)

    out = basis.copy()
    moved = (w > 0) | (np.linalg.norm(push, axis=-1) > 0)
    for b in chain:   # frames with no weight and no push keep the motion's rotations bit for bit
        out[:, b] = np.where(moved[:, None, None], local[b], basis[:, b])
    new = chain_fk(pose, rest, parents, {b: out[:, b] for b in chain})
    after = pose.copy()
    for b in chain:
        after[:, b] = new[b]
    info = _ik_summary(p, bones, pose, after, basis, out, racket_in_hand, (palm_centre, palm_normal), w, closed,
                       s, short, push, (p0, p1, e0, e1), (sw, roll, collar_deg), (arm.hips, arm.chest))
    return out, w, closed, info


def _ik_summary(p: Params, bones: LeftArm, pose: np.ndarray, after: np.ndarray, basis: np.ndarray, out: np.ndarray,
                racket_in_hand: np.ndarray, palm: tuple[np.ndarray, np.ndarray], w: np.ndarray, closed: np.ndarray,
                s: np.ndarray, short: np.ndarray, push: np.ndarray, paths: tuple[np.ndarray, np.ndarray],
                angles: tuple[np.ndarray, ...], torso: tuple[np.ndarray, np.ndarray]) -> dict:
    """left_hand_on_racket's JSON-ready summary, over the frames with w > 0.5 unless said otherwise. pose / after
    (L,B,4,4) before and after the IK, basis / out their pose-bone rotations, palm (centre, normal) in the hand
    bone's frame, s the palm's target along the shaft, short how far the full IK's wrist stays short, push (L,3),
    paths the palm and the elbow on the motion and on the full IK (L,3), angles (swivel rad, palm roll rad, collar
    deg), torso (hips, chest) heads."""
    L = len(w)
    centre, normal = palm
    p0, p1, e0, e1 = paths
    swivel, roll, collar_deg = angles
    H = after[:, bones.hand]
    p_after = np.einsum("lij,j->li", H[:, :3, :3], centre) + H[:, :3, 3]
    s_max = max(p.handle_s, p.throat_s) + 0.03
    d0, f0 = palm_on_handle(pose, racket_in_hand, bones, centre, normal, s_max)
    d1, f1 = palm_on_handle(after, racket_in_hand, bones, centre, normal, s_max)
    on = w > 0.5
    step = {k: np.linalg.norm(np.diff(x, axis=0), axis=-1) for k, x in (("fk", p0), ("ik", p1), ("out", p_after))}
    pop = np.maximum(step["out"] - np.maximum(step["fk"], step["ik"]), 0.0)
    elbow_step = np.linalg.norm(np.diff(after[:, bones.fore, :3, 3], axis=0), axis=-1)
    elbow_own = np.maximum(np.linalg.norm(np.diff(e0, axis=0), axis=-1), np.linalg.norm(np.diff(e1, axis=0), axis=-1))
    elbow_pop = np.maximum(elbow_step - elbow_own, 0.0)
    heads = pose[..., :3, 3]
    elbow0 = _elbow_deg(heads[:, bones.upper], heads[:, bones.fore], heads[:, bones.hand])
    elbow1 = _elbow_deg(*(after[:, b, :3, 3] for b in (bones.upper, bones.fore, bones.hand)))
    wrist_0, wrist_1 = twist_bend_deg(basis[:, bones.hand]), twist_bend_deg(out[:, bones.hand])
    elbow_twist0, elbow_twist1 = twist_bend_deg(basis[:, bones.fore])[0], twist_bend_deg(out[:, bones.fore])[0]
    shoulder0, shoulder1 = twist_bend_deg(basis[:, bones.upper])[0], twist_bend_deg(out[:, bones.upper])[0]
    torso0 = _torso_distance(heads[:, None, bones.fore], *torso)[:, 0]
    torso1 = _torso_distance(after[:, None, bones.fore, :3, 3], *torso)[:, 0]
    lift = np.linalg.norm(push, axis=-1)

    def most(x: np.ndarray, where: np.ndarray = on) -> float:
        return round(float(np.abs(x[where]).max(initial=0)), 1)

    return {
        "frames": L,
        "frames_on": int(on.sum()),
        "frames_on_handle": int((on & (s <= p.grip_top_s)).sum()),
        "frames_full": int((w > 0.99).sum()),               # the hand on its grip spot
        "frames_gripped": int((closed > 0.5).sum()),        # the fingers closed past halfway
        "frames_open_on": np.flatnonzero(on & (closed <= 0.5)).tolist(),   # on the racket, the fingers open
        "weight_max_step": round(float(np.abs(np.diff(w)).max()), 3) if L > 1 else 0.0,
        "closing_max_step": round(float(np.abs(np.diff(closed)).max()), 3) if L > 1 else 0.0,
        "palm_to_handle_cm": {"before": _summary(d0[on]), "after": _summary(d1[on])},
        "palm_facing_deg": {"before": _summary(f0[on], 1.0), "after": _summary(f1[on], 1.0)},
        "max_step_cm": {"before": round(float(step["fk"].max(initial=0)) * 100, 1),
                        "after": round(float(step["out"].max(initial=0)) * 100, 1)},
        "max_pop_cm": round(float(pop.max(initial=0)) * 100, 2),
        "pop_frames": (np.flatnonzero(pop > p.max_pop_m) + 1).tolist(),   # stepping more than max_pop_m beyond both
        "elbow_pop_cm": round(float(elbow_pop.max(initial=0)) * 100, 2),
        "elbow_pop_frames": (np.flatnonzero(elbow_pop > p.max_elbow_pop_m) + 1).tolist(),
        "elbow_deg_max": {"before": round(float(elbow0.max()), 1), "after": round(float(elbow1.max()), 1)},  # all
        "wrist_bend_deg_max": {"before": most(wrist_0[1]), "after": most(wrist_1[1])},
        "approach_wrist_bend_deg_max": most(wrist_1[1], (w > 0) & ~on),   # on the way (0 < w <= 0.5)
        "wrist_twist_deg_max": {"before": most(wrist_0[0]), "after": most(wrist_1[0])},
        "elbow_twist_deg_max": {"before": most(elbow_twist0), "after": most(elbow_twist1)},
        "shoulder_twist_deg_max": {"before": most(shoulder0), "after": most(shoulder1)},
        "elbow_to_torso_axis_cm_min": {"before": round(float(torso0[on].min(initial=9)) * 100, 1),
                                       "after": round(float(torso1[on].min(initial=9)) * 100, 1)},
        "swivel_deg_max": most(_wrap_deg(np.degrees(swivel * w))),   # as applied, in (-180, 180]
        "palm_roll_deg_max": most(np.degrees(roll) * w),
        "collar_deg_max": round(float((collar_deg * w).max(initial=0)), 1),
        "frames_pushed": int((lift > 0).sum()),             # the hand pushed off the racket (hand_points)
        "push_cm_max": round(float(lift.max(initial=0)) * 100, 1),
        "reach_short_frames": np.flatnonzero(on & (short > 0.01)).tolist(),
        "reach_short_cm_max": round(float(short[on].max(initial=0)) * 100, 1),
    }
