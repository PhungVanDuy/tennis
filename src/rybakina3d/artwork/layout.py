"""The character-sheet layout, modelled on the client's Naomi Osaka sheet (naomi_style.png).

Every figure on the sheet is a render of the game-ready character driven by a pose from her captured
motion; this module only says *where* each figure goes and *which* pose/camera it wants. Boxes are
normalised sheet coordinates (x0, y0, x1, y1), origin top-left.

Pose specs:
  idle                      the rest pose with the arms relaxed down (turnaround, portrait, faces)
  {stroke, at, offset}      a frame of a label-valid clip of that stroke: at = "contact" | "trophy" | "start"
                            (trophy = the toss hand's apex before a serve's contact), offset in frames
Camera: azimuth in degrees around the character relative to the way she faces in that frame (0 = in front,
90 = her left side, 180 = behind), elevation in degrees, framing = full | bust | head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SHEET_SIZE = (3840, 2560)        # px, 3:2 like the reference
STYLE_REFERENCE = "naomi_style.png"

CARD = {
    "name": "Elena Rybakina",
    "rows": [("ROLE", "Clinical Baseliner"), ("CORE MOOD", "Ice-cool"), ("VISUAL SIGNATURE", "Effortless Power")],
}


@dataclass(frozen=True)
class Slot:
    name: str
    box: tuple[float, float, float, float]
    pose: str | dict
    azimuth: float = 0.0
    elevation: float = 5.0
    framing: str = "full"
    silhouette: bool = False      # drawn as a flat dark shape (the reference's "Silhouettes" rows)
    paint: bool = True            # painted over in the reference style (silhouettes and the atlas are not)


def _silhouette(name: str, box, stroke: str, at: str, offset: int, azimuth: float) -> Slot:
    return Slot(name, box, {"stroke": stroke, "at": at, "offset": offset}, azimuth=azimuth, silhouette=True,
                paint=False)


SLOTS: list[Slot] = [
    Slot("front", (0.055, 0.03, 0.175, 0.45), "idle", azimuth=0),
    Slot("back", (0.185, 0.03, 0.300, 0.45), "idle", azimuth=180),
    # the reference's hero is a two-handed backhand with the racket out and the face up: the backhand's loading
    # moment (8 frames = 0.3 s before contact), lunging out with both hands on the racket and her eyes on the
    # incoming ball (at contact she looks down at it and the visor hides her face); camera a little above her hips
    Slot("hero", (0.315, 0.02, 0.760, 0.70), {"stroke": "backhand", "at": "contact", "offset": -8}, azimuth=-35,
         elevation=8),
    Slot("portrait", (0.755, 0.02, 0.995, 0.405), "idle", azimuth=55, elevation=2, framing="bust"),
    # the "Details" pair right of the hero's front foot (her lunge fills the hero box's lower right): a forehand
    # swinging up to the ball, seen 3/4 so the arms do not spread across the sheet, and the serve's trophy position
    Slot("action_a", (0.700, 0.43, 0.850, 0.775), {"stroke": "forehand", "at": "contact", "offset": -2}, azimuth=45),
    Slot("action_b", (0.855, 0.42, 0.995, 0.775), {"stroke": "serve", "at": "trophy", "offset": 0}, azimuth=70),
    # silhouettes: two rows of three, each a different stroke / moment, seen near side-on so that limbs and racket
    # stay apart in a flat shape (from the front the racket hides behind her)
    *(_silhouette(f"sil_{i + 1}", box, stroke, at, offset, az) for i, (box, stroke, at, offset, az) in enumerate([
        ((0.040, 0.47, 0.120, 0.69), "serve", "contact", 0, 80),        # reaching up to the ball
        ((0.125, 0.47, 0.205, 0.69), "forehand", "contact", -6, 70),    # racket coming through
        ((0.210, 0.47, 0.290, 0.69), "backhand", "contact", 8, -60),    # deep lunge, the finish
        ((0.040, 0.72, 0.120, 0.89), "forehand", "start", 0, 60),       # ready, racket out in front
        ((0.125, 0.72, 0.205, 0.89), "forehand", "contact", 6, 40),     # wrapped over the shoulder
        ((0.210, 0.72, 0.290, 0.89), "serve", "contact", 10, 70),       # landing into the court
    ])),
    # four head close-ups (the rig has no face controls: four angles instead of four expressions)
    Slot("face_1", (0.615, 0.80, 0.700, 0.955), "idle", azimuth=-25, elevation=0, framing="head"),
    Slot("face_2", (0.705, 0.80, 0.790, 0.955), "idle", azimuth=0, elevation=0, framing="head"),
    Slot("face_3", (0.795, 0.80, 0.880, 0.955), "idle", azimuth=30, elevation=0, framing="head"),
    Slot("face_4", (0.885, 0.80, 0.970, 0.955), "idle", azimuth=60, elevation=0, framing="head"),
]

CARD_BOX = (0.280, 0.745, 0.580, 0.995)
LABELS = [("Silhouettes", (0.165, 0.705)), ("Strokes", (0.165, 0.915)), ("Details", (0.850, 0.782))]


@dataclass(frozen=True)
class Atlas:
    """Second sheet: every captured clip at its contact frame, grouped by stroke ('the whole moveset')."""

    azimuth: float = 30.0
    elevation: float = 8.0
    framing: str = "full"
    columns: int = 12
    stroke_order: list[str] = field(default_factory=lambda: ["serve", "forehand", "backhand", "slice", "overhead"])


ATLAS = Atlas()


@dataclass(frozen=True)
class Motion:
    """Third deliverable: every captured clip as a video of the game-ready character in the sheet's look.

    One fixed camera per clip at the atlas's 3/4 angle (relative to the way she faces at contact), framed on the
    whole clip's motion so she moves through the frame; the clip's full retargeted action at its real-time rate;
    opaque paper background with the soft contact shadow (video models do not take alpha). A turntable of the
    idle pose shows the asset itself.
    """

    azimuth: float = 30.0
    elevation: float = 8.0
    size: tuple[int, int] = (1024, 1024)     # px, square: lunges are wide, serves tall
    samples: int = 64                        # Cycles + OIDN; 39 clips x ~60 frames
    turntable_s: float = 6.0                 # one full turn
    reel_columns: int = 8                    # the moveset reel: every clip in a grid, contacts in sync


MOTION = Motion()
