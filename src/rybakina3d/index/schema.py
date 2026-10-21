"""Structured labels Gemini returns for each shot of the compilation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class View(StrEnum):
    main_broadcast = "main_broadcast"  # elevated camera behind a baseline, whole court visible
    behind_low = "behind_low"          # low camera behind the baseline, player large
    side = "side"                      # side-on / courtside angle
    closeup = "closeup"                # head/torso close-up, crowd reaction, bench
    other = "other"                    # graphics, crowd, ceremony, drone …


class Side(StrEnum):
    near = "near"      # Rybakina is on the half of the court closest to the camera
    far = "far"
    solo = "solo"      # only Rybakina is in frame (side angles, close-ups)
    absent = "absent"


class EventType(StrEnum):
    serve = "serve"
    forehand = "forehand"
    backhand = "backhand"
    slice = "slice"
    volley = "volley"
    overhead = "overhead"
    celebration = "celebration"
    other = "other"


class Event(BaseModel):
    type: EventType
    t: float = Field(description="Racket-ball contact (or celebration start), seconds from the start of the FULL video")
    hitter: str = Field(description="'rybakina', 'opponent' or 'unknown'")
    confidence: float = Field(description="0..1")


class ShotLabel(BaseModel):
    shot_id: int
    view: View
    replay: bool
    slow_motion: bool
    rybakina_side: Side
    full_body: bool = Field(description="Rybakina's whole body (head to feet) visible for most of the shot")
    quality: int = Field(description="1..5 usefulness for capturing Rybakina's body motion: size, sharpness, occlusion")
    outfit: str = Field(description="Rybakina's outfit colours in this shot, a few words")
    events: list[Event]


class ChunkLabels(BaseModel):
    shots: list[ShotLabel]
