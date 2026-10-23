"""Typed pipeline configuration (configs/*.yaml) and the on-disk layout of a run."""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rybakina.yaml"


class Athlete(BaseModel):
    name: str
    handedness: str = "right"
    backhand: str = "two_handed"
    height_m: float = 1.75


class Source(BaseModel):
    video: Path
    youtube: str | None = None


class Remote(BaseModel):
    host: str = ""                # user@host; empty = R3D_HOST from the environment or .env (see remote.host)
    root: str = "rybakina3d"
    blender: str = "blender"


class Models(BaseModel):
    index: str
    locate: str
    image_hero: str
    image_views: str
    mesh: str
    mesh_alt: str
    rig: str
    image_paint: str = "openai/gpt-image-2.5/sunburst/edit"   # artwork: repaint renders in the reference style (fal)
    video_paint: str = "fal-ai/bernini-r/reference-edit-video"  # artwork motion: carry the painted look through a clip


class Shots(BaseModel):
    adaptive_threshold: float = 3.0
    min_shot_s: float = 0.6


class Index(BaseModel):
    proxy_height: int = 360
    chunk_s: float = 240
    fps: float = 2.0


class Select(BaseModel):
    strokes: dict[str, int]
    window_s: dict[str, tuple[float, float]]
    allowed_views: list[str]
    allow_slow_motion: bool = False
    min_quality: int = 3
    oversample: float = 2.0  # candidates per kept clip; `curate` rejects mislabelled ones
    onset_search_s: float = 0.5
    # only strokes whose contact (Gemini's time) lies in this stretch of the source, seconds; None = the whole video
    source_span_s: tuple[float, float] | None = None

    def window(self, stroke: str) -> tuple[float, float]:
        return self.window_s.get(stroke, self.window_s["default"])


class Clips(BaseModel):
    model_fps: int = 30
    time_mode: str = "retime"
    preview_height: int = 720


class Track(BaseModel):
    detector: str = "yolo11x.pt"
    conf: float = 0.25
    min_coverage: float = 0.7


class Motion(BaseModel):
    static_motion_px: float = 1.5
    default_hfov_deg: float = 32
    flip_test: bool = True


class Character(BaseModel):
    height_m: float = 1.84
    target_tris: int = 24000
    views: list[str] = Field(default_factory=lambda: ["front", "left", "back", "right"])
    # A second character on the same captured motion: its character/, anim/ and artwork/ live under
    # <run_dir>/variants/<variant>/ with <variant>_* file names, while footage, clips and motion stay shared.
    variant: str | None = None
    name: str | None = None          # display name on the artwork (default: the athlete's)
    refs_dir: Path | None = None     # the variant's own reference images (project-relative)
    geometry_resolution: str | None = None   # Meshy multi-image: "standard" | "2k" (None = the endpoint default)
    mesh_primary: str | None = None  # the generator whose rig becomes character/rigged.glb (meshy|tripo|hunyuan|hi3d)
    racket_colors: dict[str, list[float]] | None = None   # linear RGB "frame" / "grip" / "strings" (blender/racket.py)


class Curate(BaseModel):
    keep_all: bool = False  # animate every clip; the stroke checks then only flag clips instead of dropping them


class Retarget(BaseModel):
    ground_from_contacts: bool = True
    smooth_window: int = 5
    weld_m: float = 0.0     # merge skinned-mesh vertices closer than this (m) before retargeting; 0 = keep as shipped
    smooth_limb_weights: int = 0      # Laplacian passes over the limb weights before retargeting (0 = off)
    # leg skin weights (blender/retarget.py LEG_WEIGHTS), applied after the smoothing: "meshy" = the auto-rig's;
    # "band" = each knee's thigh/shin split centred on the joint, knee_band_cm either side
    leg_weights: Literal["meshy", "band"] = "meshy"
    knee_band_cm: float = Field(5.0, gt=0)
    # hands (character/hand_rig.json, made by character/hands.py from the rig's own hand mesh): "fixed" = the racket
    # at attach_racket's fixed offsets; "palm" = seated on the palm skin, butt at the heel of the hand, and a constant
    # right-hand offset that keeps the racket aimed as with "fixed" (blender/fingers.py seat_frame)
    grip: Literal["fixed", "palm"] = "fixed"
    fingers: bool = False             # 15 finger bones per hand: the right hand keyed in its grip, the left relaxed
                                      # or, with left_hand_ik, closed as far as the IK holds it on the handle
    left_hand_ik: bool = False        # left hand onto the handle where the wrists are close (blender/twohand.py)
    grip_metrics: bool = False        # the grip numbers in retarget.json on their own (on whenever any of the above is)

    @model_validator(mode="after")
    def _fingers_need_the_palm_seat(self) -> Retarget:
        if self.fingers and self.grip != "palm":   # the grip was solved on the seat, not the fixed grip's buried handle
            raise ValueError("retarget.fingers needs retarget.grip: palm")
        return self

    @property
    def hands(self) -> bool:
        """Whether any hand setting is on (the job then carries hand_rig.json and the grip metrics)."""
        return self.grip != "fixed" or self.fingers or self.left_hand_ik or self.grip_metrics


class Site(BaseModel):
    port: int = 8765


class Config(BaseModel):
    run_dir: Path
    athlete: Athlete
    source: Source
    remote: Remote
    models: Models
    shots: Shots = Shots()
    index: Index = Index()
    select: Select
    clips: Clips = Clips()
    track: Track = Track()
    motion: Motion = Motion()
    character: Character = Character()
    curate: Curate = Curate()
    retarget: Retarget = Retarget()
    site: Site = Site()

    @classmethod
    def load(cls, path: Path | str | None = None) -> Config:
        path = Path(path or os.environ.get("R3D_CONFIG", DEFAULT_CONFIG))
        cfg = cls.model_validate(_read_yaml(path))
        cfg.run_dir = _abs(cfg.run_dir)
        cfg.source.video = _abs(cfg.source.video)
        if cfg.character.refs_dir:
            cfg.character.refs_dir = _abs(cfg.character.refs_dir)
        return cfg

    @cached_property
    def run(self) -> RunLayout:
        return RunLayout(self.run_dir, self.character.variant)

    @property
    def display_name(self) -> str:
        return self.character.name or self.athlete.name


def _read_yaml(path: Path) -> dict:
    """A config file, with `extends: other.yaml` (relative to it) deep-merged underneath."""
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("extends", None)
    return _merge(_read_yaml(path.parent / base), data) if base else data


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _abs(p: Path) -> Path:
    return p if p.is_absolute() else PROJECT_ROOT / p


class ClipLayout:
    """Files belonging to one extracted clip; every stage writes into its own sub-folder."""

    def __init__(self, root: Path):
        self.root = root

    @property
    def id(self) -> str:
        return self.root.name

    meta = property(lambda s: s.root / "clip.json")
    video = property(lambda s: s.root / "clip.mp4")            # model input: every source frame @ model fps
    preview = property(lambda s: s.root / "preview.mp4")       # review copy at the true source rate
    track = property(lambda s: s.root / "track.npz")
    track_preview = property(lambda s: s.root / "track.jpg")
    gvhmr_dir = property(lambda s: s.root / "gvhmr")
    motion = property(lambda s: s.root / "motion.npz")
    overlay = property(lambda s: s.root / "overlay.mp4")       # GVHMR skeleton over the source, source rate
    camera = property(lambda s: s.root / "camera.npz")         # court-line calibration (made on the workstation)
    camera_preview = property(lambda s: s.root / "camera.jpg")
    qa = property(lambda s: s.root / "qa.json")


class RunLayout:
    """Canonical paths inside a run directory, shared by every stage on every machine.

    Footage, clips and motion sit at the run root. The character and everything made from it (anim/, artwork/)
    sit under `out`: the root itself, or variants/<variant>/ for a second character on the same motion, whose
    files are then named <variant>_* instead of rybakina_*. Reports keep paths relative to `root`.
    """

    def __init__(self, root: Path, variant: str | None = None):
        self.root = root
        self.variant = variant
        self.out = root / "variants" / variant if variant else root
        self.stem = variant or "rybakina"

    def ensure(self, *dirs: Path) -> None:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # footage
    source_dir = property(lambda s: s.root / "source")
    probe = property(lambda s: s.root / "source" / "probe.json")
    audio = property(lambda s: s.root / "source" / "audio16k.wav")
    proxy = property(lambda s: s.root / "source" / "proxy.mp4")
    shots = property(lambda s: s.root / "shots.json")
    index_dir = property(lambda s: s.root / "index")
    index = property(lambda s: s.root / "index" / "index.json")
    selection = property(lambda s: s.root / "selection.json")
    clips_dir = property(lambda s: s.root / "clips")

    def clip(self, clip_id: str) -> ClipLayout:
        return ClipLayout(self.clips_dir / clip_id)

    def clip_ids(self) -> list[str]:
        """Clips of the current selection that have been cut (stale folders from older selections are ignored)."""
        if not self.clips_dir.exists():
            return []
        cut = {p.name for p in self.clips_dir.iterdir() if (p / "clip.json").exists()}
        if self.selection.exists():
            selected = [c["id"] for c in json.loads(self.selection.read_text())["clips"]]
            return [c for c in selected if c in cut]
        return sorted(cut)

    def kept_clip_ids(self) -> list[str]:
        """The clips `curate` kept, i.e. the moveset (all cut clips before curation has run)."""
        if self.curation.exists():
            kept = set(json.loads(self.curation.read_text())["kept"])
            return [c for c in self.clip_ids() if c in kept]
        return self.clip_ids()

    curation = property(lambda s: s.root / "curation.json")
    qa_summary = property(lambda s: s.root / "qa_summary.json")
    signature = property(lambda s: s.root / "signature.json")

    # character
    character_dir = property(lambda s: s.out / "character")
    refs_dir = property(lambda s: s.out / "character" / "refs")
    refs = property(lambda s: s.out / "character" / "refs" / "refs.json")
    sheet_dir = property(lambda s: s.out / "character" / "sheet")
    views_dir = property(lambda s: s.out / "character" / "sheet" / "views")   # cut-out turnaround views
    stylize_report = property(lambda s: s.out / "character" / "sheet" / "stylize.json")
    mesh_dir = property(lambda s: s.out / "character" / "mesh")
    mesh_report = property(lambda s: s.out / "character" / "mesh" / "mesh.json")
    rig = property(lambda s: s.out / "character" / "rigged.glb")
    hand_rig = property(lambda s: s.out / "character" / "hand_rig.json")   # + hand_rig.npz (character/hands.py)
    character_report = property(lambda s: s.out / "character" / "character.json")

    # assembly + review
    anim_dir = property(lambda s: s.out / "anim")
    moveset_glb = property(lambda s: s.out / "anim" / f"{s.stem}_moveset.glb")
    moveset_fbx = property(lambda s: s.out / "anim" / f"{s.stem}_moveset.fbx")
    moveset_blend = property(lambda s: s.out / "anim" / f"{s.stem}_moveset.blend")   # stays on the GPU host
    retarget_report = property(lambda s: s.out / "anim" / "retarget.json")
    joints_dir = property(lambda s: s.out / "anim" / "joints")   # retargeted joint tracks for `qa`
    checks_dir = property(lambda s: s.out / "anim" / "checks")   # retarget review renders
    site_dir = property(lambda s: s.root / "site")

    # artwork (character sheets from the game-ready character + captured motion)
    artwork_dir = property(lambda s: s.out / "artwork")
    artwork_poses = property(lambda s: s.out / "artwork" / "poses.json")
    artwork_renders = property(lambda s: s.out / "artwork" / "renders")
    artwork_painted = property(lambda s: s.out / "artwork" / "painted")
    character_sheet = property(lambda s: s.out / "artwork" / f"{s.stem}_character_sheet.png")
    character_sheet_render = property(lambda s: s.out / "artwork" / f"{s.stem}_character_sheet_render.png")
    moveset_atlas = property(lambda s: s.out / "artwork" / f"{s.stem}_moveset_atlas.png")
    artwork_report = property(lambda s: s.out / "artwork" / "artwork.json")
    # artwork in motion (every clip as a video in the sheet's look)
    motion_dir = property(lambda s: s.out / "artwork" / "motion")
    motion_frames = property(lambda s: s.out / "artwork" / "motion" / "frames")      # GPU host only, never pulled
    motion_renders = property(lambda s: s.out / "artwork" / "motion" / "render")
    motion_painted = property(lambda s: s.out / "artwork" / "motion" / "painted")
    moveset_reel = property(lambda s: s.out / "artwork" / "motion" / f"{s.stem}_moveset_reel.mp4")
    moveset_reel_render = property(lambda s: s.out / "artwork" / "motion" / f"{s.stem}_moveset_reel_render.mp4")
    motion_report = property(lambda s: s.out / "artwork" / "motion" / "motion.json")

    def motion_renders_for(self, profile: str = "web") -> Path:
        """Clip videos of one render profile (artwork/layout.py PROFILES): `web` keeps the original folder."""
        return self.motion_renders if profile == "web" else self.motion_dir / f"render_{profile}"
