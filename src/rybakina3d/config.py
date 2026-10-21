"""Typed pipeline configuration (configs/*.yaml) and the on-disk layout of a run."""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

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


class Curate(BaseModel):
    keep_all: bool = False  # animate every clip; the stroke checks then only flag clips instead of dropping them


class Retarget(BaseModel):
    ground_from_contacts: bool = True
    smooth_window: int = 5


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
        cfg = cls.model_validate(yaml.safe_load(path.read_text()))
        cfg.run_dir = _abs(cfg.run_dir)
        cfg.source.video = _abs(cfg.source.video)
        return cfg

    @cached_property
    def run(self) -> RunLayout:
        return RunLayout(self.run_dir)


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
    """Canonical paths inside a run directory, shared by every stage on every machine."""

    def __init__(self, root: Path):
        self.root = root

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
    character_dir = property(lambda s: s.root / "character")
    refs_dir = property(lambda s: s.root / "character" / "refs")
    refs = property(lambda s: s.root / "character" / "refs" / "refs.json")
    sheet_dir = property(lambda s: s.root / "character" / "sheet")
    views_dir = property(lambda s: s.root / "character" / "sheet" / "views")   # cut-out turnaround views
    stylize_report = property(lambda s: s.root / "character" / "sheet" / "stylize.json")
    mesh_dir = property(lambda s: s.root / "character" / "mesh")
    mesh_report = property(lambda s: s.root / "character" / "mesh" / "mesh.json")
    rig = property(lambda s: s.root / "character" / "rigged.glb")
    character_report = property(lambda s: s.root / "character" / "character.json")

    # assembly + review
    anim_dir = property(lambda s: s.root / "anim")
    moveset_glb = property(lambda s: s.root / "anim" / "rybakina_moveset.glb")
    moveset_fbx = property(lambda s: s.root / "anim" / "rybakina_moveset.fbx")
    moveset_blend = property(lambda s: s.root / "anim" / "rybakina_moveset.blend")   # stays on the GPU host
    retarget_report = property(lambda s: s.root / "anim" / "retarget.json")
    joints_dir = property(lambda s: s.root / "anim" / "joints")   # retargeted joint tracks for `qa`
    checks_dir = property(lambda s: s.root / "anim" / "checks")   # retarget review renders
    site_dir = property(lambda s: s.root / "site")

    # artwork (character sheets from the game-ready character + captured motion)
    artwork_dir = property(lambda s: s.root / "artwork")
    artwork_poses = property(lambda s: s.root / "artwork" / "poses.json")
    artwork_renders = property(lambda s: s.root / "artwork" / "renders")
    artwork_painted = property(lambda s: s.root / "artwork" / "painted")
    character_sheet = property(lambda s: s.root / "artwork" / "rybakina_character_sheet.png")
    character_sheet_render = property(lambda s: s.root / "artwork" / "rybakina_character_sheet_render.png")
    moveset_atlas = property(lambda s: s.root / "artwork" / "rybakina_moveset_atlas.png")
    artwork_report = property(lambda s: s.root / "artwork" / "artwork.json")
    # artwork in motion (every clip as a video in the sheet's look)
    motion_dir = property(lambda s: s.root / "artwork" / "motion")
    motion_frames = property(lambda s: s.root / "artwork" / "motion" / "frames")      # GPU host only, never pulled
    motion_renders = property(lambda s: s.root / "artwork" / "motion" / "render")
    motion_painted = property(lambda s: s.root / "artwork" / "motion" / "painted")
    moveset_reel = property(lambda s: s.root / "artwork" / "motion" / "rybakina_moveset_reel.mp4")
    moveset_reel_render = property(lambda s: s.root / "artwork" / "motion" / "rybakina_moveset_reel_render.mp4")
    motion_report = property(lambda s: s.root / "artwork" / "motion" / "motion.json")
