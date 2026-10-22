"""`r3d` command line: one sub-command per pipeline stage (scripts/run_pipeline.sh runs them in order).

Workstation:  ingest · shots · index · select · character refs|stylize|mesh · camera · signature · site · serve
GPU host:     clips · track · motion · curate · qa · retarget  (the host's run folder holds the clips)
`r3d remote <stage> …` runs a GPU-host stage on the configured host and syncs the run folder both ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .config import Config

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_show_locals=False)
character_app = typer.Typer(no_args_is_help=True, help="Stylized character: refs → views → mesh → rig.")
app.add_typer(character_app, name="character")
artwork_app = typer.Typer(no_args_is_help=True, help="Character sheets from the game-ready character + her motion.")
app.add_typer(artwork_app, name="artwork")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Pipeline YAML (default configs/rybakina.yaml)")]
ClipsOpt = Annotated[list[str] | None, typer.Option("--clip", help="Limit to these clip ids (repeatable)")]
ForceOpt = Annotated[bool, typer.Option("--force", help="Recompute even if outputs exist")]
ProfileOpt = Annotated[str, typer.Option("--profile", help="Render profile (layout.PROFILES): web | quest3 | draft")]


def _cfg(config: Path | None) -> Config:
    return Config.load(config)


@app.command()
def ingest(config: ConfigOpt = None):
    """Probe the source video, extract audio and the upload proxy."""
    from .footage import ingest as run

    run(_cfg(config))


@app.command()
def shots(config: ConfigOpt = None):
    """Split the compilation into shots (hard cuts)."""
    from .footage import detect_shots

    detect_shots(_cfg(config))


@app.command()
def index(config: ConfigOpt = None, force: ForceOpt = False):
    """Label every shot with Gemini: camera view, replay, Rybakina's court side, strokes + contact times."""
    from .index.labeler import run

    run(_cfg(config), force=force)


@app.command()
def select(config: ConfigOpt = None, force: ForceOpt = False):
    """Pick the best stroke windows, refine contact on the audio, locate the hitter on the contact frame."""
    from .selection import run

    run(_cfg(config), force=force)


@app.command()
def clips(config: ConfigOpt = None, clip: ClipsOpt = None, force: ForceOpt = False):
    """Cut the selected windows into model-ready clips + review previews."""
    from .clips import run

    run(_cfg(config), only=clip, force=force)


@app.command()
def track(config: ConfigOpt = None, clip: ClipsOpt = None, force: ForceOpt = False):
    """Track people and lock onto Rybakina in every clip (GPU)."""
    from .tracking import run

    run(_cfg(config), only=clip, force=force)


@app.command()
def camera(config: ConfigOpt = None, clip: ClipsOpt = None, force: ForceOpt = False):
    """Calibrate the broadcast camera from the court lines: per-frame intrinsics + rotation."""
    from .camera import run

    run(_cfg(config), only=clip, force=force)


@app.command()
def motion(config: ConfigOpt = None, clip: ClipsOpt = None, force: ForceOpt = False):
    """World-grounded 3D motion with GVHMR (GPU)."""
    from .motion.recover import run

    run(_cfg(config), only=clip, force=force)


@app.command()
def curate(config: ConfigOpt = None):
    """Validate each clip's motion against its stroke label, refine contact, keep the best per stroke."""
    from .curate import run

    run(_cfg(config))


@app.command()
def signature(config: ConfigOpt = None):
    """Measure her stroke signature (knee flex, coil, contact height, toss…) on the kept clips."""
    from .signature import run

    run(_cfg(config))


@app.command()
def retarget(config: ConfigOpt = None, clip: ClipsOpt = None):
    """Retarget the kept clips' motions onto the rigged character in Blender; export GLB + FBX."""
    from .assemble import run

    run(_cfg(config), only=clip)


@app.command()
def qa(config: ConfigOpt = None):
    """Per-clip motion quality (foot skate, ground contact, jitter, reprojection, timing) → PASS / WARN / FAIL."""
    from .qa import run

    run(_cfg(config))


@character_app.command("refs")
def character_refs(config: ConfigOpt = None):
    """Collect identity + style references (client deck, public photos)."""
    from .character.refs import run

    run(_cfg(config))


@character_app.command("stylize")
def character_stylize(config: ConfigOpt = None, force: ForceOpt = False):
    """Hero image in the client art style, then the A-pose turnaround views (Gemini image)."""
    from .character.stylize import run

    run(_cfg(config), force=force)


@character_app.command("mesh")
def character_mesh(config: ConfigOpt = None, force: ForceOpt = False):
    """Image-to-3D + auto-rig through fal (Meshy, with Tripo as the A/B mesh)."""
    from .character.mesh import run

    run(_cfg(config), force=force)


@character_app.command("face")
def character_face(config: ConfigOpt = None):
    """Project the art team's headshot onto the rig's face and bake it into the base colour (art-team variants)."""
    from .character.face import run

    run(_cfg(config))


@artwork_app.command("poses")
def artwork_poses(config: ConfigOpt = None):
    """Resolve every sheet slot to a clip, frame and camera from the captured motion."""
    from .artwork.poses import run

    run(_cfg(config))


@artwork_app.command("render")
def artwork_render(config: ConfigOpt = None, force: ForceOpt = False):
    """Render every slot + the moveset atlas from the moveset GLB in Blender (GPU host)."""
    from .artwork.render import run

    run(_cfg(config), force=force)


@artwork_app.command("paint")
def artwork_paint(config: ConfigOpt = None, force: ForceOpt = False):
    """Repaint the renders in the reference sheet's style (GPT Image 2.5 Sunburst on fal), pose-checked."""
    from .artwork.paint import run

    run(_cfg(config), force=force)


@artwork_app.command("compose")
def artwork_compose(config: ConfigOpt = None):
    """Lay out the character sheet (painted + render-only versions) and the moveset atlas."""
    from .artwork.compose import run

    run(_cfg(config))


@artwork_app.command("motion-render")
def artwork_motion_render(config: ConfigOpt = None, clip: ClipsOpt = None, profile: ProfileOpt = "web",
                          force: ForceOpt = False):
    """Render every clip (+ a turntable) as a video in the sheet's look from the moveset GLB (GPU host)."""
    from .artwork.motion_render import run

    run(_cfg(config), force=force, only=clip, profile=profile)


@artwork_app.command("motion-paint")
def artwork_motion_paint(config: ConfigOpt = None, force: ForceOpt = False):
    """Carry a painted first frame through each clip with a video model (fal), silhouette-checked per frame."""
    from .artwork.motion_paint import run

    run(_cfg(config), force=force)


@artwork_app.command("motion-compose")
def artwork_motion_compose(config: ConfigOpt = None, profile: ProfileOpt = "web"):
    """The moveset reel: every clip in a grid, aligned on contact (painted where accepted + an all-render twin)."""
    from .artwork.motion_compose import run

    run(_cfg(config), profile=profile)


@app.command()
def site(config: ConfigOpt = None):
    """Build the static review site from the run folder."""
    from .site import build

    build(_cfg(config))


@app.command()
def serve(config: ConfigOpt = None, port: int | None = None):
    """Serve the review site locally."""
    from .site import serve as run

    cfg = _cfg(config)
    run(cfg, port or cfg.site.port)


@app.command()
def publish(config: ConfigOpt = None, port: int | None = None):
    """Serve the review site from the GPU host with a public Cloudflare quick-tunnel URL (results synced up first)."""
    from .remote import publish as run

    cfg = _cfg(config)
    print(run(cfg, port or cfg.site.port))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def remote(ctx: typer.Context, stage: str, config: ConfigOpt = None):
    """Run a stage on the GPU host: push code + run inputs, execute, pull outputs."""
    from .remote import run_stage

    run_stage(_cfg(config), stage, list(ctx.args))


if __name__ == "__main__":
    app()
