"""Run GPU stages on the remote host and keep the run folder in sync.

Push: code + the light run inputs (selection, labels, character).  Pull: everything except heavy
intermediates (model-rate clips, GVHMR caches), so the workstation holds only what the review site needs.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

from .config import PROJECT_ROOT, Config
from .util import log, secret, step

# Made on the workstation, so a pull must never put an older host copy back: camera.npz/jpg, and every artwork
# file except the renders (poses, the repaints and the composed sheets). rsync patterns without a leading / match
# at any depth, so they cover a character variant's variants/<name>/artwork/ as well; <stem>_* names the outputs.
WORKSTATION_ARTWORK = ["artwork/poses.json", "artwork/painted/", "artwork/*_character_sheet*", "artwork/*_moveset_*",
                       "artwork/artwork.json", "artwork/motion/painted/", "artwork/motion/*_moveset_*",
                       "artwork/motion/motion.json", "artwork/motion/motion_*.json"]
# the motion stage's PNG frames are host-side scratch: only its videos come back
PULL_EXCLUDES = ["clips/*/clip.mp4", "clips/*/gvhmr/", "clips/*/camera.npz", "clips/*/camera.jpg", "source/",
                 "*.part.*", "anim/*.blend", "anim/*.blend1", "artwork/motion/frames/", *WORKSTATION_ARTWORK]
# Clips are produced on the GPU host; only workstation-made calibrations travel back up.
PUSH_FILTERS = ["--include=/clips/", "--include=/clips/*/", "--include=/clips/*/camera.npz",
                "--exclude=/clips/*/*"]
# the host renders from poses.json and needs none of the repaints or sheets; the clip videos and their renders.json
# are made on the host, so an older workstation copy must never replace them (it would mark finished clips stale)
PUSH_EXCLUDES = ["artwork/motion/render*/", "source/proxy.mp4", "source/audio16k.wav", "anim/", "site/", "*.part.*",
                 "artwork/painted/", "artwork/*_character_sheet*", "artwork/*_moveset_*", "artwork/artwork.json",
                 "artwork/motion/painted/", "artwork/motion/*_moveset_*", "artwork/motion/motion.json",
                 "artwork/motion/motion_*.json"]


def host(cfg: Config) -> str:
    """user@host of the GPU machine: remote.host in the config, else R3D_HOST (env or .env), which keeps a
    private address out of the committed config."""
    return cfg.remote.host or secret("R3D_HOST")


def _remote_run_dir(cfg: Config) -> str:
    rel = cfg.run_dir.relative_to(PROJECT_ROOT)
    return f"{cfg.remote.root}/{rel}"


# a dropped workstation network must end a remote stage with an error, not leave ssh waiting on a dead connection
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]


def _rsync(src: str, dst: str, excludes: list[str], filters: tuple[str, ...] = ()) -> None:
    cmd = ["rsync", "-az", "-e", shlex.join(["ssh", *SSH_OPTS]), *filters] + [f"--exclude={e}" for e in excludes] \
        + [src, dst]
    subprocess.run(cmd, check=True)


def ssh(cfg: Config, command: str) -> None:
    subprocess.run(["ssh", *SSH_OPTS, host(cfg), command], check=True)


def push_code(cfg: Config) -> None:
    """The code through scripts/sync.sh, to the configured host and root."""
    env = {**os.environ, "R3D_HOST": host(cfg), "R3D_REMOTE_ROOT": cfg.remote.root}
    sync = subprocess.run([str(PROJECT_ROOT / "scripts" / "sync.sh")], env=env, capture_output=True, text=True,
                          errors="replace")
    if sync.returncode:
        raise RuntimeError(f"scripts/sync.sh failed ({sync.returncode}): {sync.stderr.strip()}")


def push(cfg: Config) -> None:
    """Code, then the run inputs."""
    push_code(cfg)
    remote_dir = _remote_run_dir(cfg)
    ssh(cfg, f"mkdir -p {shlex.quote(remote_dir)}")
    _rsync(f"{cfg.run_dir}/", f"{host(cfg)}:{remote_dir}/", PUSH_EXCLUDES, tuple(PUSH_FILTERS))


def pull(cfg: Config) -> None:
    _rsync(f"{host(cfg)}:{_remote_run_dir(cfg)}/", f"{cfg.run_dir}/", PULL_EXCLUDES)


# Publishing sends the workstation's results up (it made most of them) but none of its scratch, and never replaces a
# file the host holds in a newer version (--update); the site is rebuilt there, with links valid on that machine.
PUBLISH_EXCLUDES = ["site/", "*.part.*", "artwork/painted/attempts/", "artwork/painted/inputs/",
                    "artwork/motion/painted/sent/", "artwork/motion/painted/attempts/",
                    "artwork/motion/painted/first/attempts/", "artwork/motion/painted/first/inputs/"]
TUNNEL_BINARY = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def publish(cfg: Config, port: int) -> str:
    """Serve the review site from the GPU host behind a Cloudflare quick tunnel; returns its public URL.

    Code and run results go up, the host rebuilds the site and keeps two background processes under
    ~/<root>/logs (pid files beside their logs): `r3d serve` on 127.0.0.1:<port> and `cloudflared tunnel --url`
    to it. Both are started only when not already running, so re-publishing after new results keeps the URL.
    """
    root = cfg.remote.root
    with step("publish"):
        push_code(cfg)
        _rsync(f"{cfg.run_dir}/", f"{host(cfg)}:{_remote_run_dir(cfg)}/", PUBLISH_EXCLUDES, ("--update",))
        script = f"""set -e
cd ~/{root} && mkdir -p logs bin
.venv/bin/r3d site > logs/site.log 2>&1
running() {{ [ -f "logs/$1.pid" ] && kill -0 "$(cat "logs/$1.pid")" 2>/dev/null; }}
if ! running serve; then
  setsid nohup .venv/bin/r3d serve --port {port} > logs/serve.log 2>&1 < /dev/null & echo $! > logs/serve.pid
fi
if [ ! -x bin/cloudflared ]; then
  curl -fsSL -o bin/cloudflared.part {TUNNEL_BINARY}
  chmod +x bin/cloudflared.part && mv bin/cloudflared.part bin/cloudflared
fi
if ! running tunnel; then
  setsid nohup bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:{port} > logs/tunnel.log 2>&1 < /dev/null &
  echo $! > logs/tunnel.pid
fi
for _ in $(seq 60); do
  url=$(grep -oE 'https://[a-z0-9-]+[.]trycloudflare[.]com' logs/tunnel.log | head -1 || true)
  [ -n "$url" ] && echo "$url" && exit 0
  sleep 1
done
echo "no tunnel URL after 60 s: see ~/{root}/logs/tunnel.log" >&2; exit 1
"""
        out = subprocess.run(["ssh", *SSH_OPTS, host(cfg), script], check=True, capture_output=True, text=True)
        url = out.stdout.strip().splitlines()[-1]
        log(f"review site → [bold]{url}[/] (served from {host(cfg)}:{port})")
    return url


def run_stage(cfg: Config, stage: str, args: list[str]) -> None:
    root = cfg.remote.root
    env = (f"cd ~/{root} && export PATH=$HOME/{root}/bin:$PATH GVHMR_CHECKPOINTS=$HOME/{root}/weights "
           f"GVHMR_BODY_MODELS=$HOME/{root}/body_models R3D_BLENDER={shlex.quote(cfg.remote.blender)} "
           f"PYTHONUNBUFFERED=1")
    if os.environ.get("R3D_CONFIG"):  # same config on both machines (paths are project-relative)
        rel = Path(os.environ["R3D_CONFIG"]).resolve().relative_to(PROJECT_ROOT)
        env += f" R3D_CONFIG={shlex.quote(str(rel))}"
    with step(f"remote {stage}"):
        push(cfg)
        log(f"→ {host(cfg)}: r3d {stage} {' '.join(args)}")
        ssh(cfg, f"{env} && .venv/bin/r3d {shlex.join([*stage.split(), *args])}")
        pull(cfg)
