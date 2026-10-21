"""Small shared helpers: logging, JSON / npz io, secrets."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console

from .config import PROJECT_ROOT

console = Console(stderr=True, highlight=False)


def log(msg: str) -> None:
    console.print(f"[dim]{time.strftime('%H:%M:%S')}[/] {msg}")


@contextmanager
def step(name: str):
    t0 = time.perf_counter()
    log(f"[bold]{name}[/] …")
    yield
    log(f"[bold]{name}[/] done in {time.perf_counter() - t0:.1f}s")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default))
    tmp.replace(path)
    return path


def _json_default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def read_npz(path: Path) -> dict[str, np.ndarray]:
    """Every array of an .npz archive, loaded eagerly (the file is closed again)."""
    with np.load(path) as z:
        return dict(z)


def secret(*names: str) -> str:
    """First non-empty value among env vars / the project .env (keys are never logged)."""
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        for n in names:
            if values.get(n):
                return values[n]
    raise RuntimeError(f"missing secret: one of {names} (env or .env)")
