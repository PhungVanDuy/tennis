"""fal.ai helpers shared by the stages that call hosted models (character mesh + rig, artwork paint-over).

The key comes from FAL_KEY or FAL_API (env or .env) and only ever travels to fal; uploads go to fal's CDN.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import requests

from .util import log, secret


def client():
    os.environ.setdefault("FAL_KEY", secret("FAL_KEY", "FAL_API"))
    import fal_client

    return fal_client


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):  # decodes Content-Encoding (Tripo serves gzip)
                f.write(chunk)
    return dest


def subscribe(endpoint: str, args: dict) -> dict:
    fal = client()
    t0 = time.perf_counter()
    log(f"fal → {endpoint}")
    res = fal.subscribe(endpoint, arguments=args, with_logs=False)
    log(f"fal ← {endpoint} in {time.perf_counter() - t0:.0f}s")
    return res


def upload(path: Path) -> str:
    return client().upload_file(str(path))
