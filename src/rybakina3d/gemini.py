"""Thin Gemini API client: cached video upload + structured (Pydantic) generation with retries."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .util import log, read_json, secret, write_json

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def client():
    """Process-wide client (a dropped Client closes its HTTP pool, so keep one alive)."""
    from google import genai

    return genai.Client(api_key=secret("GEMINI_API_KEY"))


def upload_video(path: Path, cache: Path):
    """Upload once; reuse the File while the API still holds it (files expire after ~48 h)."""
    c = client()
    if cache.exists():
        meta = read_json(cache)
        try:
            f = c.files.get(name=meta["name"])
            if f.state.name == "ACTIVE" and meta.get("size") == path.stat().st_size:
                return f
        except Exception:  # noqa: BLE001 - expired or deleted: upload again
            pass
    log(f"uploading {path.name} ({path.stat().st_size / 1e6:.0f} MB) to the Gemini File API")
    f = c.files.upload(file=str(path), config={"mime_type": "video/mp4", "display_name": path.name})
    while f.state.name == "PROCESSING":
        time.sleep(5)
        f = c.files.get(name=f.name)
    if f.state.name != "ACTIVE":
        raise RuntimeError(f"upload failed: {f.state}")
    write_json(cache, {"name": f.name, "uri": f.uri, "size": path.stat().st_size})
    return f


def generate(model: str, parts: list, schema: type[T], *, temperature: float = 0.2,
             media_resolution: str | None = None, retries: int = 4) -> T:
    """Structured generation: `parts` -> an instance of `schema`, retried with exponential back-off."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        temperature=temperature,
        media_resolution=media_resolution,
    )
    c = client()
    err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = c.models.generate_content(model=model, contents=[types.Content(role="user", parts=parts)],
                                             config=cfg)
            if resp.parsed is not None:
                return resp.parsed
            return schema.model_validate_json(resp.text)
        except Exception as e:  # noqa: BLE001 - transient API / parse errors: back off and retry
            err = e
            wait = 4 * 2**attempt
            log(f"[yellow]gemini {model} attempt {attempt + 1} failed: {str(e)[:160]} — retry in {wait}s[/]")
            time.sleep(wait)
    raise RuntimeError(f"gemini {model} failed after {retries} attempts") from err


def video_part(file, start_s: float, end_s: float, fps: float):
    from google.genai import types

    return types.Part(
        file_data=types.FileData(file_uri=file.uri, mime_type=file.mime_type),
        video_metadata=types.VideoMetadata(start_offset=f"{start_s:.2f}s", end_offset=f"{end_s:.2f}s", fps=fps),
    )


def image_part(jpeg: bytes):
    from google.genai import types

    return types.Part.from_bytes(data=jpeg, mime_type="image/jpeg")


def text_part(text: str):
    from google.genai import types

    return types.Part(text=text)
