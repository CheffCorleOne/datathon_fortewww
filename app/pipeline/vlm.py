"""Local vision-language model via Ollama (runs on this machine; no cloud calls)."""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

import cv2
import httpx
import numpy as np

from .. import config

log = logging.getLogger("vlm")


def _b64(img: np.ndarray, max_side: int) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def available() -> bool:
    if not config.VLM_ENABLED:
        return False
    try:
        r = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        names = {m["name"] for m in r.json().get("models", [])}
        return config.VLM_MODEL in names or f"{config.VLM_MODEL}:latest" in names
    except Exception:  # noqa: BLE001
        return False


def ask(images: list[np.ndarray], prompt: str, schema: dict | None = None, max_side: int = 896,
        timeout: float = 60.0, num_predict: int = 400) -> dict[str, Any] | str:
    """Send frames + prompt; returns parsed JSON when `schema` is given, else raw text."""
    body: dict[str, Any] = {
        "model": config.VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [_b64(i, max_side) for i in images]}],
        "stream": False,
        "think": False,
        "keep_alive": "60m",
        # 8192 keeps the 4B model + KV cache at ~4.3 GB, leaving room for RT-DETR on an 8 GB card
        "options": {"temperature": 0, "num_ctx": 8192, "num_predict": num_predict},
    }
    if schema is not None:
        body["format"] = schema
    r = httpx.post(f"{config.OLLAMA_URL}/api/chat", json=body, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"ollama {r.status_code}: {r.text[:300]}")
    msg = r.json()["message"]
    text = msg.get("content") or ""
    if not text.strip() and msg.get("thinking"):
        log.error("VLM %s answered only in 'thinking' — use a non-thinking *-instruct model", config.VLM_MODEL)
    if schema is None:
        return text
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                out = json.loads(text[start:end + 1])
                return out if isinstance(out, dict) else {}
            except json.JSONDecodeError:
                pass
        log.warning("VLM returned non-JSON: %s", text[:200])
        return {}
