"""Turn confirmed events into the jury JSON schema (verdict + violations + evidence frames)."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .rules import RULES, rule_text
from .video import Clip

_face = None


@dataclass
class Event:
    type: str                          # key in rules.RULES
    t: float                           # representative timecode, seconds
    t_start: float
    t_end: float
    score: float
    observation: str                   # what is visible, plain language, no speculation
    evidence_t: list[float] = field(default_factory=list)  # 1-3 timestamps for evidence frames
    boxes: dict[float, list[tuple[int, int, int, int, str]]] = field(default_factory=dict)
    source: str = "cv"                 # cv | vlm | cv+vlm


def _blur_faces(img: np.ndarray) -> np.ndarray:
    """Privacy: blur faces on evidence frames (detection only, no identification)."""
    global _face
    try:
        if _face is None:
            from .faces import FaceBlur
            _face = FaceBlur()
        return _face(img)
    except Exception:  # noqa: BLE001 - never fail a report because of blurring
        return img


def _jpeg(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _annotate(img: np.ndarray, boxes: list[tuple[int, int, int, int, str]], t: float) -> np.ndarray:
    img = img.copy()
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if label:
            cv2.putText(img, label, (x1 + 3, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.rectangle(img, (0, 0), (118, 26), (0, 0, 0), -1)
    cv2.putText(img, f"t={t:.1f}s", (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def build(clip: Clip, events: list[Event]) -> dict[str, Any]:
    violations = []
    for ev in sorted(events, key=lambda e: e.t):
        rule = RULES[ev.type]
        stamps = ev.evidence_t[:3] or [ev.t]
        frames = []
        for ts in stamps:
            i = clip.index_at(ts)
            img = _annotate(_blur_faces(clip.frames[i]), ev.boxes.get(ts, []), clip.times[i])
            frames.append({"timestamp_sec": round(clip.times[i], 2), "image_base64": _jpeg(img)})
        description = f"{ev.observation} {rule['why']} Основание: {rule_text(ev.type)}."
        violations.append({
            "type": rule["title"],
            "timecode_sec": round(ev.t, 2),
            "description": description,
            "evidence_frames": frames,
            # extra, non-schema fields for the UI and for reviewers
            "type_code": ev.type,
            "interval_sec": [round(ev.t_start, 2), round(ev.t_end, 2)],
            "confidence": round(ev.score, 2),
            "rule": {k: rule[k] for k in ("document", "clause", "url", "quote")},
            "detected_by": ev.source,
        })
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
        "video": {"duration_sec": round(clip.duration, 2), "fps": round(clip.fps, 2),
                  "width": clip.native_size[0], "height": clip.native_size[1]},
    }
