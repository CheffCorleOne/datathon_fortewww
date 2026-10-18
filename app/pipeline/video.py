"""Video decoding: sample frames at a fixed rate with exact presentation timestamps."""
from __future__ import annotations

from dataclasses import dataclass

import av
import cv2
import numpy as np


@dataclass
class Clip:
    frames: list[np.ndarray]      # BGR, resized so the long side <= max_side
    times: list[float]            # seconds from video start, per frame
    fps: float                    # native fps
    duration: float
    native_size: tuple[int, int]  # (w, h)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frames[0].shape[:2]
        return w, h

    def index_at(self, t: float) -> int:
        arr = np.asarray(self.times)
        return int(np.argmin(np.abs(arr - t)))


def load(path: str, sample_fps: float = 5.0, max_side: int = 960, progress=None) -> Clip:
    frames: list[np.ndarray] = []
    times: list[float] = []
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate or 30)
        if s.duration:
            duration = float(s.duration * s.time_base)
        elif c.duration:
            duration = float(c.duration / av.time_base)
        else:
            duration = 0.0
        w, h = s.codec_context.width, s.codec_context.height
        scale = min(1.0, max_side / max(w, h))
        step = 1.0 / sample_fps
        next_t = 0.0
        start = None
        for frame in c.decode(s):
            if frame.pts is None:
                continue
            t = float(frame.pts * s.time_base)
            if start is None:
                start = t
            t -= start
            if t + 1e-3 < next_t:
                continue
            img = frame.to_ndarray(format="bgr24")
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            frames.append(img)
            times.append(round(t, 3))
            next_t += step
            while next_t <= t:
                next_t += step
            if progress and duration:
                progress("decoding", min(0.15, 0.15 * t / duration))
    if not frames:
        raise ValueError("no decodable video frames")
    duration = max(duration, times[-1])
    return Clip(frames=frames, times=times, fps=fps, duration=duration, native_size=(w, h))
