"""Multi-object tracking of people (ByteTrack via supervision, MIT) + simple per-track statistics."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import supervision as sv

from .detect import Det


@dataclass
class Track:
    id: int
    t: list[float] = field(default_factory=list)
    idx: list[int] = field(default_factory=list)          # frame indices in the clip
    boxes: list[np.ndarray] = field(default_factory=list)  # x1,y1,x2,y2

    @property
    def centers(self) -> np.ndarray:
        b = np.asarray(self.boxes)
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)

    @property
    def span(self) -> float:
        return self.t[-1] - self.t[0] if self.t else 0.0

    def box_at(self, i: int) -> np.ndarray | None:
        try:
            return self.boxes[self.idx.index(i)]
        except ValueError:
            return None


def track_people(dets: list[list[Det]], times: list[float], fps: float) -> dict[int, Track]:
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))), lost_track_buffer=int(3 * fps),
                           minimum_matching_threshold=0.7, track_activation_threshold=0.4)
    tracks: dict[int, Track] = {}
    for i, (frame_dets, t) in enumerate(zip(dets, times)):
        people = [d for d in frame_dets if d.label == "person"]
        if people:
            xyxy = np.array([d.box for d in people], dtype=np.float32)
            conf = np.array([d.score for d in people], dtype=np.float32)
            det = sv.Detections(xyxy=xyxy, confidence=conf, class_id=np.zeros(len(people), dtype=int))
        else:
            det = sv.Detections.empty()
        out = tracker.update_with_detections(det)
        for box, tid in zip(out.xyxy, out.tracker_id):
            tr = tracks.setdefault(int(tid), Track(int(tid)))
            tr.t.append(t)
            tr.idx.append(i)
            tr.boxes.append(np.asarray(box, dtype=np.float32))
    return tracks


def objects_by_frame(dets: list[list[Det]], label: str) -> dict[int, list[Det]]:
    out: dict[int, list[Det]] = defaultdict(list)
    for i, frame_dets in enumerate(dets):
        for d in frame_dets:
            if d.label == label:
                out[i].append(d)
    return out
