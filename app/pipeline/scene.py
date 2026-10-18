"""Scene dynamics from detections: who is in the room, entries/exits, where the workstation is.

Everything here is relative to the clip itself (no fixed pixel zones), so it survives a camera move:
- people seen through the glass walls are dropped by relative size (they are far, hence small);
- the workstation is wherever the laptop is detected most consistently;
- the "cashier" is the in-room person closest to the workstation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Det


@dataclass
class Scene:
    times: list[float]
    people: list[list[Det]]              # in-room people per frame
    count: np.ndarray                    # smoothed in-room people count per frame
    laptop: tuple[float, float, float, float] | None
    frame_size: tuple[int, int]          # (w, h)
    events: list[dict] = field(default_factory=list)
    landmarks: dict = field(default_factory=dict)  # machine / safe / door boxes from the VLM
    machine_act: np.ndarray | None = None           # motion energy at the counting machine, per frame

    def t_index(self, t: float) -> int:
        return int(np.argmin(np.abs(np.asarray(self.times) - t)))

    track_of: list[dict] = field(default_factory=list)  # per frame: track id -> box
    cashier_track: int | None = None

    def cashier_box(self, i: int) -> tuple[float, float, float, float] | None:
        """The cashier = the longest-present track; fall back to the person nearest the workstation."""
        ps = self.people[i]
        if not ps:
            return None
        if self.cashier_track is not None and i < len(self.track_of):
            b = self.track_of[i].get(self.cashier_track)
            if b is not None:
                return b
            # cashier track lost in this frame: take the person closest to where it was last seen
            for j in range(i - 1, max(-1, i - 40), -1):
                last = self.track_of[j].get(self.cashier_track)
                if last is not None:
                    lc = _center(last)
                    best = min(ps, key=lambda d: np.hypot(*(np.subtract(_center(d.box), lc))))
                    if np.hypot(*(np.subtract(_center(best.box), lc))) < 1.0 * (last[3] - last[1]):
                        return best.box
                    break
        if self.laptop is None:
            return max(ps, key=lambda d: _area(d.box)).box
        lx, ly = _center(self.laptop)
        return min(ps, key=lambda d: np.hypot(*(np.subtract(_center(d.box), (lx, ly))))).box


def _area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _center(b) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def _laptop_box(dets: list[list[Det]]):
    boxes = [max((d for d in fd if d.label == "laptop"), key=lambda d: d.score).box
             for fd in dets if any(d.label == "laptop" and d.score >= 0.6 for d in fd)]
    if len(boxes) < max(3, len(dets) // 10):
        return None
    return tuple(np.median(np.asarray(boxes), axis=0).tolist())


def build(dets: list[list[Det]], times: list[float], frame_size: tuple[int, int], fps: float) -> Scene:
    w, h = frame_size
    all_people = [d for fd in dets for d in fd if d.label == "person" and d.score >= 0.8]
    ref = np.percentile([_area(d.box) for d in all_people], 75) if all_people else 0.03 * w * h
    min_area = max(0.35 * ref, 0.012 * w * h)

    people = []
    for fd in dets:
        keep = [d for d in fd if d.label == "person" and d.score >= 0.6 and _area(d.box) >= min_area]
        # drop near-duplicate boxes (same person detected twice)
        keep.sort(key=lambda d: -d.score)
        uniq: list[Det] = []
        for d in keep:
            if all(_iou(d.box, u.box) < 0.6 for u in uniq):
                uniq.append(d)
        people.append(uniq)

    raw = np.array([len(p) for p in people], dtype=float)
    k = max(3, int(round(1.75 * fps)) | 1)
    count = np.round(_median_filter(raw, k)).astype(int)
    scene = Scene(times=times, people=people, count=count, laptop=_laptop_box(dets), frame_size=frame_size)
    scene.events = _events(scene, min_hold=max(2, int(round(1.5 * fps))))
    scene.track_of, lengths = _track(people, max_gap=int(3 * fps))
    if lengths:
        scene.cashier_track = max(lengths, key=lengths.get)
    return scene


def _track(people: list[list[Det]], max_gap: int) -> tuple[list[dict], dict]:
    """Greedy IoU/centre tracker; returns per-frame {track_id: box} and track lengths in frames."""
    active: dict[int, tuple[int, tuple]] = {}  # id -> (last frame, last box)
    per_frame: list[dict] = []
    lengths: dict[int, int] = {}
    next_id = 0
    for i, ps in enumerate(people):
        assigned: dict[int, tuple] = {}
        cands = sorted(ps, key=lambda d: -d.score)
        for d in cands:
            best, best_cost = None, None
            for tid, (lf, lb) in active.items():
                if tid in assigned or i - lf > max_gap:
                    continue
                iou = _iou(d.box, lb)
                dist = np.hypot(*(np.subtract(_center(d.box), _center(lb)))) / max(1.0, lb[3] - lb[1])
                if iou < 0.2 and dist > 0.6:
                    continue
                cost = dist - iou
                if best_cost is None or cost < best_cost:
                    best, best_cost = tid, cost
            if best is None:
                best = next_id
                next_id += 1
            assigned[best] = d.box
            active[best] = (i, d.box)
            lengths[best] = lengths.get(best, 0) + 1
        per_frame.append(assigned)
    return per_frame, lengths


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def _events(scene: Scene, min_hold: int) -> list[dict]:
    """Stable level changes of the in-room count: enter / exit / empty."""
    c = scene.count
    ev: list[dict] = []
    if len(c) == 0:
        return ev
    level, start = c[0], 0
    segments = []
    for i in range(1, len(c) + 1):
        if i == len(c) or c[i] != level:
            segments.append((int(level), start, i - 1))
            if i < len(c):
                level, start = c[i], i
    # merge segments shorter than min_hold into neighbours
    stable = [s for s in segments if s[2] - s[1] + 1 >= min_hold or s is segments[-1] or s is segments[0]]
    for prev, cur in zip(stable, stable[1:]):
        if cur[0] == prev[0]:
            continue
        t = scene.times[cur[1]]
        t_last = scene.times[prev[2]]
        kind = "enter" if cur[0] > prev[0] else "exit"
        ev.append({"kind": kind, "t": t, "t_prev_end": t_last, "before": prev[0], "after": cur[0],
                   "i": cur[1], "i_prev_end": prev[2], "hold_end": scene.times[cur[2]]})
    return ev
