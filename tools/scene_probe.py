"""Print the in-room people timeline and enter/exit events for dataset videos (debug aid).

Caches detections in data/cache/<id>_det.pkl so tuning scene logic is fast.
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import scene as scene_mod, video  # noqa: E402

CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
VIDEOS = ROOT / "data" / "open" / "videos"
FPS = 4


def load(vid: str):
    f = CACHE / f"{vid}_det.pkl"
    path = [p for p in VIDEOS.glob("*.mov") if p.stem.replace("Х", "X") == vid][0]
    if f.exists():
        clip_meta, dets = pickle.loads(f.read_bytes())
        return clip_meta, dets
    from app.pipeline.detect import Detector
    clip = video.load(str(path), sample_fps=FPS)
    dets = Detector.get()(clip.frames, thr=0.3)
    meta = {"times": clip.times, "size": clip.size, "fps": clip.fps, "duration": clip.duration}
    f.write_bytes(pickle.dumps((meta, dets)))
    return meta, dets


def main() -> None:
    labels = {}
    for r in csv.DictReader((ROOT / "data" / "labels.csv").open(encoding="utf-8")):
        labels.setdefault(r["canonical"], []).append(r["t_sec"] or "-")
    ids = sys.argv[1:] or sorted(labels, key=lambda v: int(v.split("-")[1]))
    for vid in ids:
        meta, dets = load(vid)
        sc = scene_mod.build(dets, meta["times"], tuple(meta["size"]), FPS)
        runs = []
        for i, c in enumerate(sc.count):
            if not runs or runs[-1][0] != c:
                runs.append([c, meta["times"][i]])
        tl = " ".join(f"{c}@{t:.0f}" for c, t in runs)
        ev = "; ".join(f"{e['kind']} {e['before']}->{e['after']} t={e['t']:.1f}" for e in sc.events)
        print(f"{vid:5s} gt={labels.get(vid)} laptop={'y' if sc.laptop else 'n'} | {tl} | {ev}")


if __name__ == "__main__":
    main()
