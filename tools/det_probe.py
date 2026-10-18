"""Run the detector over a video and print per-second object counts (debug aid)."""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import video  # noqa: E402
from app.pipeline.detect import Detector  # noqa: E402


def main() -> None:
    name = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
    path = next((ROOT / "data" / "open" / "videos").glob(f"*{name.split('-')[1]}.mov"))
    path = [p for p in (ROOT / "data" / "open" / "videos").glob("*.mov") if p.stem.replace("Х", "X") == name][0]
    t0 = time.time()
    clip = video.load(str(path), sample_fps=4)
    t1 = time.time()
    det = Detector.get()
    t2 = time.time()
    dets = det(clip.frames, thr=thr)
    t3 = time.time()
    print(f"{name}: {len(clip.frames)} frames {clip.size} decode {t1 - t0:.1f}s load {t2 - t1:.1f}s detect {t3 - t2:.1f}s")
    for i in range(0, len(dets), 4):
        c = Counter(d.label for d in dets[i])
        extra = " ".join(f"{d.label}:{d.score:.2f}@{int(d.box[0])},{int(d.box[1])},{int(d.box[2])},{int(d.box[3])}"
                         for d in dets[i] if d.label != "person")
        print(f"{clip.times[i]:6.1f}s person={c.get('person', 0)} {extra}")


if __name__ == "__main__":
    main()
