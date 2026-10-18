"""Ask the local VLM a question about frames at given times (debug aid).

usage: python tools/vlm_probe.py X-20 5,6,7 "question"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import video, vlm  # noqa: E402


def frames_at(name: str, times: list[float], max_side: int = 960):
    path = [p for p in (ROOT / "data" / "open" / "videos").glob("*.mov") if p.stem.replace("Х", "X") == name][0]
    clip = video.load(str(path), sample_fps=4, max_side=max_side)
    return [clip.frames[clip.index_at(t)] for t in times]


def main() -> None:
    name, ts, q = sys.argv[1], sys.argv[2], sys.argv[3]
    times = [float(x) for x in ts.split(",")]
    imgs = frames_at(name, times)
    t0 = time.time()
    out = vlm.ask(imgs, q, max_side=int(sys.argv[4]) if len(sys.argv) > 4 else 896, num_predict=300)
    print(f"[{time.time() - t0:.1f}s] {out}")


if __name__ == "__main__":
    main()
