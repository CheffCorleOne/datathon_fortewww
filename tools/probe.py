"""Print container metadata (duration, fps, resolution, rotation) for every dataset video."""
from __future__ import annotations

import csv
from pathlib import Path

import av

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "open" / "videos"
OUT = ROOT / "data" / "probe.csv"


def main() -> None:
    rows = []
    for p in sorted(VIDEOS.glob("*.mov")):
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            dur = float(s.duration * s.time_base) if s.duration else float(c.duration / av.time_base)
            rot = s.metadata.get("rotate") or ""
            rows.append({
                "file": p.name,
                "w": s.codec_context.width,
                "h": s.codec_context.height,
                "fps": round(float(s.average_rate), 2),
                "dur": round(dur, 2),
                "frames": s.frames,
                "codec": s.codec_context.name,
                "rotate": rot,
            })
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
