"""Render contact sheets (grids of timestamped frames) for manual review.

For violation videos: frames from t-8s to t+8s around each labelled timecode.
For clean videos: frames evenly spread over the whole clip.
Output: data/sheets/<id>[_<t>].jpg
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "open" / "videos"
LABELS = ROOT / "data" / "labels.csv"
OUT = ROOT / "data" / "sheets"


CROP: tuple[float, float, float, float] | None = None  # relative x0, y0, x1, y1


def grab(path: Path, times: list[float], width: int) -> list[tuple[float, np.ndarray]]:
    """Decode the frames closest to each requested time (sequential decode, robust for .mov)."""
    want = sorted(times)
    got: list[tuple[float, np.ndarray]] = []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        i = 0
        for frame in c.decode(s):
            if i >= len(want):
                break
            t = float(frame.pts * s.time_base)
            if t + 1e-3 >= want[i]:
                img = frame.to_ndarray(format="bgr24")
                if CROP:
                    H, W = img.shape[:2]
                    img = img[int(CROP[1] * H):int(CROP[3] * H), int(CROP[0] * W):int(CROP[2] * W)]
                h, w = img.shape[:2]
                img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
                while i < len(want) and t + 1e-3 >= want[i]:
                    got.append((t, img))
                    i += 1
    return got


def sheet(frames: list[tuple[float, np.ndarray]], cols: int, mark: float | None) -> np.ndarray:
    tiles = []
    for t, img in frames:
        img = img.copy()
        color = (0, 0, 255) if mark is not None and abs(t - mark) < 0.51 else (255, 255, 255)
        cv2.rectangle(img, (0, 0), (110, 30), (0, 0, 0), -1)
        cv2.putText(img, f"{t:6.1f}s", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if color == (0, 0, 255):
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 4)
        tiles.append(img)
    h, w = tiles[0].shape[:2]
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)]
    return np.vstack(rows)


def duration(path: Path) -> float:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return float(s.duration * s.time_base) if s.duration else float(c.duration / av.time_base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="video ids to render")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--span", type=float, default=8.0)
    ap.add_argument("--overview", type=int, default=16, help="frames for clean-video overview")
    ap.add_argument("--crop", type=float, nargs=4, help="relative x0 y0 x1 y1 crop before resizing")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--center", type=float, help="override centre time (s)")
    args = ap.parse_args()

    global CROP
    CROP = tuple(args.crop) if args.crop else None
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8")))
    for r in rows:
        if args.only and r["id"] not in args.only:
            continue
        path = VIDEOS / r["file"]
        dur = duration(path)
        if r["t_sec"] or args.center is not None:
            t0 = args.center if args.center is not None else float(r["t_sec"])
            times = [round(x, 2) for x in np.arange(max(0, t0 - args.span), min(dur - 0.05, t0 + args.span) + 1e-6, args.step)]
            frames = grab(path, times, args.width)
            img = sheet(frames, cols=args.cols, mark=t0)
            name = f"{r['id']}_{int(t0)}s{args.suffix}.jpg"
        else:
            times = list(np.linspace(0, max(0, dur - 0.2), args.overview))
            frames = grab(path, times, args.width)
            img = sheet(frames, cols=args.cols if args.cols != 6 else 4, mark=None)
            name = f"{r['id']}_clean{args.suffix}.jpg"
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(f"{name}: {len(frames)} frames, dur {dur:.1f}s")


if __name__ == "__main__":
    main()
