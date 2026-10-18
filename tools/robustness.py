"""Make view-changed copies of dataset videos to approximate "another camera angle".

Variants: horizontal mirror, 15% off-centre crop + rescale, small rotation, perspective warp.
Output: data/robust/<variant>/<file>.mp4 — feed to tools/eval.py via --videos-dir.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data" / "open" / "videos"
OUT = ROOT / "data" / "robust"


def transform(name: str, w: int, h: int):
    if name == "mirror":
        return lambda f: cv2.flip(f, 1)
    if name == "crop":
        x0, y0 = int(0.12 * w), int(0.05 * h)
        x1, y1 = int(0.97 * w), int(0.90 * h)
        return lambda f: cv2.resize(f[y0:y1, x0:x1], (w, h))
    if name == "rotate":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 8, 1.1)
        return lambda f: cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if name == "perspective":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0.08 * w, 0.04 * h], [0.95 * w, 0], [w, h], [0.02 * w, 0.92 * h]])
        m = cv2.getPerspectiveTransform(src, dst)
        return lambda f: cv2.warpPerspective(f, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    raise ValueError(name)


def convert(src: Path, dst: Path, variant: str, max_side: int = 1280) -> None:
    with av.open(str(src)) as cin:
        s = cin.streams.video[0]
        w0, h0 = s.codec_context.width, s.codec_context.height
        sc = min(1.0, max_side / max(w0, h0))
        w, h = int(w0 * sc) // 2 * 2, int(h0 * sc) // 2 * 2
        fn = transform(variant, w, h)
        rate = s.average_rate or 30
        dst.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(dst), "w") as cout:
            o = cout.add_stream("libx264", rate=rate)
            o.width, o.height, o.pix_fmt = w, h, "yuv420p"
            o.options = {"crf": "23", "preset": "veryfast"}
            for frame in cin.decode(s):
                img = cv2.resize(frame.to_ndarray(format="bgr24"), (w, h))
                nf = av.VideoFrame.from_ndarray(fn(img), format="bgr24")
                for p in o.encode(nf):
                    cout.mux(p)
            for p in o.encode():
                cout.mux(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["mirror", "crop", "perspective"])
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for v in args.variants:
        for src in sorted(VIDEOS.glob("*.mov")):
            if args.only and src.stem.replace("Х", "X") not in args.only:
                continue
            dst = OUT / v / src.name
            if not dst.exists():
                convert(src, dst, v)
                print(v, src.name)


if __name__ == "__main__":
    main()
