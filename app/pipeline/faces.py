"""Face blurring for evidence frames (YuNet detector, OpenCV Zoo, MIT). Detection only — no identification."""
from __future__ import annotations

import urllib.request

import cv2
import numpy as np

from .. import config

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = config.MODELS_DIR / "face_detection_yunet_2023mar.onnx"


def ensure_model() -> None:
    if not YUNET_PATH.exists():
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)


class FaceBlur:
    def __init__(self, score: float = 0.6):
        ensure_model()
        self.det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score, 0.3, 50)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(img)
        if faces is None:
            return img
        out = img.copy()
        for f in faces:
            x, y, fw, fh = (int(v) for v in f[:4])
            pad = int(0.25 * max(fw, fh))
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = out[y1:y2, x1:x2]
            k = max(15, (max(x2 - x1, y2 - y1) // 3) | 1)
            out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out
