"""Frame-quality middleware: neural super-resolution for the small crops sent to the VLM.

Swin2SR classical x2 (Apache-2.0) — a restoration model, not a GAN, so it sharpens real detail instead
of inventing texture (important when the question is "is this a banknote or a phone case?").
Only crops are enhanced; full frames for the detector stay untouched to keep the 5-minute budget.
"""
from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
import torch

log = logging.getLogger("enhance")
MODEL_ID = "caidas/swin2SR-classical-sr-x2-64"
MAX_IN_SIDE = 320  # larger crops are already sharp enough; also bounds the attention cost


class SuperRes:
    _inst: "SuperRes | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = Swin2SRForImageSuperResolution.from_pretrained(MODEL_ID).to(self.device).eval()
        log.info("Swin2SR loaded on %s", self.device)

    @classmethod
    def get(cls) -> "SuperRes":
        with cls._lock:
            if cls._inst is None:
                cls._inst = SuperRes()
            return cls._inst

    @torch.inference_mode()
    def x2(self, bgr: np.ndarray) -> np.ndarray:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        inputs = self.proc(rgb, return_tensors="pt").to(self.device)
        out = self.model(**inputs).reconstruction.squeeze(0).float().clamp(0, 1).cpu().numpy()
        out = (np.moveaxis(out, 0, -1) * 255.0).round().astype(np.uint8)
        h, w = bgr.shape[:2]
        out = out[: 2 * h, : 2 * w]  # the processor pads to a window multiple
        return np.ascontiguousarray(out[:, :, ::-1])


def upscale(crop: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize `crop` to `size` (w, h); when enlarging a small crop, go through x2 super-resolution first."""
    w, h = size
    ch, cw = crop.shape[:2]
    if max(cw, ch) <= MAX_IN_SIDE and (w > 1.3 * cw or h > 1.3 * ch):
        try:
            crop = SuperRes.get().x2(crop)
        except Exception as exc:  # noqa: BLE001 - enhancement must never break the analysis
            log.warning("super-resolution failed, falling back to bicubic: %s", exc)
    interp = cv2.INTER_AREA if crop.shape[1] > w else cv2.INTER_CUBIC
    return cv2.resize(crop, (w, h), interpolation=interp)
