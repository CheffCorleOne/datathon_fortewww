"""Object detection with RT-DETRv2 (Apache-2.0, via transformers) on GPU when available."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np
import torch

log = logging.getLogger("detect")

MODEL_ID = "PekingU/rtdetr_v2_r50vd"
KEEP = {"person", "cell phone", "handbag", "backpack", "suitcase", "book", "laptop", "remote",
        "bottle", "cup", "scissors", "keyboard", "mouse", "tv", "chair"}


@dataclass
class Det:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    label: str
    score: float


class Detector:
    _inst: "Detector | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, torch_dtype=self.dtype).to(self.device).eval()
        self.id2label = self.model.config.id2label
        log.info("RT-DETR loaded on %s", self.device)

    @classmethod
    def get(cls) -> "Detector":
        with cls._lock:
            if cls._inst is None:
                cls._inst = Detector()
            return cls._inst

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray], thr: float = 0.35, batch: int = 8,
                 progress=None) -> list[list[Det]]:
        out: list[list[Det]] = []
        for i in range(0, len(frames), batch):
            chunk = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames[i:i + batch]]  # BGR -> RGB
            inputs = self.proc(images=chunk, return_tensors="pt").to(self.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
            res = self.model(**inputs)
            sizes = torch.tensor([f.shape[:2] for f in chunk], device=self.device)
            posts = self.proc.post_process_object_detection(res, threshold=thr, target_sizes=sizes)
            for p in posts:
                dets = []
                for score, lab, box in zip(p["scores"].tolist(), p["labels"].tolist(), p["boxes"].tolist()):
                    name = self.id2label[int(lab)]
                    if name in KEEP:
                        dets.append(Det(tuple(box), name, float(score)))
                out.append(dets)
            if progress:
                progress("detecting", 0.15 + 0.35 * min(1.0, (i + batch) / len(frames)))
        return out
