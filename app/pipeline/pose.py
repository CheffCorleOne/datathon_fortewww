"""Cashier wrists via ViTPose (Apache-2.0, transformers), top-down on the cashier box.

Used only around manual-counting candidates, so it costs a few dozen frames per video, not the whole clip.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import torch

log = logging.getLogger("pose")
MODEL_ID = "usyd-community/vitpose-base-simple"
L_WRIST, R_WRIST = 9, 10  # COCO keypoint order


class Pose:
    _inst: "Pose | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = VitPoseForPoseEstimation.from_pretrained(MODEL_ID).to(self.device).eval()
        log.info("ViTPose loaded on %s", self.device)

    @classmethod
    def get(cls) -> "Pose":
        with cls._lock:
            if cls._inst is None:
                cls._inst = Pose()
            return cls._inst

    @torch.inference_mode()
    def wrists(self, frames: list[np.ndarray], boxes: list[tuple]) -> list[list[tuple[float, float, float]]]:
        """For each (frame, person box x1y1x2y2): [(x, y, score) for left and right wrist]."""
        out: list[list[tuple[float, float, float]]] = []
        for i in range(0, len(frames), 8):
            fr = [np.ascontiguousarray(f[:, :, ::-1]) for f in frames[i:i + 8]]
            bx = [[[b[0], b[1], b[2] - b[0], b[3] - b[1]]] for b in boxes[i:i + 8]]  # COCO xywh, one person
            inputs = self.proc(fr, boxes=bx, return_tensors="pt").to(self.device)
            res = self.model(**inputs)
            posts = self.proc.post_process_pose_estimation(res, boxes=bx)
            for p in posts:
                kp = p[0]["keypoints"].cpu().numpy()
                sc = p[0]["scores"].cpu().numpy()
                out.append([(float(kp[k][0]), float(kp[k][1]), float(sc[k])) for k in (L_WRIST, R_WRIST)])
        return out
