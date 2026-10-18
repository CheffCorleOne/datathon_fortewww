"""In-process job queue: one GPU worker thread, results persisted to runs/<job_id>/result.json."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config

log = logging.getLogger("jobs")


@dataclass
class Job:
    id: str
    video: Path
    filename: str
    status: str = "processing"
    stage: str = "queued"
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    finished: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        if self.status == "done" and self.result is not None:
            return self.result
        out: dict[str, Any] = {"job_id": self.id, "status": self.status, "stage": self.stage,
                               "progress": round(self.progress, 2)}
        if self.status == "failed":
            out["error"] = self.error
        return out


class JobStore:
    def __init__(self, analyze: Callable[[Path, Callable[[str, float], None]], dict[str, Any]]):
        self._analyze = analyze
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._q: queue.Queue[str] = queue.Queue()
        self._threads = [threading.Thread(target=self._loop, daemon=True, name=f"worker-{i}")
                         for i in range(config.WORKERS)]
        for t in self._threads:
            t.start()

    def submit(self, video: Path, filename: str, job_id: str | None = None) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex[:12], video=video, filename=filename)
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job.id)
        return job

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job.public()
        saved = config.RUNS_DIR / job_id / "result.json"
        if saved.exists():
            return json.loads(saved.read_text(encoding="utf-8"))
        return None

    def queue_position(self, job_id: str) -> int:
        with self._lock:
            pending = [j for j in self._jobs.values() if j.status == "processing" and j.stage == "queued"]
        pending.sort(key=lambda j: j.created)
        ids = [j.id for j in pending]
        return ids.index(job_id) + 1 if job_id in ids else 0

    def _loop(self) -> None:
        while True:
            job_id = self._q.get()
            with self._lock:
                job = self._jobs[job_id]
            self._run(job)

    def _run(self, job: Job) -> None:
        def progress(stage: str, frac: float) -> None:
            job.stage, job.progress = stage, frac

        t0 = time.time()
        run_dir = config.RUNS_DIR / job.id
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            progress("starting", 0.01)
            result = self._analyze(job.video, progress)
            debug = result.pop("debug", None)
            if debug is not None:
                (run_dir / "debug.json").write_text(json.dumps(debug, ensure_ascii=False, default=str),
                                                    encoding="utf-8")
            result = {"job_id": job.id, "status": "done", **result,
                      "processing_time_sec": round(time.time() - t0, 1), "filename": job.filename}
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            job.result, job.status, job.stage, job.progress = result, "done", "done", 1.0
            log.info("job %s done in %.1fs: %s, %d events", job.id, time.time() - t0,
                     result.get("verdict"), len(result.get("violations", [])))
        except Exception as exc:  # noqa: BLE001 - surface any pipeline failure as a failed job
            job.status, job.stage, job.error = "failed", "failed", f"{type(exc).__name__}: {exc}"
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            log.exception("job %s failed", job.id)
        finally:
            job.finished = time.time()
