"""FastAPI entry point: jury API (/api/v1, X-API-Key) + browser UI (/ and /ui/*)."""
from __future__ import annotations

import logging
import secrets
import shutil
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .jobs import JobStore
from .pipeline.analyze import analyze_video, warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="Cash-desk discipline video checker", version="1.0")
store = JobStore(analyze_video)
STATIC = Path(__file__).parent / "static"
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if not config.API_KEY or not x_api_key or not secrets.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


async def _accept(file: UploadFile) -> dict:
    name = file.filename or "video.mov"
    ext = Path(name).suffix.lower() or ".mov"
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"unsupported file type {ext}; expected .mov")
    job_id = store.new_id()
    dst = config.UPLOAD_DIR / f"{job_id}{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    with dst.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                out.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"file larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)
    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="empty file")
    job = store.submit(dst, name, job_id=job_id)
    log.info("accepted %s (%.1f MB) as job %s", name, size / 1e6, job.id)
    return {"job_id": job.id, "status": "processing"}


def _job(job_id: str) -> JSONResponse:
    data = store.get(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="job not found")
    if data.get("status") == "processing":
        data = {**data, "queue_position": store.queue_position(job_id)}
    return JSONResponse(data)


# ---- jury API -------------------------------------------------------------

@app.post("/api/v1/analyze", dependencies=[Depends(require_key)])
async def api_analyze(file: UploadFile = File(...)):
    return await _accept(file)


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
def api_job(job_id: str):
    return _job(job_id)


# ---- browser UI (same handlers, no key so the reviewer can just open the page) ----

@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _accept(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return _job(job_id)


@app.get("/ui/jobs/{job_id}/video")
def ui_job_video(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="bad job id")
    files = sorted(config.UPLOAD_DIR.glob(f"{job_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="video not found")
    media = "video/quicktime" if files[0].suffix.lower() == ".mov" else "video/mp4"
    return FileResponse(files[0], media_type=media)


@app.get("/ui/rules")
def ui_rules():
    from .pipeline.rules import catalog
    return catalog()


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def _startup() -> None:
    if not config.API_KEY:
        log.warning("API_KEY is not set: /api/v1 will reject every request")
    warmup()


def _cleanup_uploads(keep: int = 50) -> None:
    files = sorted(config.UPLOAD_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[keep:]:
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
