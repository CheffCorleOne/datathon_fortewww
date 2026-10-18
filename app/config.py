from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

API_KEY = os.environ.get("API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl:4b-instruct")
VLM_ENABLED = os.environ.get("VLM_ENABLED", "1") == "1"
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", ROOT / "uploads"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "300"))
JOB_TIMEOUT_SEC = int(os.environ.get("JOB_TIMEOUT_SEC", "270"))
WORKERS = int(os.environ.get("WORKERS", "1"))

for d in (UPLOAD_DIR, RUNS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)
