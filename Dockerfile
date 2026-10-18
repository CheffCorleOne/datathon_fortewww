# Service image (FastAPI + RT-DETR). The VLM runs in the separate `ollama` service, see docker-compose.yml.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf
WORKDIR /app

RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
      "pydantic>=2.7" "httpx>=0.27" "av>=12.0" "opencv-python-headless>=4.10" "pillow>=10.3" \
      "transformers>=4.50" "accelerate>=1.0" "supervision>=0.25"

COPY app ./app

# Pre-fetch model weights at build time so the container needs no internet at runtime.
RUN python -c "from huggingface_hub import snapshot_download as d; \
[d(m) for m in ('PekingU/rtdetr_v2_r50vd', 'caidas/swin2SR-classical-sr-x2-64', 'usyd-community/vitpose-base-simple')]" && \
    python -c "from app.pipeline.faces import ensure_model; ensure_model()"

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
