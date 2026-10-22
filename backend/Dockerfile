# Build context: repository root (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf

# No apt packages needed: torch, ctranslate2 (faster-whisper), PyAV and soundfile wheels vendor their native libs.
WORKDIR /app

# CPU wheels by default (small image). For NVIDIA GPUs build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# GPU use is auto-detected at runtime (ctranslate2 / torch.cuda).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX_URL} torch==2.5.1 torchaudio==2.5.1

COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install -r requirements.txt -r requirements-ml.txt

COPY backend/ /app/
COPY schemas/ /app/schemas/
COPY seed/ /app/seed/
COPY eval/ /app/eval/

ENV SCHEMAS_DIR=/app/schemas \
    SEED_DIR=/app/seed \
    EVAL_DIR=/app/eval \
    BACKEND_DIR=/app \
    AUDIO_DIR=/data/audio

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
