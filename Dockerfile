# FrameBucket data service image. The same image runs the API (default command) and the worker:
#   docker run ... <image>                         -> API on port 8001
#   docker run ... <image> python -m app.worker    -> background worker
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

# Writable work area for temp files and the downloaded face-recognition models.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/tmp /data/face-models \
    && chown -R app:app /data
USER app

ENV FB_TEMP_ROOT=/data/tmp \
    FB_FACE_MODEL_DIR=/data/face-models \
    PORT=8001

EXPOSE 8001
CMD ["sh", "-c", "exec python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
