# HAWIYA — CPU inference, using the original YOLO + ArcIris models.
# Model checkpoints and the original model driver are mounted at runtime.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=1 \
    HAWIYA_AI_ROOT=/app/HAWIYA_AI

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Explicit CPU wheels avoid accidentally downloading CUDA support on the server.
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
       'torch==2.5.1+cpu' 'torchvision==0.20.1+cpu'

COPY requirements.deploy.txt ./
RUN python -m pip install -r requirements.deploy.txt

COPY server.py engine.py ./
COPY static/ ./static/
COPY hawiya_eye_stock.jpg ./hawiya_eye_stock.jpg

RUN mkdir -p /app/demo_cache && chown -R 10001:10001 /app
USER 10001:10001
EXPOSE 8800
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8800/api/status', timeout=6)" || exit 1
CMD ["sh", "-c", "python -u server.py --ensure-blue && exec python -m uvicorn server:app --host 0.0.0.0 --port 8800 --workers 1"]
