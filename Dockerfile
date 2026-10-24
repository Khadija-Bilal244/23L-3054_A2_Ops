# Slim base + CPU-only TensorFlow (tensorflow-cpu drops the ~500 MB of CUDA libs; inference needs no GPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /srv

# 1) dependencies first: this layer is rebuilt only when requirements change,
#    so editing app code reuses the cached (slow, large) pip layer.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# 2) source code last (changes often)
COPY app ./app

# non-root user; the model is mounted read-only at /models (not baked in, no secrets in the image)
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"5000\")}/health', timeout=4)"

# web UI by default; the API service overrides this with: python -m app.api
CMD ["python", "-m", "app.main"]
