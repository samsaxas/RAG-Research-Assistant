# syntax=docker/dockerfile:1
#
# Build (production embeddings, ~2 GB image, needs >= 2 GB RAM at runtime):
#   docker build -t rag-research-analyst .
# Build (light image, TF-IDF fallback, ~300 MB, fits a 512 MB free tier):
#   docker build --build-arg EMBEDDING_BACKEND=tfidf -t rag-research-analyst:light .
#
# The same image runs on Render, Google Cloud Run, AWS (ECS/App Runner) or any Docker host: it listens on
# $PORT (default 8080) and takes all configuration from environment variables.

FROM python:3.11-slim

ARG EMBEDDING_BACKEND=huggingface
ENV EMBEDDING_BACKEND=${EMBEDDING_BACKEND} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    UPLOAD_DIR=/app/data/uploads

WORKDIR /app

# --- dependencies first (this layer is cached until requirements change) ---
COPY requirements-api.txt requirements-hf.txt ./
RUN pip install -r requirements-api.txt
# CPU-only PyTorch is ~200 MB; the default Linux wheel bundles CUDA and is ~2 GB.
RUN if [ "$EMBEDDING_BACKEND" = "huggingface" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu \
        && pip install -r requirements-hf.txt; \
    fi

# --- run as a non-root user ---
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/uploads /app/.cache/huggingface \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser api ./api
COPY --chown=appuser:appuser data/sample_docs ./data/sample_docs
# The real-world document (data/real_docs/NIST.AI.100-1.pdf, fetched with scripts/fetch_real_doc.py). If it is
# present at build time the service indexes it instead of the 3 tiny samples; the README file keeps the folder valid.
COPY --chown=appuser:appuser data/real_docs ./data/real_docs

USER appuser

# Bake the embedding model into the image so containers start fast and never depend on huggingface.co at runtime.
RUN if [ "$EMBEDDING_BACKEND" = "huggingface" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"; \
    fi

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,sys,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8080'), timeout=4)" || exit 1

# One worker on purpose: the index lives in this process's memory (see README, "Scaling notes").
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
