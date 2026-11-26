# One image for both services (api + ui); compose picks the command.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch first (keeps the image ~1.5 GB smaller than the CUDA build)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding + reranker models into the image (no download at start-up)
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

COPY app ./app
COPY ui ./ui
COPY eval ./eval
COPY scripts ./scripts
COPY data/pdfs ./data/pdfs

# data/state and logs are volume mount points: they must exist (owned by buddy) in the
# image, so Docker initialises the named volumes with that owner instead of root
RUN useradd --create-home --uid 10001 buddy \
    && mkdir -p /app/data/state /app/data/uploads /app/logs \
    && chown -R buddy:buddy /app /opt/hf
USER buddy

EXPOSE 8080 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8080/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
