# ── Stage 1: build wheel ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir build>=1.2 && \
    python -m build --wheel --outdir /dist


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user
RUN useradd --create-home --shell /bin/bash mlops

WORKDIR /app

# Install the wheel with the web extra (FastAPI + uvicorn + jinja2)
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl"[web]" && rm /tmp/*.whl

# Pre-create default directories that the serve command reads from
RUN mkdir -p /app/runs /app/outputs/model_registry /app/outputs/dataset_registry && \
    chown -R mlops:mlops /app

USER mlops

# Expose default dashboard port
EXPOSE 8000

# Mount points for run state and registry data
VOLUME ["/app/runs", "/app/outputs"]

# Health check — the dashboard root should return 200
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

ENTRYPOINT ["agentic-mlops"]
CMD ["serve", "--port", "8000", \
     "--runs-dir", "/app/runs", \
     "--registry-dir", "/app/outputs/model_registry", \
     "--dataset-registry-dir", "/app/outputs/dataset_registry"]
