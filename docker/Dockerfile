# ============================================================
# Stage 1: Builder — install Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for building wheels (PyPDF, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

# Install all deps into a prefix we can copy later
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    httpx>=0.26.0 \
    python-multipart>=0.0.22 \
    python-docx \
    pypdf \
    langfuse>=2.0.0 \
    accelerate>=0.20.0 \
    transformers>=4.30.0

# ============================================================
# Stage 2: Runtime — slim final image
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/

# Create dirs the app expects
RUN mkdir -p logs data/documents_ro .cache

# Expose the API port
EXPOSE 8000

# Health check — matches the /health endpoint
# start-period is long to allow model downloads on first boot
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run via uvicorn (workers=1 because embedding models are loaded in-process)
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
