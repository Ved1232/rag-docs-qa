# ─────────────────────────────────────────────────────────────────
# Dockerfile — Multi-stage build for RAG Pipeline
# ─────────────────────────────────────────────────────────────────

# ── STAGE 1: BUILDER ─────────────────────────────────────────────
# Installs all Python dependencies into /install
# This stage is discarded after build — only /install is kept
FROM python:3.11-slim AS builder

WORKDIR /build

# Copy requirements first — Docker layer cache means pip install
# only re-runs when requirements.txt changes, not on every code change
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── STAGE 2: FINAL IMAGE ─────────────────────────────────────────
FROM python:3.11-slim

# Standard Python Docker best practices
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install curl for the health check
# curl is needed by: HEALTHCHECK CMD curl -f http://localhost:8501/...
# We install it here not in builder so it ends up in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user — security best practice
# Running as root means container escape = root on host
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

# Create required directories and set ownership
RUN mkdir -p chroma_db docs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Document the port — actual mapping happens in docker-compose
EXPOSE 8501

# Health check — docker-compose uses this for depends_on conditions
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit on all interfaces so Docker port mapping works
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]