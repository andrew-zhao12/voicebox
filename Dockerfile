# ============================================================
# Voicebox — TTS/STT server with the web UI
# 3-stage build: Frontend → Python deps → Runtime
#
# Build variants (PYTORCH_VARIANT build arg):
#   cpu   (default)  docker compose up --build
#   cu128 (NVIDIA)   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
#   rocm  (AMD)      docker compose -f docker-compose.yml -f docker-compose.rocm.yml up --build
# Any other PyTorch CUDA index name (cu126, cu130) works too; see the torch step.
# ============================================================

# Top-level ARG so it is visible to all stages.
ARG PYTORCH_VARIANT=cpu

# === Stage 1: Build frontend ===
FROM oven/bun:1 AS frontend

WORKDIR /build

# Copy workspace config and frontend source
COPY package.json bun.lock CHANGELOG.md ./
COPY app/ ./app/
COPY web/ ./web/

# Normalize line endings first (a Windows CRLF checkout would otherwise
# defeat the `-z 's/,\n  ]/…/'` match below, since it's LF-anchored), then
# strip workspaces not needed for web build, and fix trailing comma
RUN sed -i 's/\r$//' package.json && \
    sed -i '/"tauri"/d; /"landing"/d' package.json && \
    sed -i -z 's/,\n  ]/\n  ]/' package.json
RUN bun install --no-save
# Build frontend (skip tsc — upstream has pre-existing type errors)
RUN cd web && bunx --bun vite build


# === Stage 2: Python dependencies ===
# python:3.12-slim — the backend uses 3.12 syntax (pyproject: requires-python >= 3.12).
FROM python:3.12-slim AS backend-builder

# Re-declare ARGs inside the stage (Docker scoping requirement).
ARG PYTORCH_VARIANT=cpu
# ROCm wheel index. Default 6.3 (RDNA1/2/3); set ROCM_VERSION=7.2 for RDNA4.
ARG ROCM_VERSION=6.3
# torch/torchaudio are the only packages requirements.lock does not pin, because
# the variant picks their wheels.  Empty TORCH_VERSION = the variant's default.
ARG TORCH_VERSION=
ARG TORCHAUDIO_VERSION=2.11.0

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than --prefix: pip then sees the torch installed below
# and never resolves a second copy from PyPI while installing the lock.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip

# 1. torch for the chosen variant.  The cu128 index stops at torch 2.11 (newer
#    builds moved to cu126/cu130) and the ROCm indexes lag further, so ROCm
#    takes whatever its index currently has.  Every later install keeps the
#    variant index primary so nothing pulls the default CUDA torch from PyPI.
RUN set -eu; \
    case "$PYTORCH_VARIANT" in \
      cpu)   index="https://download.pytorch.org/whl/cpu";   default_torch="2.14.0" ;; \
      cu128) index="https://download.pytorch.org/whl/cu128"; default_torch="2.11.0" ;; \
      cu*)   index="https://download.pytorch.org/whl/${PYTORCH_VARIANT}"; default_torch="2.14.0" ;; \
      rocm)  index="https://download.pytorch.org/whl/rocm${ROCM_VERSION}"; default_torch="" ;; \
      *) echo "PYTORCH_VARIANT must be cpu, cu<version> or rocm (got '$PYTORCH_VARIANT')" >&2; exit 1 ;; \
    esac; \
    version="${TORCH_VERSION:-$default_torch}"; \
    if [ -n "$version" ]; then \
      spec="torch==${version} torchaudio==${TORCHAUDIO_VERSION}"; \
    else \
      spec="torch torchaudio"; \
    fi; \
    pip install --no-cache-dir --index-url "$index" $spec; \
    printf '[global]\nindex-url = %s\nextra-index-url = https://pypi.org/simple\n' "$index" > /etc/pip.conf

# 2. Everything else, pinned. scripts/lock-backend.sh regenerates the lock.
COPY backend/requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# 3. Two packages whose own pins conflict with the stack, installed without
#    dependencies (their real dependencies are already in the lock).
RUN pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7 hume-tada==0.1.9


# === Stage 3: Runtime ===
FROM python:3.12-slim

# Create non-root user; the entrypoint joins GPU device groups at runtime.
RUN groupadd -r voicebox && \
    useradd -r -g voicebox -m -s /bin/bash voicebox

WORKDIR /app

# Install only runtime system dependencies (gosu drops root in the entrypoint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# The virtualenv from the builder stage (same base image, so its python symlink resolves)
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Copy backend application code
COPY --chown=voicebox:voicebox backend/ /app/backend/

# Copy built frontend from frontend stage
COPY --from=frontend --chown=voicebox:voicebox /build/web/dist /app/frontend/

# Create data directories owned by non-root user
RUN mkdir -p /app/data/generations /app/data/profiles /app/data/cache \
    && chown -R voicebox:voicebox /app/data

# Expose the API port
EXPOSE 17493

# Liveness: /health answers without a key as soon as the port is open.
# Readiness (models resident) is GET /health/ready — point load balancers there.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:17493/health || exit 1

# Entrypoint joins GPU groups then drops to the voicebox user.
# Normalize CRLF (a Windows checkout otherwise leaves the shebang as
# `#!/bin/sh\r`, which Linux can't resolve — reported as a misleading
# "no such file or directory" even though the file exists).
COPY --chmod=755 scripts/rocm-entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# SIGTERM: uvicorn stops accepting, waits up to 40 s for open responses (streams,
# SSE), then the app drains its generation queue (VOICEBOX_DRAIN_TIMEOUT_S).
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "17493", "--timeout-graceful-shutdown", "40"]
