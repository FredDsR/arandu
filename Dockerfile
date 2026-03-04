# Arandu Docker Image
# Multi-stage build for optimized image size using official uv images

# Stage 1: Builder - Use official uv image
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files and source code
# Note: src/ must be present before sync for the package to be installed
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY prompts/ prompts/

# Install dependencies and the arandu package using uv sync
# --frozen ensures exact lockfile reproduction without re-resolving
# PyTorch CUDA 12.4 versions are configured via tool.uv.sources in pyproject.toml
# This installs both dependencies AND the arandu CLI tool
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Install runtime dependencies (cache mounts reduce peak disk usage during build
# and speed up rebuilds by reusing downloaded .deb packages)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/prompts /app/prompts
COPY --from=builder /app/pyproject.toml /app/pyproject.toml

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create directories for volumes
RUN mkdir -p /app/input /app/results /app/credentials

# Default command
ENTRYPOINT ["arandu"]
CMD ["--help"]
