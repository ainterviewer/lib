# syntax=docker/dockerfile:1

# The runtime server: one image per runtime version, `python -m
# ainterviewer.worker` on port 8700.

# --- base: shared environment ---
FROM python:3.12-slim AS base
LABEL org.opencontainers.image.source="https://github.com/ainterviewer/lib"

ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_NO_MANAGED_PYTHON=1 \
  UV_PYTHON_DOWNLOADS=never \
  PYTHONUNBUFFERED=1 \
  PATH="/app/.venv/bin:$PATH"

# --- build: the library and the worker extra, into a venv ---
FROM base AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.13 /uv /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --no-dev --extra worker --no-install-project

# Installed, not editable: the runtime stage takes the venv alone.
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --locked --no-dev --extra worker --no-editable

# --- runtime: the venv and nothing else ---
FROM base AS runtime

# Set by the release workflow: a preview's runtime version name (a release
# leaves it empty and is named by its library version), and the commit built.
ARG RUNTIME_VERSION=""
ARG WORKER_BUILD=""
ENV RUNTIME_VERSION=${RUNTIME_VERSION} \
  WORKER_BUILD=${WORKER_BUILD} \
  WORKER_PORT=8700 \
  AINTERVIEWER_AGENTS_LOG=-

RUN useradd --system --no-create-home worker
COPY --from=build /app/.venv /app/.venv
WORKDIR /app
USER worker

EXPOSE 8700
HEALTHCHECK CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ['WORKER_PORT']}/health\")"]
ENTRYPOINT ["python", "-m", "ainterviewer.worker"]
