# syntax=docker/dockerfile:1

# Both base images are pinned to exact digests, like the actions in .github/workflows. Dependabot proposes updates.

# uv, the tool that installs the locked dependencies (0.12.23)
FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv

# --- Build stage: install the locked runtime dependencies and the package into a virtual environment ------------------
# python 3.14.8
FROM python:3.14.8-slim-trixie@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2 AS builder

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so that this layer is reused when only the code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the package itself (with the files of the web page), installed as a regular package and not editable, so the
# final image needs no source tree.
COPY README.md LICENSE ./
COPY chat_with_any_website_agent ./chat_with_any_website_agent
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# --- Final stage: only the virtual environment, run as an unprivileged user ----------------------------------------
FROM python:3.14.8-slim-trixie@sha256:f85c5697265c178cc6887276c55fe16cf3d14ca35c3df6a5eab3b360534a55d2 AS runtime

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

USER app

EXPOSE 8000

# /api/health answers at once and calls no outside service. The port is the PORT that hosting platforms set (Render
# does), or 8000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT', '8000'), timeout=3)"]

# Settings come from environment variables (see .env.example); no .env file is copied into the image.
# Behind a proxy, set FORWARDED_ALLOW_IPS to the proxy's address so that each visitor's own address is used for the
# limits per visitor; uvicorn reads that variable. WEB_ALLOWED_HOSTS lists the names the server answers to (on Render the
# name of the service is added by itself). The factory builds the application from those settings.
CMD ["sh", "-c", "exec uvicorn --factory chat_with_any_website_agent.web:create_app_from_environment --host 0.0.0.0 --port ${PORT:-8000} --no-access-log"]
