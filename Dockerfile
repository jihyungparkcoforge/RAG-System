FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY main.py ./
COPY src ./src
COPY docs ./docs

# Run as a non-root user; the volume mount points must exist and be writable.
RUN useradd --create-home app \
    && mkdir -p /app/chroma_data /models \
    && chown app:app /app/chroma_data /models
USER app

ENTRYPOINT ["python", "main.py"]
