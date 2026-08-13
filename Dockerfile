FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=60 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MPLBACKEND=Agg

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /usr/local/bin/uv

# Resolve dependencies from the lockfile before copying source, so edits to src/ don't
# invalidate the dependency layer. --frozen fails loudly if uv.lock is out of date
# rather than silently resolving something different from what was tested.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

# The platform injects PORT; 8081 is only the local default.
ENV PORT=8081
EXPOSE 8081

CMD ["python", "src/main.py"]
