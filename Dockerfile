FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/
ARG TORCH_VARIANT=cu128
ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    HOME=/tmp
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra train --extra corpus --extra ${TORCH_VARIANT}
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
CMD ["python", "-m", "sumbase", "--help"]
