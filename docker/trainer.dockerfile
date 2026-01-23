FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Setup work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Use copy mode for Docker
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONPATH=/app/src

RUN uv sync --frozen --no-dev

COPY src/ ./src/
COPY configs/ ./configs/

ENTRYPOINT ["uv", "run", "src/ai_real_image_classification/train.py"]
