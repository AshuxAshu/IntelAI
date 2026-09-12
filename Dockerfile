FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libegl1 libgl1 libglib2.0-0 libturbojpeg0 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /workspace
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra cpu --extra dev
COPY . .
