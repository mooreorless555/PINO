# PINO serverless worker for RunPod.
# Mirrors the setup proven on a RunPod Pod: uv-managed Python 3.8.19 venv, torch 2.4.1 (cu124).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MPLBACKEND=Agg \
    UV_LINK_MODE=copy \
    PINO_DIR=/app \
    PATH="/app/.venv/bin:/root/.local/bin:${PATH}" \
    VIRTUAL_ENV="/app/.venv"

# System deps: git (CLIP is a git dependency), and runtime libs for opencv/mmcv/matplotlib.
# ffmpeg is optional (the handler skips video render) but harmless to include.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the repo's required package manager).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

# Resolve dependencies first (better layer caching). pyproject pins Python 3.8.19 + torch cu124.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Same fixes we needed on the Pod, plus the serverless SDK and gdown for the weight download.
RUN uv pip install "lightning_cloud==0.5.57" gdown runpod

# Copy the repo (handler.py, tools/, models/, configs/, prepare/, ...).
COPY . /app

# Bake the 2.6 GB pretrained checkpoint into the image so cold workers start without a download.
# ALTERNATIVE (smaller image, faster build): delete this line and instead attach the `pino_data`
# network volume to the endpoint — handler.py auto-finds the ckpt at /runpod-volume/PINO/checkpoints.
RUN bash prepare/download_pretrain_model.sh

CMD ["python", "-u", "handler.py"]
