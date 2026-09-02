# Reproducible LIBERO eval image. Daily work uses `uv sync` on the 8 GB laptop;
# this image is the bit-reproducible path for GPU jobs.
#
# Build: docker build -t backstop-libero .
# Run:   docker run --gpus all --rm -e MUJOCO_GL=osmesa \
#          -v $PWD/data:/app/data -v $PWD/artifacts:/app/artifacts \
#          backstop-libero just smoke

FROM huggingface/lerobot-gpu:latest

ENV MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONUNBUFFERED=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md .python-version ./
COPY src ./src
COPY configs ./configs
COPY docs ./docs
COPY justfile ./
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[sim,dev]" \
    && pip install --no-cache-dir "mujoco==3.3.2"

RUN mkdir -p /home/user_lerobot/.libero && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/libero-assets', repo_type='dataset', local_dir='/home/user_lerobot/.libero/assets')" || \
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='lerobot/libero-assets', repo_type='dataset', local_dir='/root/.libero/assets')"

CMD ["bash"]
