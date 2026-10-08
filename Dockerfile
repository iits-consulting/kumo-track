# syntax=docker/dockerfile:1
# CPU (small, ~2 GB):  docker build -t kumo-track .
# CUDA (~9 GB):        docker build --build-arg TORCH_INDEX=https://pypi.org/simple -t kumo-track:cuda .
#                      run with --gpus all (host needs nvidia-container-toolkit)

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

# CPU-only torch wheels; PyPI's default torch bundles the full CUDA stack.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    UV_LINK_MODE=copy
WORKDIR /app

# torch/torchvision pinned to the versions in uv.lock; the rest resolves from
# PyPI against the already-installed torch. The `postgres` + `azure` extras ship
# the deployment backends (psycopg pool + azure-storage-blob) — both are imported
# lazily at runtime, so they MUST be in the image for DATABASE_URL=postgresql://
# and STORAGE_BACKEND=azure to work.
COPY pyproject.toml ./
RUN uv venv "$VIRTUAL_ENV" && \
    uv pip install --index-url "$TORCH_INDEX" torch==2.12.0 torchvision==0.27.0 && \
    uv pip install -r pyproject.toml --extra postgres --extra azure

FROM python:3.12-slim-bookworm

# opencv-python-headless wheels link against glib
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=build --chown=app:app /app/.venv /app/.venv
# The app resolves its static assets via __file__, so ship the source tree
# as-is (annotate_app.py already puts src/ on sys.path) instead of installing.
COPY --chown=app:app scripts/annotate_app.py scripts/
COPY --chown=app:app src src
# Compile the frontend CSS with the Tailwind standalone CLI (repo has no Node).
# One RUN, no extra stage: ACR Tasks builds with the classic builder, which has
# no BuildKit (`ADD --chmod`) and failed to export a single-file `COPY --from`.
# Tailwind v4 scans for class names relative to the cwd, so cd into static/.
RUN python -c "import urllib.request as u; u.urlretrieve('https://github.com/tailwindlabs/tailwindcss/releases/download/v4.3.3/tailwindcss-linux-x64', '/tmp/tw')" \
    && chmod 755 /tmp/tw \
    && cd src/kumo_track/annotate/static && /tmp/tw -i css/app.css -o css/app.build.css --minify \
    && chown app:app css/app.build.css && rm /tmp/tw
# App config (manual-edit tool). Mount over it or set CONFIG_FILE to change at runtime.
COPY --chown=app:app config.toml ./

# SAM3 weights (~3.3 GB), clips, and the annotation DB live outside the image —
# mount /app/hf-cache, /app/data, /app/outputs to persist them.
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/app/hf-cache \
    PORT=8080
RUN mkdir -p /app/hf-cache /app/data/videos /app/outputs && \
    chown -R app:app /app/hf-cache /app/data /app/outputs

USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/' % os.environ.get('PORT','8080'))" || exit 1

CMD ["python", "scripts/annotate_app.py"]
