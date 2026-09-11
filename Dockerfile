# syntax=docker/dockerfile:1.7
#
# usfeat -- ultrasound feature extraction.
#
# Two stages. The builder downloads model weights (and needs network); the
# runtime carries them, so the container runs fully offline on the
# collaborator's machine.
#
# Build:
#   HF_TOKEN=hf_xxx docker build -t usfeat:latest --secret id=hf_token,env=HF_TOKEN .
#
# The token is mounted as a BuildKit secret rather than passed as a build arg,
# so it does not appear in `docker history` or in any image layer. It is needed
# only to download model weights at build time; the runtime stage has no
# credentials and makes no network calls.

# ------------------------------------------------------------------ builder
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image to a sane size. For a GPU host, rebuild with
# --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch torchvision

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# PyRadiomics needs its own step. The 3.1.0 sdist on PyPI declares its version as
# 3.0.1a1, so pip refuses it as inconsistent metadata; and the git build needs
# versioneer present up front, which build isolation will not provide.
RUN pip install "numpy<2" setuptools wheel versioneer && \
    pip install --no-build-isolation \
      "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics@v3.1.0" && \
    python -c "import radiomics; print('pyradiomics', radiomics.__version__)"

COPY src/ /build/src/
COPY tools/ /build/tools/
COPY config/ /build/config/

# Bake the weights. Not --strict: a gated repo that has not been granted must not
# break the build, it must produce an image where that one extractor is absent
# and says so.
ENV HF_HOME=/opt/usfeat/weights \
    USFEAT_WEIGHTS_DIR=/opt/usfeat/weights
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
    python /build/tools/fetch_weights.py \
        --dest /opt/usfeat/weights \
        --config /build/config/default.yaml

# ------------------------------------------------------------------ runtime
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/usfeat/weights \
    USFEAT_WEIGHTS_DIR=/opt/usfeat/weights \
    PYTHONPATH=/opt/usfeat/src

# libgomp for torch, libglib for opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /opt/usfeat/weights /opt/usfeat/weights

COPY src/     /opt/usfeat/src/
COPY config/  /opt/usfeat/config/
COPY tools/   /opt/usfeat/tools/

# Run as a non-root user; /data is mounted read-only, /out is written.
RUN useradd --create-home --uid 1000 usfeat && \
    mkdir -p /data /out && \
    chown -R usfeat:usfeat /out
USER usfeat

WORKDIR /work
VOLUME ["/data", "/out"]

ENTRYPOINT ["python", "-m", "usfeat.cli"]
CMD ["extract", "--config", "/opt/usfeat/config/default.yaml"]
