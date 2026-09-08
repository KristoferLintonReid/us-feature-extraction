# usfeat -- ultrasound feature extraction.
#
# Two stages. The builder downloads model weights (and needs network); the
# runtime carries them, so the container runs fully offline on the collaborator's
# machine. Octave is installed for the TexLab module.
#
# Build:
#   docker build -t usfeat:latest \
#     --build-arg HF_TOKEN=hf_xxx \
#     --build-arg TEXLAB_KEY="$(cat build/texlab.key)" .
#
# The TexLab payload (build/texlab.enc) must exist before building; produce it
# with tools/pack_texlab.py. Without it the image still builds and every other
# feature family works -- TexLab simply reports itself unavailable.

# ------------------------------------------------------------------ builder
FROM python:3.11-slim-bookworm AS builder

ARG HF_TOKEN=""
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

COPY src/ /build/src/
COPY tools/ /build/tools/
COPY config/ /build/config/

# Bake the weights. Not --strict: a gated repo that has not been granted must not
# break the build, it must produce an image where that one extractor is absent
# and says so.
ENV HF_TOKEN=${HF_TOKEN} \
    HF_HOME=/opt/usfeat/weights \
    USFEAT_WEIGHTS_DIR=/opt/usfeat/weights
RUN python /build/tools/fetch_weights.py \
        --dest /opt/usfeat/weights \
        --config /build/config/default.yaml

# ------------------------------------------------------------------ runtime
FROM python:3.11-slim-bookworm AS runtime

ARG TEXLAB_KEY=""
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/opt/usfeat/weights \
    USFEAT_WEIGHTS_DIR=/opt/usfeat/weights \
    PYTHONPATH=/opt/usfeat/src

# GNU Octave and the three packages TexLAB_cli loads (statistics, image,
# parallel), plus libgomp for torch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        octave \
        octave-statistics \
        octave-image \
        octave-parallel \
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

# The encrypted TexLab payload. Optional: the wildcard means the build succeeds
# whether or not build/texlab.enc exists.
COPY build*/texlab.en[c] /opt/usfeat/texlab/

# The payload key, baked in so the collaborator does not need one. See
# docs/SECURITY.md for exactly what this protects against and what it does not.
RUN mkdir -p /opt/usfeat/texlab && \
    if [ -n "${TEXLAB_KEY}" ]; then \
        printf '%s' "${TEXLAB_KEY}" > /opt/usfeat/texlab/texlab.key && \
        chmod 0400 /opt/usfeat/texlab/texlab.key ; \
    fi

# Run as a non-root user; /data is mounted read-only, /out is written.
RUN useradd --create-home --uid 1000 usfeat && \
    mkdir -p /data /out && \
    chown -R usfeat:usfeat /out /opt/usfeat/texlab
USER usfeat

WORKDIR /work
VOLUME ["/data", "/out"]

ENTRYPOINT ["python", "-m", "usfeat.cli"]
CMD ["extract", "--config", "/opt/usfeat/config/default.yaml"]
