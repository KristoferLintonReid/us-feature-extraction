#!/usr/bin/env bash
# Convenience wrapper. Handles the volume mounts so you do not have to.
#
#   ./run.sh <data-dir> <output-dir> [extra usfeat args...]
#
# Examples:
#   ./run.sh ~/ovarian_data ~/ovarian_features
#   ./run.sh ~/ovarian_data ~/ovarian_features --limit 10
#   ./run.sh ~/ovarian_data ~/ovarian_features --extractors pyradiomics,dinov2

set -euo pipefail

IMAGE="${USFEAT_IMAGE:-usfeat:latest}"

if [ $# -lt 2 ]; then
    sed -n '2,12p' "$0" | sed 's/^# \?//'
    exit 1
fi

DATA_DIR="$(cd "$1" && pwd)"
shift
OUT_DIR="$1"
shift
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: image '$IMAGE' not found. Build it first:" >&2
    echo "  docker build -t $IMAGE ." >&2
    exit 1
fi

# Built as one list. Note the deliberate avoidance of separate empty arrays:
# expanding an empty array under `set -u` is an error in bash 3.2, which macOS
# still ships, and that fails before the container ever starts.
DOCKER_ARGS=(--rm --shm-size=2g)

if docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    DOCKER_ARGS+=(--gpus all)
    echo "NVIDIA runtime detected; enabling GPU."
fi

echo "data:   $DATA_DIR  (read-only, never leaves this machine)"
echo "output: $OUT_DIR"
echo

# --network none: the container is given no network interface at all. Every
# model weight is baked into the image and nothing in the pipeline makes an
# outbound call, so this costs nothing -- and it means the images cannot leave
# this machine even in principle. Set USFEAT_ALLOW_NETWORK=1 only if you are
# deliberately doing something that needs it.
if [ "${USFEAT_ALLOW_NETWORK:-0}" = "1" ]; then
    echo "WARNING: container network is ENABLED (USFEAT_ALLOW_NETWORK=1)."
else
    DOCKER_ARGS+=(--network none)
fi

# --shm-size (set above) matters: the TexLab payload is decrypted into /dev/shm,
# and the Docker default of 64 MB is not enough for it.
docker run \
    "${DOCKER_ARGS[@]}" \
    -v "$DATA_DIR":/data:ro \
    -v "$OUT_DIR":/out \
    "$IMAGE" \
    extract --data /data --out /out "$@"

echo
echo "done. results in $OUT_DIR"
echo "summary: $OUT_DIR/logs/summary.json"
echo "errors:  $OUT_DIR/logs/errors.csv"
