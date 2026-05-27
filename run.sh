#!/usr/bin/env bash
set -euo pipefail

IMAGE=orthrus-serve
NGC_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=${PORT:-8080}
NO_BUILD=0
DIFFUSION_FLAG=""
DEBUG_FLAG=""
BASE_MODEL_FLAG=""
THINKING_FLAG=""
QUANT_FLAG=""

# --quant takes a value, so we need a small state machine.
next_is_quant=0
for arg in "$@"; do
    if [[ "$next_is_quant" == "1" ]]; then
        QUANT_FLAG="-e ORTHRUS_QUANT=${arg}"
        next_is_quant=0
        continue
    fi
    if [[ "$arg" == "--no-build" ]]; then
        NO_BUILD=1
    elif [[ "$arg" == "--no-diffusion" ]]; then
        DIFFUSION_FLAG="-e ORTHRUS_DIFFUSION=0"
    elif [[ "$arg" == "--debug" ]]; then
        DEBUG_FLAG="-e ORTHRUS_DEBUG=1"
    elif [[ "$arg" == "--with-base-model" ]]; then
        BASE_MODEL_FLAG="-e ORTHRUS_BASE_MODEL=1"
    elif [[ "$arg" == "--enable-thinking" ]]; then
        THINKING_FLAG="-e ORTHRUS_ENABLE_THINKING=true"
    elif [[ "$arg" == "--disable-thinking" ]]; then
        THINKING_FLAG="-e ORTHRUS_ENABLE_THINKING=false"
    elif [[ "$arg" == "--quant" ]]; then
        next_is_quant=1
    fi
done

# --- Prerequisites ---
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install the NVIDIA driver and try again." >&2
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "ERROR: NVIDIA Container Toolkit not detected in Docker runtime." >&2
    echo "  Install it from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/" >&2
    exit 1
fi

CONTAINER_NAME=orthrus-serve

COMMON_DOCKER_ARGS=(
    --rm
    --name "${CONTAINER_NAME}"
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    -p "${PORT}:${PORT}"
    -e PORT="${PORT}"
)

# Remove any stopped container with this name left over from a previous run
# (--rm handles clean exit, but crashes can leave a tombstone)
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

if [[ "$NO_BUILD" -eq 1 ]]; then
    echo "==> Skipping build; using ${NGC_IMAGE} with mounted source ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" ${DIFFUSION_FLAG} ${DEBUG_FLAG} ${BASE_MODEL_FLAG} ${THINKING_FLAG} ${QUANT_FLAG} \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "${NGC_IMAGE}" \
        bash -c 'pip install --no-deps transformers==5.8.1 accelerate==1.13.0 && pip install -e ".[dev]" && python -m orthrus_serve.main'
else
    echo "==> Building image ${IMAGE} ..."
    docker build -t "${IMAGE}" "${SCRIPT_DIR}"

    echo "==> Starting server on port ${PORT} ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" ${DIFFUSION_FLAG} ${DEBUG_FLAG} ${BASE_MODEL_FLAG} ${THINKING_FLAG} ${QUANT_FLAG} "${IMAGE}"
fi
