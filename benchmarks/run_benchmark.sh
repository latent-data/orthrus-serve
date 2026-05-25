#!/usr/bin/env bash
#
# Run benchmarks/benchmark.py inside the orthrus-serve container so it sees the
# same torch / transformers / flash_attn build as the server.
#
# Usage:
#   benchmarks/run_benchmark.sh                 # build image, run default prompts
#   benchmarks/run_benchmark.sh --no-build      # skip build, mount source into NGC image
#   benchmarks/run_benchmark.sh --include-nodiff --prompts long --runs 3
#
# Anything other than --no-build is forwarded to benchmark.py.
set -euo pipefail

IMAGE=orthrus-serve
NGC_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NO_BUILD=0

BENCH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-build" ]]; then
        NO_BUILD=1
    else
        BENCH_ARGS+=("$arg")
    fi
done

if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install the NVIDIA driver and try again." >&2
    exit 1
fi

if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "ERROR: NVIDIA Container Toolkit not detected in Docker runtime." >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/results"

COMMON_DOCKER_ARGS=(
    --rm
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    -v "${PROJECT_DIR}:/workspace"
    -w /workspace/benchmarks
)

if [[ "$NO_BUILD" -eq 1 ]]; then
    echo "==> Skipping build; using ${NGC_IMAGE} with mounted source ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" "${NGC_IMAGE}" \
        bash -c 'pip install --no-deps transformers==5.8.1 accelerate==1.13.0 && python -u benchmark.py "$@"' \
        -- "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"
else
    echo "==> Building image ${IMAGE} ..."
    docker build -t "${IMAGE}" "${PROJECT_DIR}"

    echo "==> Running benchmark ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
        python -u benchmark.py "${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}"
fi
