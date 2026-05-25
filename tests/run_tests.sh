#!/usr/bin/env bash
#
# Run pytest inside the orthrus-serve container so torch / transformers /
# flash_attn match what the server uses. Mirrors benchmarks/run_benchmark.sh.
#
# Usage:
#   tests/run_tests.sh                  # build image, run all tests
#   tests/run_tests.sh --no-build       # skip docker build, mount source into NGC image
#   tests/run_tests.sh tests/test_endpoint.py -v   # any extra args forwarded to pytest
set -euo pipefail

IMAGE=orthrus-serve
NGC_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
NO_BUILD=0

PYTEST_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-build" ]]; then
        NO_BUILD=1
    else
        PYTEST_ARGS+=("$arg")
    fi
done

if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. The image will still build but flash_attn may fail to initialise." >&2
fi

COMMON_DOCKER_ARGS=(
    --rm
    --gpus all
    --ipc=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${PROJECT_DIR}:/workspace"
    -w /workspace
)

if [[ "$NO_BUILD" -eq 1 ]]; then
    echo "==> Skipping build; using ${NGC_IMAGE} with mounted source ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" "${NGC_IMAGE}" \
        bash -c 'pip install --no-deps transformers==5.8.1 accelerate==1.13.0 && pip install -e ".[dev]" && pytest "$@"' \
        -- "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
else
    echo "==> Building image ${IMAGE} ..."
    docker build -t "${IMAGE}" "${PROJECT_DIR}"

    echo "==> Running pytest ..."
    docker run "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
        pytest "${PYTEST_ARGS[@]+"${PYTEST_ARGS[@]}"}"
fi
