ARG NGC_TAG=25.12-py3
FROM nvcr.io/nvidia/pytorch:${NGC_TAG}

ARG NGC_TAG=25.12-py3
ENV SERVE_IMAGE_TAG=${NGC_TAG}
ENV SERVE_IMAGE_NAME=nvcr.io/nvidia/pytorch:${NGC_TAG}

# Pin both model revisions. Update these when the upstream repos change.
ENV ORTHRUS_REVISION=977a617772e91c966a8cd9b551f4151f9824b6fa
ENV QWEN_REVISION=b968826d9c46dd6066d109eabc6255188de91218

ENV PORT=8080

WORKDIR /workspace

COPY pyproject.toml .
COPY src/ src/

# Install FastAPI/uvicorn/pydantic/prometheus-client normally (no torch conflict).
RUN pip install ".[dev]"

# --no-deps is critical: transformers/accelerate must not pull in a generic torch
# wheel that would overwrite the NGC container's custom sm_121 build.
RUN pip install --no-deps \
    transformers==5.8.1 \
    accelerate==1.13.0

EXPOSE ${PORT}

CMD ["python", "-m", "orthrus_serve.main"]
