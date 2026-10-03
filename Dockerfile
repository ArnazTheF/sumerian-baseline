
FROM python:3.11-slim

ARG TORCH_VARIANT=cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    HOME=/tmp

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*


RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
      --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
 || pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
 && python -c "import torch, torchvision; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

CMD ["python", "-m", "sumbase", "--help"]
