ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    wget \
    nano \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp/

RUN pip3 install --upgrade pip && \
    ARCH=$(uname -m) && \
    export CARGO_NET_GIT_FETCH_WITH_CLI=true && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
        pip3 install -r /tmp/requirements.txt; \
    else \
        python3 -c "import site, os; d = site.getsitepackages()[0]; os.makedirs(os.path.join(d, 'polars-0.20.31.dist-info'), exist_ok=True); open(os.path.join(d, 'polars-0.20.31.dist-info', 'METADATA'), 'w').write('Metadata-Version: 2.1\nName: polars\nVersion: 0.20.31\n')" && \
        pip3 install ultralytics pandas tqdm nvidia-ml-py onnx; \
    fi

ARG PYTHON_SRC_DIR=phase0_synthetic

WORKDIR /workspace

COPY ${PYTHON_SRC_DIR}/benchmark.py /workspace/
COPY models/yolo11s.pt /workspace/
COPY shared/gpu_monitor.py /workspace/

CMD ["python", "benchmark.py"]
