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

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /tmp/

RUN pip3 install --upgrade pip && \
    ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
        pip3 install -r /tmp/requirements.txt; \
    else \
        pip3 install ultralytics pandas tqdm nvidia-ml-py onnx; \
    fi

WORKDIR /workspace

COPY benchmark.py /workspace/
COPY yolo11s.pt /workspace/

CMD ["python", "benchmark.py"]
