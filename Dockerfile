FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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
    pip3 install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip3 install -r /tmp/requirements.txt

WORKDIR /workspace

COPY benchmark.py /workspace/
COPY yolo11s.pt /workspace/

CMD ["python", "benchmark.py"]
