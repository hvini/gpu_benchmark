#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <engine> <precision>"
    echo "Example: $0 tensorrt fp16"
    echo "Valid Engines: pytorch, onnx, tensorrt"
    echo "Valid Precisions: fp16, fp32, int8"
    exit 1
fi

ENGINE=$1
PRECISION=$2
ARCH=$(uname -m)

echo "Detected Architecture: $ARCH"

if [ "$ARCH" = "aarch64" ]; then
    BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.2.1-py3"
    if [ -x "$(command -v tegrastats)" ]; then
        TEGRA_PATH=$(command -v tegrastats)
        DOCKER_GPU_ARGS="--runtime nvidia -v $TEGRA_PATH:/usr/bin/tegrastats"
    else
        DOCKER_GPU_ARGS="--runtime nvidia"
    fi
    echo "Using Jetson Base Image: $BASE_IMAGE"
else
    BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
    DOCKER_GPU_ARGS="--gpus all"
    echo "Using x86_64 Base Image: $BASE_IMAGE"
fi

echo "Building Docker image for Phase 1 Python..."
docker build --build-arg BASE_IMAGE=$BASE_IMAGE --build-arg PYTHON_SRC_DIR=phase1_dataset/python -t yolo-phase1-py ../../

echo "Creating results directory..."
mkdir -p ../../results/phase1_dataset/python_eval

SIZES=(640 1280 1920)

for size in "${SIZES[@]}"; do
    if [ "$ENGINE" = "pytorch" ] && [ "$PRECISION" = "int8" ]; then
        echo "Skipping PyTorch with INT8"
        continue
    fi
    
    echo "================================================="
    echo "Running Phase 1 Python Eval: Engine=$ENGINE | Precision=$PRECISION | Size=$size"
    echo "================================================="
    
    docker run --rm $DOCKER_GPU_ARGS --shm-size=8g \
        -v "$(pwd)/../../results/phase1_dataset/python_eval:/workspace/results" \
        -v "$(pwd)/../../datasets:/workspace/datasets" \
        -v "$(pwd)/../../models:/workspace/models" \
        -e IMAGE_SIZE="$size" \
        -e PRECISION="$PRECISION" \
        -e ENGINE="$ENGINE" \
        -e OUTPUT_DIR="/workspace/results" \
        yolo-phase1-py python3 eval_map.py
done

echo "Session completed successfully! Results are in ../../results/phase1_dataset/python_eval"
