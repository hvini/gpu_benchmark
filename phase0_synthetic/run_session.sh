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
    # Standard L4T ML image for Jetson (comes with PyTorch & TensorRT pre-installed)
    BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.2.1-py3"
    if [ -x "$(command -v tegrastats)" ]; then
        TEGRA_PATH=$(command -v tegrastats)
        DOCKER_GPU_ARGS="--runtime nvidia -v $TEGRA_PATH:/usr/bin/tegrastats"
    else
        DOCKER_GPU_ARGS="--runtime nvidia"
    fi
    echo "Using Jetson Base Image: $BASE_IMAGE"
else
    # Default x86_64 image for desktop/server GPUs
    BASE_IMAGE="nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04"
    DOCKER_GPU_ARGS="--gpus all"
    echo "Using x86_64 Base Image: $BASE_IMAGE"
fi

echo "Building Docker image..."
docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t yolo-benchmark ..

echo "Creating results directory..."
mkdir -p ../results/phase0_synthetic

# We loop through sizes for this specific session
SIZES=(640 1280 1920)

for size in "${SIZES[@]}"; do
    
    # Skip unsupported combinations
    if [ "$ENGINE" = "pytorch" ] && [ "$PRECISION" = "int8" ]; then
        echo "Skipping PyTorch with INT8 (not natively supported without QAT)"
        continue
    fi
    
    echo "================================================="
    echo "Running Benchmark: Engine=$ENGINE | Precision=$PRECISION | Size=$size"
    echo "================================================="
    
    docker run --rm $DOCKER_GPU_ARGS \
        -v "$(pwd)/../results/phase0_synthetic:/workspace/results" \
        -e IMAGE_SIZE="$size" \
        -e PRECISION="$PRECISION" \
        -e ENGINE="$ENGINE" \
        -e ITERATIONS=1000 \
        -e WARMUP=100 \
        -e RUNS=3 \
        yolo-benchmark
        
done

echo "Session completed successfully! Results are in ./results"

