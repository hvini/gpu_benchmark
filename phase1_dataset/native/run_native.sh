#!/bin/bash

# ============================================================
# Phase 1 – Dataset Validation: Native C++ Benchmark Runner
# ============================================================
# Usage:  ./run_native.sh <engine> <precision>
# Example: ./run_native.sh tensorrt fp16
#
# Measures inference latency on real COCO val2017 images.
# mAP is NOT computed here; run phase1_dataset/python/run_session.sh
# for authoritative accuracy numbers.
#
# Output: results/phase1_dataset/native/<GPU>_..._native.csv
# ============================================================

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <engine> <precision>"
    echo "Example: $0 tensorrt fp16"
    echo "Valid Engines:    pytorch, onnx, tensorrt"
    echo "Valid Precisions: fp16, fp32, int8"
    exit 1
fi

ENGINE=$1
PRECISION=$2
ARCH=$(uname -m)

echo "=========================================="
echo "Phase 1 Native C++ Benchmark Suite"
echo "Architecture: $ARCH | Engine: $ENGINE | Precision: $PRECISION"
echo "=========================================="

# ── Architecture-specific settings ─────────────────────────────────────────
if [ "$ARCH" = "aarch64" ]; then
    BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.2.1-py3"
    CUDA_ARCHS="87"
    TORCH_ARCHS="8.7"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz"
    DOCKER_GPU_ARGS="--runtime=nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats"
else
    BASE_IMAGE="nvcr.io/nvidia/pytorch:25.01-py3"
    CUDA_ARCHS="75;80;86;89;90;120"
    TORCH_ARCHS="7.5;8.0;8.6;8.9;9.0;12.0"
    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz"
    DOCKER_GPU_ARGS="--gpus all"
fi

if docker image inspect yolo-phase1-native >/dev/null 2>&1; then
    echo "Docker image 'yolo-phase1-native' already exists. Skipping build."
else
    echo "Building Phase 1 Native Docker Image..."
    docker build -f ../../Dockerfile.native \
        --build-arg BASE_IMAGE="$BASE_IMAGE" \
        --build-arg CUDA_ARCHS="$CUDA_ARCHS" \
        --build-arg TORCH_ARCHS="$TORCH_ARCHS" \
        --build-arg ORT_URL="$ORT_URL" \
        --build-arg NATIVE_SRC_DIR="phase1_dataset/native" \
        -t yolo-phase1-native \
        ../..
fi

mkdir -p ../../results/phase1_dataset/native

DATASETS_ROOT="$(pwd)/../../datasets"
SIZES=(640 1280 1920)

for size in "${SIZES[@]}"; do

    if [ "$ENGINE" = "pytorch" ] && [ "$PRECISION" = "int8" ]; then
        echo "Skipping PyTorch with INT8 (not natively supported without QAT)"
        continue
    fi

    echo "-------------------------------------------------"
    echo "Resolution: ${size}x${size} | Engine: $ENGINE | Precision: $PRECISION"
    echo "-------------------------------------------------"

    # ── Model export ────────────────────────────────────────────────────────
    TARGET=""
    if [ "$ENGINE" = "pytorch" ];  then TARGET="yolo11s_${size}_${PRECISION}.torchscript"; fi
    if [ "$ENGINE" = "onnx" ];     then TARGET="yolo11s_${size}_${PRECISION}.onnx"; fi
    if [ "$ENGINE" = "tensorrt" ]; then TARGET="yolo11s_${size}_${PRECISION}.engine"; fi

    if [ -f "$TARGET" ]; then
        echo "✅ Found $TARGET. Skipping export."
    else
        echo "⚙️  Exporting $TARGET..."

        if [ "$ENGINE" = "pytorch" ]; then
            docker run --rm $DOCKER_GPU_ARGS \
                -v "$(pwd)/../..:/workspace" \
                -w /workspace/phase1_dataset/native \
                yolo-phase1-native \
                python3 -c "
import os; from ultralytics import YOLO
m = YOLO('../../models/yolo11s.pt')
p = str(m.export(format='torchscript', imgsz=$size, half=('$PRECISION'=='fp16'), device=0))
tgt = '$TARGET'
if os.path.basename(p) != tgt: os.replace(p, tgt)
"
        elif [ "$ENGINE" = "onnx" ]; then
            docker run --rm $DOCKER_GPU_ARGS \
                -v "$(pwd)/../..:/workspace" \
                -w /workspace/phase1_dataset/native \
                yolo-phase1-native \
                python3 -c "
import os; from ultralytics import YOLO
m = YOLO('../../models/yolo11s.pt')
p = str(m.export(format='onnx', imgsz=$size, half=('$PRECISION'=='fp16'),
                 int8=('$PRECISION'=='int8'), simplify=True, device=0))
tgt = '$TARGET'
if os.path.basename(p) != tgt: os.replace(p, tgt)
"
        elif [ "$ENGINE" = "tensorrt" ]; then
            ONNX_TEMP="yolo11s_${size}_${PRECISION}_temp.onnx"

            docker run --rm $DOCKER_GPU_ARGS \
                -v "$(pwd)/../..:/workspace" \
                -w /workspace/phase1_dataset/native \
                yolo-phase1-native \
                python3 -c "
import os; from ultralytics import YOLO
m = YOLO('../../models/yolo11s.pt')
p = str(m.export(format='onnx', imgsz=$size, half=('$PRECISION'=='fp16'), simplify=True, device=0))
tgt = '$ONNX_TEMP'
if os.path.basename(p) != tgt: os.replace(p, tgt)
"
            TRT_FLAGS=""
            if [ "$PRECISION" = "fp16" ]; then TRT_FLAGS="--fp16"; fi
            if [ "$PRECISION" = "int8" ]; then TRT_FLAGS="--int8"; fi

            echo "⚙️  Compiling ONNX → TensorRT engine..."
            docker run --rm $DOCKER_GPU_ARGS \
                -v "$(pwd)/../..:/workspace" \
                -w /workspace/phase1_dataset/native \
                yolo-phase1-native \
                /usr/src/tensorrt/bin/trtexec \
                    --onnx="$ONNX_TEMP" \
                    --saveEngine="$TARGET" \
                    --timingCacheFile="trt_global.cache" \
                    $TRT_FLAGS

            rm -f "$ONNX_TEMP"
        fi
    fi

    # ── Run C++ latency benchmark over COCO val2017 ──────────────────────────
    echo "================================================="
    echo "Running Phase 1 Native: $ENGINE | $PRECISION | ${size}x${size}"
    echo "================================================="

    docker run --rm $DOCKER_GPU_ARGS \
        --shm-size=8g \
        -v "$(pwd)/../../results/phase1_dataset/native:/workspace/results" \
        -v "$(pwd)/$TARGET:/workspace/phase1_dataset/native/$TARGET:ro" \
        -v "$DATASETS_ROOT:/workspace/datasets" \
        -w /workspace/phase1_dataset/native \
        -e ENGINE="$ENGINE" \
        -e PRECISION="$PRECISION" \
        -e IMAGE_SIZE="$size" \
        -e WARMUP=100 \
        -e MODEL="yolo11s.pt" \
        -e OUTPUT_DIR="/workspace/results" \
        -e COCO_IMAGES_DIR="/workspace/datasets/coco/images/val2017" \
        yolo-phase1-native

done

echo ""
echo "Phase 1 Native Benchmark completed!"
echo "Results → ../../results/phase1_dataset/native/"
echo ""
echo "For mAP scores run: cd ../python && ./run_session.sh $ENGINE $PRECISION"