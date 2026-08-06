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

echo "=========================================="
echo "Starting Native C++ Benchmark Suite"
echo "Architecture: $ARCH | Engine: $ENGINE | Precision: $PRECISION"
echo "=========================================="

if [ "$ARCH" = "aarch64" ]; then
    BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r35.2.1-py3"
else
    # Official NVIDIA TensorRT image (contains trtexec and native TRT headers)
    BASE_IMAGE="nvcr.io/nvidia/tensorrt:24.08-py3"
fi

echo "Building Native Docker Image..."
docker build -f Dockerfile.native --build-arg BASE_IMAGE=$BASE_IMAGE -t yolo-native-benchmark .

mkdir -p results

SIZES=(640 1280 1920)

for size in "${SIZES[@]}"; do
    
    if [ "$ENGINE" = "pytorch" ] && [ "$PRECISION" = "int8" ]; then
        echo "Skipping PyTorch with INT8 (not natively supported without QAT)"
        continue
    fi
    
    echo "-------------------------------------------------"
    echo "Checking model formats for Resolution: ${size}x${size}"
    echo "-------------------------------------------------"

    TARGET=""
    if [ "$ENGINE" = "pytorch" ]; then TARGET="yolo11s_${size}_${PRECISION}.torchscript"; fi
    if [ "$ENGINE" = "onnx" ]; then TARGET="yolo11s_${size}_${PRECISION}.onnx"; fi
    if [ "$ENGINE" = "tensorrt" ]; then TARGET="yolo11s_${size}_${PRECISION}.engine"; fi

    if [ -f "$TARGET" ]; then
        echo "✅ Found $TARGET. Skipping export."
    else
        echo "⚙️ Exporting $TARGET (This will take a moment)..."
        
        if [ "$ENGINE" = "pytorch" ]; then
            docker run --rm --gpus all -v "$(pwd):/data" -w /data yolo-native-benchmark \
                python3 -c "import os; from ultralytics import YOLO; m=YOLO('yolo11s.pt'); p=str(m.export(format='torchscript', imgsz=$size, half=('$PRECISION'=='fp16'), device=0)); os.replace(p, '$TARGET') if os.path.basename(p) != '$TARGET' else None"
        
        elif [ "$ENGINE" = "onnx" ]; then
            docker run --rm --gpus all -v "$(pwd):/data" -w /data yolo-native-benchmark \
                python3 -c "import os; from ultralytics import YOLO; m=YOLO('yolo11s.pt'); p=str(m.export(format='onnx', imgsz=$size, half=('$PRECISION'=='fp16'), int8=('$PRECISION'=='int8'), simplify=True, device=0)); os.replace(p, '$TARGET') if os.path.basename(p) != '$TARGET' else None"
        
        elif [ "$ENGINE" = "tensorrt" ]; then
            # 1. Export to ONNX via Python
            ONNX_TEMP="yolo11s_${size}_${PRECISION}.onnx"
            docker run --rm --gpus all -v "$(pwd):/data" -w /data yolo-native-benchmark \
                python3 -c "import os; from ultralytics import YOLO; m=YOLO('yolo11s.pt'); p=str(m.export(format='onnx', imgsz=$size, half=('$PRECISION'=='fp16'), simplify=True, device=0)); os.replace(p, '$ONNX_TEMP') if os.path.basename(p) != '$ONNX_TEMP' else None"
            
            # 2. Compile ONNX to TensorRT Engine natively using NVIDIA's trtexec
            TRT_FLAGS=""
            if [ "$PRECISION" = "fp16" ]; then TRT_FLAGS="--fp16"; fi
            if [ "$PRECISION" = "int8" ]; then TRT_FLAGS="--int8"; fi
            
            echo "⚙️ Compiling ONNX to TensorRT Engine natively using trtexec..."
            docker run --rm --gpus all -v "$(pwd):/data" -w /data yolo-native-benchmark \
                trtexec --onnx=$ONNX_TEMP --saveEngine=$TARGET $TRT_FLAGS
        fi
    fi

    echo "================================================="
    echo "Running C++ Native Benchmark: Engine=$ENGINE | Precision=$PRECISION | Size=${size}x${size}"
    echo "================================================="
    
    docker run --rm --gpus all \
        -v "$(pwd)/results:/workspace/results" \
        -v "$(pwd):/data" \
        -w /data \
        -e IMAGE_SIZE="$size" \
        -e PRECISION="$PRECISION" \
        -e ENGINE="$ENGINE" \
        -e ITERATIONS=1000 \
        -e WARMUP=100 \
        -e RUNS=3 \
        yolo-native-benchmark \
        /workspace/build/benchmark_native
        
done

echo "Native Benchmark session completed! Results are in ./results"