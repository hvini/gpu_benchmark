import os
import sys
import time
import platform
import torch
import pandas as pd
from ultralytics import YOLO
from ultralytics import settings

# Dynamically add the project root to sys.path to import the shared module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from gpu_monitor import GPUMonitor

# ==========================================================
# Configuration
# ==========================================================
MODEL = os.getenv("MODEL", "yolo11s.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
PRECISION = os.getenv("PRECISION", "fp16").lower()
ENGINE = os.getenv("ENGINE", "pytorch").lower()
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1")) # Keep at 1 to mimic real-time edge latency

# Set output directory to the new Phase 1 folder structure
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/phase1_dataset/python_eval')))

datasets_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets'))
settings.update({'datasets_dir': datasets_root})

# ==========================================================
# Setup & Engine Export (Preserved from Phase 0)
# ==========================================================
print("========================================")
print("Phase 1: Dataset Validation (Python)")
print("========================================")
print("Model:", MODEL)
print("Resolution:", IMAGE_SIZE)
print("Precision:", PRECISION)
print("Engine:", ENGINE)
print("Batch Size:", BATCH_SIZE)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("========================================\n")

model = YOLO(os.path.join("../../models", MODEL) if not os.path.isabs(MODEL) else MODEL)

if ENGINE == "pytorch":
    pass
else:
    export_format = "engine" if ENGINE == "tensorrt" else "onnx"
    half = (PRECISION == "fp16")
    int8 = (PRECISION == "int8")
    data = "coco8.yaml" if int8 else None

    model_name = os.path.splitext(os.path.basename(MODEL))[0]
    target_name = f"../../models/{model_name}_{IMAGE_SIZE}_{PRECISION}.{export_format}"
    
    if os.path.exists(target_name):
        print(f"Loading existing {ENGINE} model: {target_name}")
        model = YOLO(target_name, task="detect")
    else:
        print(f"Exporting model to {ENGINE} (half={half}, int8={int8})...")
        export_path = model.export(
            format=export_format, 
            imgsz=IMAGE_SIZE, 
            half=half, 
            int8=int8, 
            data=data,
            dynamic=False,
            simplify=True if ENGINE=="onnx" else False,
            nms=True if ENGINE=="tensorrt" else False # Bake NMS into TRT for exact parity
        )
        if export_path != target_name:
            os.rename(export_path, target_name)
        model = YOLO(target_name, task="detect")

# ==========================================================
# Benchmark Execution
# ==========================================================
print("\nStarting COCO Dataset Evaluation...")
print("Note: If COCO val2017 is missing, Ultralytics will automatically download it (~19GB).")

gpu_monitor = GPUMonitor(interval=0.1)
gpu_monitor.start()

# Run the validation
metrics = model.val(
    data="coco.yaml", 
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=0,
    half=(PRECISION == "fp16"),
    rect=False,
    workers=4,
    verbose=True,
    plots=False
)

gpu_monitor.stop()

# ==========================================================
# Parse Results
# ==========================================================
gpu_metrics = gpu_monitor.get_results()
speeds = metrics.speed # Dictionary in ms per image: {'preprocess': x, 'inference': y, 'postprocess': z}

# Total end-to-end latency per image
total_latency_ms = speeds['preprocess'] + speeds['inference'] + speeds['postprocess']
fps_e2e = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0

result = {
    "gpu": torch.cuda.get_device_name(0),
    "model": os.path.basename(MODEL),
    "resolution": IMAGE_SIZE,
    "precision": PRECISION,
    "engine": ENGINE,
    "batch_size": BATCH_SIZE,
    
    # Accuracy Metrics
    "map50": metrics.box.map50,
    "map50_95": metrics.box.map,
    
    # Latency Breakdown (ms)
    "preprocess_ms": speeds['preprocess'],
    "inference_ms": speeds['inference'],
    "nms_ms": speeds['postprocess'],
    "total_latency_ms": total_latency_ms,
    "fps_e2e": fps_e2e,
    
    # Software Context
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda
}

result.update(gpu_metrics)

# Calculate efficiency
if result["avg_power_W"] > 0:
    result["fps_per_watt"] = result["fps_e2e"] / result["avg_power_W"]
else:
    result["fps_per_watt"] = 0

# ==========================================================
# Save Output
# ==========================================================
print("\n========================================")
print("FINAL RESULT SUMMARY")
print("========================================")
print(pd.DataFrame([result]).to_string(index=False))

os.makedirs(OUTPUT_DIR, exist_ok=True)

gpu_file_str = result['gpu'].replace(' ', '_')
model_str = os.path.splitext(os.path.basename(MODEL))[0]
filename = f"{gpu_file_str}_{model_str}_{ENGINE}_{PRECISION}_{IMAGE_SIZE}_python.csv"
output_path = os.path.join(OUTPUT_DIR, filename)

pd.DataFrame([result]).to_csv(output_path, index=False)

print(f"\nSaved Phase 1 Python results to: {output_path}")