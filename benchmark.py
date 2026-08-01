import os
import time
import threading
import platform
import urllib.request

import cv2
import torch
import numpy as np
import pandas as pd
import pynvml

from ultralytics import YOLO
from tqdm import tqdm


# ==========================================================
# Configuration
# ==========================================================

MODEL = os.getenv("MODEL", "yolo11s.pt")

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

PRECISION = os.getenv("PRECISION", "fp16").lower()

WARMUP = int(os.getenv("WARMUP", "100"))

ITERATIONS = int(os.getenv("ITERATIONS", "1000"))

RUNS = int(os.getenv("RUNS", "5"))

ENGINE = os.getenv("ENGINE", "pytorch").lower()


# ==========================================================
# GPU Monitor (Power + Utilization)
# ==========================================================

class GPUMonitor:

    def __init__(self, interval=0.1):

        self.interval = interval

        self.running = False

        self.samples = []

        pynvml.nvmlInit()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)


    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._sample,
            daemon=True
        )

        self.thread.start()


    def _sample(self):

        while self.running:

            power = (
                pynvml.nvmlDeviceGetPowerUsage(
                    self.handle
                )
                /
                1000.0
            )


            util = (
                pynvml.nvmlDeviceGetUtilizationRates(
                    self.handle
                )
            )


            mem = (
                pynvml.nvmlDeviceGetMemoryInfo(
                    self.handle
                )
            )


            self.samples.append({

                "power":
                    power,

                "gpu_util":
                    util.gpu,

                "memory_MB":
                    mem.used / 1024**2
            })


            time.sleep(self.interval)



    def stop(self):

        self.running = False

        self.thread.join()



    def get_results(self):

        if len(self.samples) == 0:

            return {

                "avg_power_W": 0,
                "max_power_W": 0,

                "gpu_util_mean_pct": 0,
                "gpu_util_max_pct": 0,

                "gpu_memory_mean_MB": 0,
                "gpu_memory_max_MB": 0
            }


        power = [
            x["power"]
            for x in self.samples
        ]


        util = [
            x["gpu_util"]
            for x in self.samples
        ]


        memory = [
            x["memory_MB"]
            for x in self.samples
        ]


        return {

            "avg_power_W":
                np.mean(power),

            "max_power_W":
                np.max(power),


            "gpu_util_mean_pct":
                np.mean(util),

            "gpu_util_max_pct":
                np.max(util),


            "gpu_memory_mean_MB":
                np.mean(memory),

            "gpu_memory_max_MB":
                np.max(memory)
        }



# ==========================================================
# Setup
# ==========================================================

print("========================")
print("Benchmark configuration")
print("========================")

print("Model:", MODEL)
print("Resolution:", IMAGE_SIZE)
print("Precision:", PRECISION)
print("Engine:", ENGINE)
print("Runs:", RUNS)
print("Iterations:", ITERATIONS)

print()

print(
    "CUDA available:",
    torch.cuda.is_available()
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


torch.backends.cudnn.benchmark = True


model = YOLO(MODEL)

if ENGINE == "pytorch":

    model.to("cuda")

    if PRECISION == "fp16":

        print("Using FP16")

        model.model.half()

    elif PRECISION == "fp32":

        print("Using FP32")

    else:

        raise ValueError(
            "PyTorch PRECISION must be fp16 or fp32"
        )

else:

    export_format = "engine" if ENGINE == "tensorrt" else "onnx"
    half = (PRECISION == "fp16")
    int8 = (PRECISION == "int8")
    data = "coco8.yaml" if int8 else None

    import os
    model_name = os.path.splitext(os.path.basename(MODEL))[0]
    target_name = f"{model_name}_{IMAGE_SIZE}_{PRECISION}.{export_format}"
    
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
            simplify=True if ENGINE=="onnx" else False
        )
        if export_path != target_name:
            os.rename(export_path, target_name)
        model = YOLO(target_name, task="detect")



IMAGE_PATH = "data/road_image.jpg"
IMAGE_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"\nDownloading real road image from {IMAGE_URL}...")
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
    urllib.request.urlretrieve(IMAGE_URL, IMAGE_PATH)

img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Failed to load image at {IMAGE_PATH}")

dummy = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))



# ==========================================================
# Warmup
# ==========================================================

print("\nWarmup...")


with torch.inference_mode():

    for _ in range(WARMUP):

        model.predict(
            dummy,
            imgsz=IMAGE_SIZE,
            device=0,
            verbose=False
        )


torch.cuda.synchronize()



# ==========================================================
# Benchmark
# ==========================================================

all_results = []


gpu_monitor = GPUMonitor()

gpu_monitor.start()



for run in range(RUNS):

    print(
        f"\nRun {run+1}/{RUNS}"
    )


    times = []


    with torch.inference_mode():

        for _ in tqdm(range(ITERATIONS)):


            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )


            start.record()


            model.predict(
                dummy,
                imgsz=IMAGE_SIZE,
                device=0,
                verbose=False
            )


            end.record()


            torch.cuda.synchronize()


            times.append(
                start.elapsed_time(end)
            )



    times = np.array(times)



    all_results.append({

        "run":
            run + 1,

        "fps":
            1000 / np.mean(times),

        "latency_mean_ms":
            np.mean(times),

        "latency_std_ms":
            np.std(times),

        "p95_latency_ms":
            np.percentile(times,95),

        "p99_latency_ms":
            np.percentile(times,99)

    })



gpu_monitor.stop()



runs_df = pd.DataFrame(
    all_results
)



# ==========================================================
# Final Result
# ==========================================================

gpu_metrics = gpu_monitor.get_results()



result = {


    "gpu":
        torch.cuda.get_device_name(0),


    "model":
        MODEL,


    "resolution":
        IMAGE_SIZE,


    "precision":
        PRECISION,


    "engine":
        ENGINE,


    "runs":
        RUNS,


    "iterations":
        ITERATIONS,



    "fps_mean":
        runs_df["fps"].mean(),


    "fps_std":
        runs_df["fps"].std(),



    "latency_mean_ms":
        runs_df["latency_mean_ms"].mean(),


    "latency_std_ms":
        runs_df["latency_std_ms"].mean(),



    "p95_latency_ms":
        runs_df["p95_latency_ms"].mean(),


    "p99_latency_ms":
        runs_df["p99_latency_ms"].mean(),



    "python":
        platform.python_version(),


    "torch":
        torch.__version__,


    "cuda":
        torch.version.cuda

}



result.update(gpu_metrics)



if result["avg_power_W"] > 0:

    result["fps_per_watt"] = (
        result["fps_mean"]
        /
        result["avg_power_W"]
    )

else:

    result["fps_per_watt"] = 0



print("\n================")
print("FINAL RESULT")
print("================")


print(
    pd.DataFrame([result])
    .to_string(index=False)
)



output_dir = os.getenv(
    "OUTPUT_DIR",
    "/workspace/results"
)


os.makedirs(
    output_dir,
    exist_ok=True
)



filename = (

    f"{result['gpu'].replace(' ','_')}_"

    f"{MODEL.replace('.pt','')}_"

    f"{ENGINE}_"

    f"{PRECISION}_"

    f"{IMAGE_SIZE}.csv"

)



output_path = os.path.join(
    output_dir,
    filename
)



pd.DataFrame([result]).to_csv(
    output_path,
    index=False
)


print("\nSaved:")
print(output_path)
