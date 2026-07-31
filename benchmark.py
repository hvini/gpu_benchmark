import os
import time
import threading
import platform

import torch
import numpy as np
import pandas as pd
import pynvml

from ultralytics import YOLO
from tqdm import tqdm


# ==========================================================
# Configuration (change using docker -e)
# ==========================================================

MODEL = os.getenv("MODEL", "yolo11s.pt")

IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

PRECISION = os.getenv("PRECISION", "fp16").lower()

WARMUP = int(os.getenv("WARMUP", "100"))

ITERATIONS = int(os.getenv("ITERATIONS", "1000"))

RUNS = int(os.getenv("RUNS", "5"))


# ==========================================================
# Power monitoring
# ==========================================================

class PowerMonitor:

    def __init__(self, interval=0.1):

        self.interval = interval
        self.running = False
        self.samples = []

        pynvml.nvmlInit()

        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)


    def start(self):

        self.running = True

        self.thread = threading.Thread(
            target=self._sample
        )

        self.thread.start()


    def _sample(self):

        while self.running:

            power = (
                pynvml.nvmlDeviceGetPowerUsage(self.handle)
                / 1000.0
            )

            self.samples.append(power)

            time.sleep(self.interval)


    def stop(self):

        self.running = False

        self.thread.join()


    def get_results(self):

        if len(self.samples) == 0:

            return {
                "avg_power_W": 0,
                "max_power_W": 0
            }


        return {

            "avg_power_W":
                np.mean(self.samples),

            "max_power_W":
                np.max(self.samples)
        }


# ==========================================================
# GPU information
# ==========================================================

def gpu_memory():

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return mem.used / 1024**2



# ==========================================================
# Setup
# ==========================================================

print("========================")
print("Benchmark configuration")
print("========================")

print("Model:", MODEL)
print("Resolution:", IMAGE_SIZE)
print("Precision:", PRECISION)
print("Runs:", RUNS)
print("Iterations:", ITERATIONS)


print()

print("CUDA available:",
      torch.cuda.is_available())

print("GPU:",
      torch.cuda.get_device_name(0))


torch.backends.cudnn.benchmark = True


model = YOLO(MODEL)

model.to("cuda")


if PRECISION == "fp16":

    print("Using FP16")

    model.model.half()


elif PRECISION == "fp32":

    print("Using FP32")


else:

    raise ValueError(
        "PRECISION must be fp16 or fp32"
    )


dummy = np.random.randint(
    0,
    255,
    (
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),
    dtype=np.uint8
)



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


power_monitor = PowerMonitor()

power_monitor.start()


for run in range(RUNS):

    print(f"\nRun {run+1}/{RUNS}")


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

        "run": run + 1,

        "fps":
            1000 / np.mean(times),

        "latency_mean_ms":
            np.mean(times),

        "latency_std_ms":
            np.std(times),

        "p50_latency_ms":
            np.percentile(times, 50),

        "p95_latency_ms":
            np.percentile(times, 95),

        "p99_latency_ms":
            np.percentile(times, 99)

    })



power_monitor.stop()


runs_df = pd.DataFrame(all_results)



# ==========================================================
# Final result
# ==========================================================

power = power_monitor.get_results()


result = {

    "gpu":
        torch.cuda.get_device_name(0),

    "model":
        MODEL,

    "resolution":
        IMAGE_SIZE,

    "precision":
        PRECISION,

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


    "gpu_memory_MB":
        gpu_memory(),


    "python":
        platform.python_version(),

    "torch":
        torch.__version__,

    "cuda":
        torch.version.cuda
}


result.update(power)


# FPS per watt

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


filename = (
    f"{result['gpu'].replace(' ', '_')}_"
    f"{MODEL.replace('.pt','')}_"
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
