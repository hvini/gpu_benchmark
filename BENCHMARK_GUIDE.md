# YOLO Inference Academic Benchmark Guide

This document outlines the rigorous, academically valid benchmarking protocol for testing YOLO inference in real-time scenarios (e.g., autonomous driving systems). It details the required system preparation and the execution sequence.

## 1. Pre-Benchmark Checklist (Crucial)

Before running the benchmark suite on **any** target machine, you must prepare the system to ensure results are reproducible and unaffected by dynamic scaling or thermal throttling. Complete these steps in order:

### A. Isolate the Environment
- Ensure no desktop environment (X11/Wayland) is rendering complex UI on the target GPU. 
- Run the benchmark via SSH or drop to a TTY (`sudo systemctl isolate multi-user.target`).
- Ensure no other Docker containers, inference jobs, or heavy background processes are running.

### B. Optimize the Host CPU
The CPU feeds data to the GPU. If the CPU governor is in "powersave", latency will spike.
- Set the CPU governor to `performance`:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```
  *(Alternatively: `sudo cpupower frequency-set -g performance`)*

### C. Lock GPU Clocks & Power Limits (Desktop/Server GPUs: Ada, Blackwell, etc.)
Modern GPUs dynamically adjust their clock speeds. For an academic benchmark, lock the clock to eliminate variance.
- Find your GPU's max boost clock by running:
  ```bash
  nvidia-smi -q -d CLOCK
  ```
  *Look under the `Max Clocks` section for the `Graphics` value (e.g., `2550 MHz`).*
- Before applying locks, enable persistence mode so your settings aren't reset between runs:
  ```bash
  sudo nvidia-smi -pm 1
  ```
- Lock the clock explicitly to that exact frequency (e.g., if your max graphics clock is 2550 MHz):
  ```bash
  sudo nvidia-smi -lgc 2550
  ```
  *(To reset later: `sudo nvidia-smi -rgc`)*
- Maximize the power limit to prevent throttling. Run `nvidia-smi -q -d POWER` to find the `Max Power Limit` (e.g., `450.00 W`), then apply it:
  ```bash
  sudo nvidia-smi -pl 450
  ```

### D. Jetson Orin Specifics (Edge / ARM)
Jetson devices have specific power profiles.
- Enable the maximum performance profile (MAXN) and lock the clocks:
  ```bash
  sudo nvpmodel -m 0
  sudo jetson_clocks
  ```

### E. Monitor Thermals
- Keep an eye on thermals using `watch -n 1 nvidia-smi`. If the GPU hits its thermal limit (usually 85-90°C), it will aggressively downclock, ruining the data. Ensure adequate cooling before proceeding.

---

## 2. Execution Sequence

Once the hardware environment is locked and optimized, you can run the benchmark session. The script has been split so that you run **1 engine** and **1 precision** per session. The script will automatically loop across the different resolutions (640, 1280, 1920) for that configuration.

### Step 1: Prepare the script
Ensure the automated benchmark script has executable permissions (already done, but good to know):
```bash
chmod +x run_session.sh
```

### Step 2: Run the Official Academic Test Suite
Execute the session script by providing the `engine` and `precision` as arguments. The script automatically identifies your architecture (ARM for Jetson vs x86_64) and builds the correct Docker container.

For a fair and comprehensive academic comparison, you should run the following sessions in this exact sequence. *Note: PyTorch does not natively support INT8 without special Quantization Aware Training (QAT) in this pipeline, so it is excluded from INT8. TensorRT's INT8 uses Post-Training Quantization (PTQ), which is perfectly fair to compare against FP16/FP32 as it represents the peak capability and intended use-case of modern edge/server hardware.*

**1. The Baseline (PyTorch)**
Run the native PyTorch engine to establish your baseline latency and throughput.
```bash
./run_session.sh pytorch fp32
./run_session.sh pytorch fp16
```

**2. The Intermediate (ONNX)**
ONNX is widely used in C++ stacks and represents standard graph optimizations.
```bash
./run_session.sh onnx fp16
```

**3. Maximum Performance (TensorRT)**
TensorRT is NVIDIA's highly optimized engine and represents the absolute peak performance of the hardware.
```bash
./run_session.sh tensorrt fp16
./run_session.sh tensorrt int8
```

### Step 3: What to Collect (Data Analysis)
Once all the above sessions complete, your testing is done. You do not need to parse console logs. All performance metrics are automatically extracted and saved as CSV files in the newly created `results/` directory.

- **What to collect:** Copy the entire `./results/` folder. It will contain a CSV file for every Size + Precision + Engine combination.
- **Data included:** Each CSV contains crucial academic metrics:
  - `fps_mean`, `fps_std` (Throughput)
  - `latency_mean_ms`, `latency_std_ms`, `p95_latency_ms`, `p99_latency_ms` (Real-time determinism)
  - `avg_power_W`, `gpu_util_mean_pct`, `fps_per_watt` (Efficiency metrics)
