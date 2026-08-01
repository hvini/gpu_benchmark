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

### Step 2: Run a Benchmark Session
Execute the session script by providing the `engine` and `precision` as arguments. The script automatically identifies your architecture (ARM for Jetson vs x86_64) and builds the correct Docker container.

**Valid Engines:** `pytorch`, `onnx`, `tensorrt`
**Valid Precisions:** `fp16`, `fp32`, `int8` *(Note: PyTorch with INT8 is unsupported)*

**Examples:**
```bash
# Run a TensorRT FP16 session
./run_session.sh tensorrt fp16

# Run an ONNX INT8 session
./run_session.sh onnx int8

# Run a PyTorch FP32 session
./run_session.sh pytorch fp32
```

### Step 3: Analyze the Results
Once the script completes, all performance metrics will be saved as CSV files in the newly created `results/` directory.

- **Location:** `./results/*.csv`
- Each CSV contains crucial academic metrics: FPS (Mean/Std), Latency (Mean, Std, p95, p99), GPU power usage, and Memory utilization.
