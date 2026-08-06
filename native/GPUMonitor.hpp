#ifndef GPU_MONITOR_HPP
#define GPU_MONITOR_HPP

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <nvml.h>

struct GPUSample {
    double power_w;
    double gpu_util_pct;
    double memory_mb;
};

struct GPUMetrics {
    double avg_power_W = 0.0;
    double max_power_W = 0.0;
    double gpu_util_mean_pct = 0.0;
    double gpu_util_max_pct = 0.0;
    double gpu_memory_mean_MB = 0.0;
    double gpu_memory_max_MB = 0.0;
};

class GPUMonitor {
private:
    std::atomic<bool> running{false};
    std::thread worker_thread;
    std::vector<GPUSample> samples;
    nvmlDevice_t device_handle;
    int sample_interval_ms;

    void sample_loop() {
        while (running) {
            GPUSample sample{0.0, 0.0, 0.0};
            
            // Power in mW -> Convert to W
            unsigned int power_mw = 0;
            if (nvmlDeviceGetPowerUsage(device_handle, &power_mw) == NVML_SUCCESS) {
                sample.power_w = power_mw / 1000.0;
            }

            // Utilization rates
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(device_handle, &util) == NVML_SUCCESS) {
                sample.gpu_util_pct = static_cast<double>(util.gpu);
            }

            // Memory info
            nvmlMemory_t mem;
            if (nvmlDeviceGetMemoryInfo(device_handle, &mem) == NVML_SUCCESS) {
                sample.memory_mb = static_cast<double>(mem.used) / (1024.0 * 1024.0);
            }

            samples.push_back(sample);
            std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
        }
    }

public:
    GPUMonitor(int interval_ms = 100) : sample_interval_ms(interval_ms) {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(0, &device_handle);
    }

    ~GPUMonitor() {
        stop();
        nvmlShutdown();
    }

    void start() {
        samples.clear();
        running = true;
        worker_thread = std::thread(&GPUMonitor::sample_loop, this);
    }

    void stop() {
        if (running) {
            running = false;
            if (worker_thread.joinable()) {
                worker_thread.join();
            }
        }
    }

    GPUMetrics get_results() {
        if (samples.empty()) return GPUMetrics{};

        GPUMetrics res;
        double sum_power = 0.0, sum_util = 0.0, sum_mem = 0.0;
        res.max_power_W = 0.0;
        res.gpu_util_max_pct = 0.0;
        res.gpu_memory_max_MB = 0.0;

        for (const auto& s : samples) {
            sum_power += s.power_w;
            sum_util += s.gpu_util_pct;
            sum_mem += s.memory_mb;

            res.max_power_W = std::max(res.max_power_W, s.power_w);
            res.gpu_util_max_pct = std::max(res.gpu_util_max_pct, s.gpu_util_pct);
            res.gpu_memory_max_MB = std::max(res.gpu_memory_max_MB, s.memory_mb);
        }

        size_t n = samples.size();
        res.avg_power_W = sum_power / n;
        res.gpu_util_mean_pct = sum_util / n;
        res.gpu_memory_mean_MB = sum_mem / n;

        return res;
    }
};

#endif