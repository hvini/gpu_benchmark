#pragma once

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <string>
#include <regex>
#include <cstdio>
#include <cstdlib>

#ifdef USE_NVML
#include <nvml.h>
#endif

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
    std::thread monitor_thread;
    
    std::vector<double> power_readings;
    std::vector<double> util_readings;
    std::vector<double> mem_readings;

    double calculate_mean(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    }

    double calculate_max(const std::vector<double>& v) {
        if (v.empty()) return 0.0;
        return *std::max_element(v.begin(), v.end());
    }

public:
    void start() {
        running = true;

#ifdef USE_NVML
        // --- DESKTOP (NVML) ---
        nvmlInit();
        monitor_thread = std::thread([this]() {
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(0, &device);
            while (running) {
                unsigned int power = 0;
                if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) power_readings.push_back(power / 1000.0);
                nvmlUtilization_t util;
                if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) util_readings.push_back(util.gpu);
                nvmlMemory_t mem;
                if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) mem_readings.push_back(mem.used / (1024.0 * 1024.0));
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
#else
        // --- JETSON (tegrastats) ---
        monitor_thread = std::thread([this]() {
            // Equivalent to Python's subprocess.Popen
            FILE* pipe = popen("tegrastats --interval 50", "r");
            if (!pipe) return;

            // Equivalent to Python's re.compile
            std::regex gr3d_pattern(R"(GR3D(?:_FREQ)?\s+(\d+)%)");
            std::regex power_pattern(R"((?:VDD_IN|POM_5V_IN|VDD_SYS_GPU|VDD_SYS_SOC)\s+(\d+)(?:mW)?/\d+(?:mW)?)");
            std::regex fallback_power_pattern(R"((?:VDD|POM|VIN|PWR)[A-Za-z0-9_]*\s+(\d+)(?:mW)?/\d+(?:mW)?)");
            std::regex ram_pattern(R"(RAM\s+(\d+)/\d+MB)");

            char buffer[512];
            // Read stdout pipe line-by-line in memory
            while (running && fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                std::string line(buffer);
                std::smatch match;

                // GPU Utilization
                if (std::regex_search(line, match, gr3d_pattern)) {
                    util_readings.push_back(std::stod(match[1].str()));
                }

                // Power
                if (std::regex_search(line, match, power_pattern)) {
                    power_readings.push_back(std::stod(match[1].str()) / 1000.0);
                } else {
                    double max_pwr = 0.0;
                    auto words_begin = std::sregex_iterator(line.begin(), line.end(), fallback_power_pattern);
                    auto words_end = std::sregex_iterator();
                    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
                        double p = std::stod((*i)[1].str()) / 1000.0;
                        if (p > max_pwr) max_pwr = p;
                    }
                    if (max_pwr > 0.0) power_readings.push_back(max_pwr);
                }

                // Memory
                if (std::regex_search(line, match, ram_pattern)) {
                    mem_readings.push_back(std::stod(match[1].str()));
                }
            }
            pclose(pipe);
        });
#endif
    }

    void stop() {
        running = false;
#ifdef USE_NVML
        nvmlShutdown();
#else
        // Kill tegrastats to force the stdout pipe to close, unblocking fgets
        std::system("pkill -9 tegrastats"); 
#endif
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }
    }

    GPUMetrics get_results() {
        GPUMetrics metrics;
        metrics.avg_power_W = calculate_mean(power_readings);
        metrics.max_power_W = calculate_max(power_readings);
        metrics.gpu_util_mean_pct = calculate_mean(util_readings);
        metrics.gpu_util_max_pct = calculate_max(util_readings);
        metrics.gpu_memory_mean_MB = calculate_mean(mem_readings);
        metrics.gpu_memory_max_MB = calculate_max(mem_readings);
        return metrics;
    }
};
