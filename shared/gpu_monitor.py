import threading
import pynvml
import numpy as np

class GPUMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.samples = []
        self.nvml_available = False
        self.tegrastats_proc = None

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_available = True
        except Exception:
            pass

    def start(self):
        self.running = True
        
        if not self.nvml_available:
            import subprocess
            try:
                self.tegrastats_proc = subprocess.Popen(
                    ['stdbuf', '-oL', 'tegrastats', '--interval', str(int(self.interval * 1000))],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            except Exception as e:
                try:
                    self.tegrastats_proc = subprocess.Popen(
                        ['tegrastats', '--interval', str(int(self.interval * 1000))],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1
                    )
                except Exception as e2:
                    print(f"Tegrastats error (Jetson monitor failed): {e2}")

        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        if self.nvml_available:
            while self.running:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                self.samples.append({
                    "power": power,
                    "gpu_util": util.gpu,
                    "memory_MB": mem.used / (1024**2)
                })
                import time
                time.sleep(self.interval)
        else:
            import time
            if not self.tegrastats_proc:
                while self.running:
                    self.samples.append({"power": 0.0, "gpu_util": 0.0, "memory_MB": 0.0})
                    time.sleep(self.interval)
                return
            
            import re
            gr3d_pattern = re.compile(r'GR3D(?:_FREQ)?\s+(\d+)%')
            power_pattern = re.compile(r'(?:VDD_IN|POM_5V_IN|VDD_SYS_GPU|VDD_SYS_SOC)\s+(\d+)(?:mW)?/\d+(?:mW)?')
            fallback_power_pattern = re.compile(r'(?:VDD|POM|VIN|PWR)[A-Za-z0-9_]*\s+(\d+)(?:mW)?/\d+(?:mW)?')
            ram_pattern = re.compile(r'RAM\s+(\d+)/\d+MB')
            
            first_line = True
            while self.running:
                line = self.tegrastats_proc.stdout.readline()
                if not line:
                    break
                    
                gpu_util = 0.0
                power_w = 0.0
                memory_mb = 0.0
                
                m_gr3d = gr3d_pattern.search(line)
                if m_gr3d: gpu_util = float(m_gr3d.group(1))
                
                m_power = power_pattern.search(line)
                if m_power: 
                    power_w = float(m_power.group(1)) / 1000.0
                else:
                    power_matches = fallback_power_pattern.findall(line)
                    if power_matches:
                        power_w = max([float(p) for p in power_matches]) / 1000.0
                
                m_ram = ram_pattern.search(line)
                if m_ram: memory_mb = float(m_ram.group(1))
                
                self.samples.append({
                    "power": power_w,
                    "gpu_util": gpu_util,
                    "memory_MB": memory_mb
                })

    def stop(self):
        self.running = False
        if not self.nvml_available and self.tegrastats_proc:
            self.tegrastats_proc.terminate()
            self.tegrastats_proc.wait()
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

        power = [x["power"] for x in self.samples]
        util = [x["gpu_util"] for x in self.samples]
        memory = [x["memory_MB"] for x in self.samples]

        return {
            "avg_power_W": np.mean(power),
            "max_power_W": np.max(power),
            "gpu_util_mean_pct": np.mean(util),
            "gpu_util_max_pct": np.max(util),
            "gpu_memory_mean_MB": np.mean(memory),
            "gpu_memory_max_MB": np.max(memory)
        }
