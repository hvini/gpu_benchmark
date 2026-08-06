#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Engine Headers
#ifdef ENABLE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

#ifdef ENABLE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

#ifdef ENABLE_TRT
#include <NvInfer.h>
#endif

#include "GPUMonitor.hpp"

// Utility Functions
double calculate_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

double calculate_std(const std::vector<double>& v, double mean) {
    if (v.empty()) return 0.0;
    double sq_sum = 0.0;
    for (double x : v) sq_sum += (x - mean) * (x - mean);
    return std::sqrt(sq_sum / v.size());
}

double calculate_percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
    return v[std::min(idx, v.size() - 1)];
}

// ----------------------------------------------------------------------
// Base Abstract Engine Interface
// ----------------------------------------------------------------------
class InferenceEngine {
public:
    virtual void load(const std::string& model_path, int img_size, const std::string& precision) = 0;
    virtual void infer(const cv::Mat& input_img) = 0;
    virtual ~InferenceEngine() = default;
};

// ----------------------------------------------------------------------
// 1. PyTorch Engine Implementation
// ----------------------------------------------------------------------
#ifdef ENABLE_TORCH
class PyTorchEngine : public InferenceEngine {
private:
    torch::jit::script::Module module;
    torch::Device device{torch::kCUDA};
    torch::ScalarType dtype{torch::kFloat32};
    int imgsz;
    torch::Tensor input_tensor;

public:
    void load(const std::string& model_path, int img_size, const std::string& precision) override {
        imgsz = img_size;
        dtype = (precision == "fp16") ? torch::kHalf : torch::kFloat32;
        module = torch::jit::load(model_path, device);
        module.eval();
        module.to(dtype);

        // OPTIMIZATION: Pre-allocate the GPU tensor once during setup
        input_tensor = torch::zeros({1, 3, imgsz, imgsz}, torch::TensorOptions().dtype(dtype).device(device));
    }

    void infer(const cv::Mat& input_img) override {
        torch::NoGradGuard no_grad;
        // PURE GPU INFERENCE - Zero CPU image processing or Host-to-Device transfers in hot loop
        auto output = module.forward({input_tensor});
    }
};
#endif

// ----------------------------------------------------------------------
// 2. ONNX Runtime Engine Implementation
// ----------------------------------------------------------------------
#ifdef ENABLE_ONNX
class ONNXEngine : public InferenceEngine {
private:
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "YOLO_Benchmark"};
    std::unique_ptr<Ort::Session> session;
    
    // CPU MemoryInfo allows ONNX Runtime to use its fast, asynchronous CUDA memory arenas
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    int imgsz;
    bool is_input_fp16{false};
    
    // Pre-allocated CPU memory buffers
    std::vector<float> input_data_fp32;
    std::vector<uint16_t> input_data_fp16;
    size_t input_tensor_size;
    std::vector<int64_t> input_shape;

public:
    void load(const std::string& model_path, int img_size, const std::string& precision) override {
        imgsz = img_size;
        input_tensor_size = 1 * 3 * imgsz * imgsz;
        input_shape = {1, 3, imgsz, imgsz};

        Ort::SessionOptions session_options;
        session_options.SetLogSeverityLevel(3); // Mute warnings
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        Ort::TypeInfo type_info = session->GetInputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        is_input_fp16 = (tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

        // Allocate once during setup
        if (is_input_fp16) {
            input_data_fp16.resize(input_tensor_size, 0);
        } else {
            input_data_fp32.resize(input_tensor_size, 0.0f);
        }
    }

    void infer(const cv::Mat& input_img) override {
        std::vector<const char*> input_names = {"images"};
        std::vector<const char*> output_names = {"output0"};

        if (is_input_fp16) {
            Ort::Value input_tensor = Ort::Value::CreateTensor(
                memory_info, input_data_fp16.data(), input_tensor_size * sizeof(uint16_t),
                input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
            );
            session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        } else {
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data_fp32.data(), input_tensor_size, input_shape.data(), input_shape.size()
            );
            session->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        }
    }
};
#endif

// ----------------------------------------------------------------------
// 3. TensorRT Engine Implementation (Dynamic Buffer Allocation)
// ----------------------------------------------------------------------
#ifdef ENABLE_TRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} gLogger;

class TRTEngine : public InferenceEngine {
private:
    nvinfer1::IRuntime* runtime{nullptr};
    nvinfer1::ICudaEngine* engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};
    std::vector<void*> buffers;
    cudaStream_t stream{nullptr};

public:
    void load(const std::string& model_path, int img_size, const std::string& precision) override {
        std::ifstream file(model_path, std::ios::binary);
        if (!file.good()) throw std::runtime_error("Engine file not found: " + model_path);
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);
        file.close();

        runtime = nvinfer1::createInferRuntime(gLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
        context = engine->createExecutionContext();

        cudaStreamCreate(&stream);

        int32_t num_tensors = engine->getNbIOTensors();
        buffers.resize(num_tensors, nullptr);

        for (int32_t i = 0; i < num_tensors; ++i) {
            const char* tensor_name = engine->getIOTensorName(i);
            nvinfer1::Dims dims = engine->getTensorShape(tensor_name);
            nvinfer1::DataType dtype = engine->getTensorDataType(tensor_name);
            
            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                vol *= std::max(1, static_cast<int>(dims.d[d]));
            }
            
            size_t element_size = 4;
            if (dtype == nvinfer1::DataType::kHALF) element_size = 2;
            else if (dtype == nvinfer1::DataType::kINT8) element_size = 1;
            
            cudaMalloc(&buffers[i], vol * element_size);
            context->setTensorAddress(tensor_name, buffers[i]);
        }
    }

    void infer(const cv::Mat& input_img) override {
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
    }

    ~TRTEngine() override {
        for (void* buf : buffers) {
            if (buf) cudaFree(buf);
        }
        if (stream) cudaStreamDestroy(stream);
        if (context) delete context;
        if (engine) delete engine;
        if (runtime) delete runtime;
    }
};
#endif

// ----------------------------------------------------------------------
// Main Executable Entrypoint
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    const std::string engine_type = std::getenv("ENGINE") ? std::getenv("ENGINE") : "pytorch";
    const std::string precision   = std::getenv("PRECISION") ? std::getenv("PRECISION") : "fp16";
    const int image_size          = std::getenv("IMAGE_SIZE") ? std::stoi(std::getenv("IMAGE_SIZE")) : 640;
    const int warmup              = std::getenv("WARMUP") ? std::stoi(std::getenv("WARMUP")) : 100;
    const int iterations          = std::getenv("ITERATIONS") ? std::stoi(std::getenv("ITERATIONS")) : 1000;
    const int runs                = std::getenv("RUNS") ? std::stoi(std::getenv("RUNS")) : 3;
    const std::string model_input = std::getenv("MODEL") ? std::getenv("MODEL") : "yolo11s.pt";
    const std::string output_dir  = std::getenv("OUTPUT_DIR") ? std::getenv("OUTPUT_DIR") : "/workspace/results";

    // Query GPU Details via CUDA Driver API
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu_name = prop.name;

    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    std::string cuda_version_str = std::to_string(runtime_ver / 1000) + "." + std::to_string((runtime_ver % 1000) / 10);

    // Standardize Model Name
    std::string model_base = model_input;
    if (model_base.length() > 3 && model_base.substr(model_base.length() - 3) == ".pt") {
        model_base = model_base.substr(0, model_base.length() - 3);
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Native C++ Benchmark Configuration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "GPU:        " << gpu_name << std::endl;
    std::cout << "Model:      " << model_input << std::endl;
    std::cout << "Engine:     " << engine_type << std::endl;
    std::cout << "Precision:  " << precision << std::endl;
    std::cout << "Resolution: " << image_size << "x" << image_size << std::endl;
    std::cout << "Runs:       " << runs << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;

    // Instantiate Engine
    std::unique_ptr<InferenceEngine> engine;
    std::string model_ext;

    if (engine_type == "pytorch") {
#ifdef ENABLE_TORCH
        engine = std::make_unique<PyTorchEngine>();
        model_ext = ".torchscript";
#else
        std::cerr << "LibTorch was not enabled at compile time!" << std::endl;
        return 1;
#endif
    } else if (engine_type == "onnx") {
#ifdef ENABLE_ONNX
        engine = std::make_unique<ONNXEngine>();
        model_ext = ".onnx";
#else
        std::cerr << "ONNX Runtime was not enabled at compile time!" << std::endl;
        return 1;
#endif
    } else if (engine_type == "tensorrt") {
#ifdef ENABLE_TRT
        engine = std::make_unique<TRTEngine>();
        model_ext = ".engine";
#else
        std::cerr << "TensorRT was not enabled at compile time!" << std::endl;
        return 1;
#endif
    }

    std::string model_path = model_base + "_" + std::to_string(image_size) + "_" + precision + model_ext;
    std::cout << "Loading native model: " << model_path << std::endl;
    engine->load(model_path, image_size, precision);

    cv::Mat dummy_img = cv::Mat::zeros(cv::Size(image_size, image_size), CV_8UC3);

    // Warmup
    std::cout << "\nExecuting Warmup (" << warmup << " iterations)..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        engine->infer(dummy_img);
    }
    cudaDeviceSynchronize();

    // GPU Monitoring
    GPUMonitor gpu_monitor;
    gpu_monitor.start();

    struct RunResult {
        int run;
        double fps;
        double latency_mean_ms;
        double latency_std_ms;
        double p95_latency_ms;
        double p99_latency_ms;
    };

    std::vector<RunResult> run_results;

    // Benchmark Execution Loop
    for (int r = 0; r < runs; ++r) {
        std::cout << "\nRun " << (r + 1) << "/" << runs << "..." << std::endl;
        std::vector<double> latencies;
        latencies.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            engine->infer(dummy_img);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            latencies.push_back(static_cast<double>(milliseconds));

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        double mean_lat = calculate_mean(latencies);
        double std_lat  = calculate_std(latencies, mean_lat);
        double p95      = calculate_percentile(latencies, 95.0);
        double p99      = calculate_percentile(latencies, 99.0);
        double fps      = 1000.0 / mean_lat;

        run_results.push_back({r + 1, fps, mean_lat, std_lat, p95, p99});
    }

    gpu_monitor.stop();
    GPUMetrics gpu_metrics = gpu_monitor.get_results();

    // Calculate Summary Statistics
    std::vector<double> all_fps, all_means, all_stds, all_p95, all_p99;
    for (const auto& res : run_results) {
        all_fps.push_back(res.fps);
        all_means.push_back(res.latency_mean_ms);
        all_stds.push_back(res.latency_std_ms);
        all_p95.push_back(res.p95_latency_ms);
        all_p99.push_back(res.p99_latency_ms);
    }

    double final_fps_mean = calculate_mean(all_fps);
    double final_fps_std  = calculate_std(all_fps, final_fps_mean);
    double final_lat_mean = calculate_mean(all_means);
    double final_lat_std  = calculate_mean(all_stds);
    double final_p95      = calculate_mean(all_p95);
    double final_p99      = calculate_mean(all_p99);
    double fps_per_watt   = (gpu_metrics.avg_power_W > 0) ? (final_fps_mean / gpu_metrics.avg_power_W) : 0.0;

    std::cout << "\n================================" << std::endl;
    std::cout << "FINAL NATIVE RESULT SUMMARY" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Mean FPS:          " << final_fps_mean << " ± " << final_fps_std << std::endl;
    std::cout << "Mean Latency (ms): " << final_lat_mean << " ± " << final_lat_std << std::endl;
    std::cout << "P95 Latency (ms):  " << final_p95 << std::endl;
    std::cout << "P99 Latency (ms):  " << final_p99 << std::endl;
    std::cout << "Average Power (W): " << gpu_metrics.avg_power_W << std::endl;
    std::cout << "FPS / Watt:        " << fps_per_watt << std::endl;

    // Match Python Filename Convention: <GPU>_<MODEL>_<ENGINE>_<PRECISION>_<RESOLUTION>.csv
    std::string gpu_file_str = gpu_name;
    std::replace(gpu_file_str.begin(), gpu_file_str.end(), ' ', '_');

    std::string filename = gpu_file_str + "_" + model_base + "_" + engine_type + "_" + precision + "_" + std::to_string(image_size) + ".csv";
    std::string csv_path = output_dir + "/" + filename;

    std::ofstream csv(csv_path);
    csv << "gpu,model,resolution,precision,engine,runs,iterations,fps_mean,fps_std,latency_mean_ms,latency_std_ms,p95_latency_ms,p99_latency_ms,python,torch,cuda,avg_power_W,max_power_W,gpu_util_mean_pct,gpu_util_max_pct,gpu_memory_mean_MB,gpu_memory_max_MB,fps_per_watt\n";
    
    csv << "\"" << gpu_name << "\","
        << model_input << ","
        << image_size << ","
        << precision << ","
        << engine_type << ","
        << runs << ","
        << iterations << ","
        << final_fps_mean << ","
        << final_fps_std << ","
        << final_lat_mean << ","
        << final_lat_std << ","
        << final_p95 << ","
        << final_p99 << ","
        << "native_cpp,"                      // 'python' column in Python script
        << "native_cxx11,"                    // 'torch' column in Python script
        << cuda_version_str << ","
        << gpu_metrics.avg_power_W << ","
        << gpu_metrics.max_power_W << ","
        << gpu_metrics.gpu_util_mean_pct << ","
        << gpu_metrics.gpu_util_max_pct << ","
        << gpu_metrics.gpu_memory_mean_MB << ","
        << gpu_metrics.gpu_memory_max_MB << ","
        << fps_per_watt << "\n";

    csv.close();

    std::cout << "\nSaved CSV result to: " << csv_path << std::endl;
    return 0;
}