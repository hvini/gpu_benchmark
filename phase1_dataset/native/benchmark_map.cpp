/**
 * Phase 1 – Dataset Validation: Native C++ Latency Benchmark
 *
 * Measures inference latency on real COCO val2017 images.
 * mAP is intentionally NOT computed here; it is the exclusive responsibility
 * of the Python eval_map.py script (which uses Ultralytics' own pipeline,
 * making it the authoritative source of accuracy numbers).
 *
 * This binary's sole purpose is to produce latency statistics measured on
 * real image content (letterbox-preprocessed), unlike the phase0 benchmark
 * which uses a static synthetic tensor.
 *
 * Output
 * ──────
 * <OUTPUT_DIR>/<GPU>_<MODEL>_<ENGINE>_<PRECISION>_<SIZE>_native.csv
 *   Columns: gpu, model, resolution, precision, engine,
 *            images_processed, images_skipped,
 *            preprocess_ms, inference_ms, inference_std_ms,
 *            total_latency_ms, fps_e2e,
 *            p95_latency_ms, p99_latency_ms,
 *            python, torch, cuda,
 *            avg_power_W, max_power_W,
 *            gpu_util_mean_pct, gpu_util_max_pct,
 *            gpu_memory_mean_MB, gpu_memory_max_MB,
 *            fps_per_watt
 *
 * Configuration via environment variables (see main()).
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <chrono>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Engine headers
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

#include "../../shared/GPUMonitor.hpp"

namespace fs = std::filesystem;

// ──────────────────────────────────────────────────────────────────────────────
// Statistics helpers
// ──────────────────────────────────────────────────────────────────────────────
static double calc_mean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}
static double calc_std(const std::vector<double>& v, double mean) {
    if (v.empty()) return 0.0;
    double sq = 0.0;
    for (double x : v) sq += (x - mean) * (x - mean);
    return std::sqrt(sq / static_cast<double>(v.size()));
}
static double calc_percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(std::ceil(p / 100.0 * v.size())) - 1;
    return v[std::min(idx, v.size() - 1)];
}

// ──────────────────────────────────────────────────────────────────────────────
// Preprocessing: letterbox (grey pad, preserve aspect ratio)
// Mirrors Ultralytics' pipeline for realistic CPU-side preprocessing latency.
// ──────────────────────────────────────────────────────────────────────────────
static cv::Mat letterbox(const cv::Mat& src, int imgsz, int& pad_w, int& pad_h) {
    float scale = std::min(static_cast<float>(imgsz) / src.cols,
                           static_cast<float>(imgsz) / src.rows);
    int new_w = static_cast<int>(std::round(src.cols * scale));
    int new_h = static_cast<int>(std::round(src.rows * scale));

    int total_pad_w = imgsz - new_w;
    int total_pad_h = imgsz - new_h;
    pad_w = total_pad_w / 2;
    pad_h = total_pad_h / 2;
    int right  = total_pad_w - pad_w;
    int bottom = total_pad_h - pad_h;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_h, bottom, pad_w, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // BGR → RGB, uint8 → float32 [0, 1]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    return rgb;
}

// Pack HWC float32 cv::Mat into a flat vector<float> in NCHW order
static std::vector<float> hwc_to_nchw(const cv::Mat& img) {
    int H = img.rows, W = img.cols, C = img.channels();
    std::vector<float> out(C * H * W);
    std::vector<cv::Mat> channels(C);
    cv::split(img, channels);
    for (int c = 0; c < C; ++c)
        std::memcpy(out.data() + c * H * W, channels[c].data, H * W * sizeof(float));
    return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// Abstract Engine Interface
// Infer() accepts a pre-packed NCHW float32 blob and discards the output —
// we only care about timing, not predictions.
// ──────────────────────────────────────────────────────────────────────────────
class InferenceEngine {
public:
    virtual void load(const std::string& model_path, int img_size,
                      const std::string& precision) = 0;
    virtual void infer(const std::vector<float>& nchw_input) = 0;
    virtual ~InferenceEngine() = default;
};

// ──────────────────────────────────────────────────────────────────────────────
// 1. PyTorch Engine
// ──────────────────────────────────────────────────────────────────────────────
#ifdef ENABLE_TORCH
class PyTorchEngine : public InferenceEngine {
    torch::jit::script::Module module_;
    torch::Device               device_{torch::kCUDA};
    torch::ScalarType           dtype_{torch::kFloat32};
    int imgsz_{640};

public:
    void load(const std::string& model_path, int img_size,
              const std::string& precision) override {
        imgsz_ = img_size;
        dtype_ = (precision == "fp16") ? torch::kHalf : torch::kFloat32;
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        module_.to(dtype_);
    }

    void infer(const std::vector<float>& nchw_input) override {
        torch::NoGradGuard no_grad;
        auto opts  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto cpu_t = torch::from_blob(const_cast<float*>(nchw_input.data()),
                                       {1, 3, imgsz_, imgsz_}, opts);
        auto gpu_t = cpu_t.to(device_).to(dtype_);
        module_.forward({gpu_t});
    }
};
#endif

// ──────────────────────────────────────────────────────────────────────────────
// 2. ONNX Runtime Engine
// ──────────────────────────────────────────────────────────────────────────────
#ifdef ENABLE_ONNX
class ONNXEngine : public InferenceEngine {
    Ort::Env                      env_{ORT_LOGGING_LEVEL_WARNING, "YOLO_Phase1"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo               cpu_mem_{
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

    int    imgsz_{640};
    bool   is_fp16_in_{false};
    size_t input_numel_{0};
    std::vector<int64_t> input_shape_;
    std::vector<uint16_t> fp16_buf_;  // pre-allocated, reused across calls

public:
    void load(const std::string& model_path, int img_size,
              const std::string& precision) override {
        imgsz_       = img_size;
        input_numel_ = 1ULL * 3 * imgsz_ * imgsz_;
        input_shape_ = {1, 3, imgsz_, imgsz_};

        Ort::SessionOptions opts;
        opts.SetLogSeverityLevel(3);
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        OrtCUDAProviderOptions cuda_opts;
        cuda_opts.device_id = 0;
        cuda_opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        try {
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        } catch (...) {
            std::cout << "[WARN] CUDA EP unavailable; falling back to CPU." << std::endl;
        }

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);

        auto type_info = session_->GetInputTypeInfo(0);
        is_fp16_in_    = (type_info.GetTensorTypeAndShapeInfo().GetElementType()
                          == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

        if (is_fp16_in_) fp16_buf_.resize(input_numel_);
    }

    void infer(const std::vector<float>& nchw_input) override {
        std::vector<const char*> in_names  = {"images"};
        std::vector<const char*> out_names = {"output0"};

        if (is_fp16_in_) {
            // fp32 → fp16 via bit manipulation
            for (size_t i = 0; i < input_numel_; ++i) {
                float f = nchw_input[i];
                uint32_t fi; std::memcpy(&fi, &f, 4);
                uint16_t sign  = (fi >> 31) & 1;
                uint16_t exp16 = static_cast<uint16_t>(
                    std::max(0, std::min(31, static_cast<int>((fi >> 23 & 0xFF)) - (127 - 15))));
                uint16_t mant  = static_cast<uint16_t>((fi & 0x7FFFFF) >> 13);
                fp16_buf_[i]   = static_cast<uint16_t>((sign << 15) | (exp16 << 10) | mant);
            }
            auto in_t = Ort::Value::CreateTensor(
                cpu_mem_, fp16_buf_.data(), input_numel_ * 2,
                input_shape_.data(), input_shape_.size(),
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
            session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &in_t, 1,
                          out_names.data(), 1);
        } else {
            auto in_t = Ort::Value::CreateTensor<float>(
                cpu_mem_, const_cast<float*>(nchw_input.data()), input_numel_,
                input_shape_.data(), input_shape_.size());
            session_->Run(Ort::RunOptions{nullptr}, in_names.data(), &in_t, 1,
                          out_names.data(), 1);
        }
    }
};
#endif

// ──────────────────────────────────────────────────────────────────────────────
// 3. TensorRT Engine
// ──────────────────────────────────────────────────────────────────────────────
#ifdef ENABLE_TRT
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR)
            std::cout << "[TRT] " << msg << std::endl;
    }
} g_trt_logger;

class TRTEngine : public InferenceEngine {
    nvinfer1::IRuntime*          runtime_{nullptr};
    nvinfer1::ICudaEngine*       engine_{nullptr};
    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t                 stream_{nullptr};
    std::vector<void*>           d_buffers_;
    int                          in_idx_{0};
    size_t                       in_bytes_{0};

public:
    void load(const std::string& model_path, int /*img_size*/,
              const std::string& /*precision*/) override {
        std::ifstream f(model_path, std::ios::binary);
        if (!f.good()) throw std::runtime_error("Engine file not found: " + model_path);
        f.seekg(0, std::ios::end); size_t sz = f.tellg(); f.seekg(0);
        std::vector<char> buf(sz); f.read(buf.data(), sz); f.close();

        runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
        engine_  = runtime_->deserializeCudaEngine(buf.data(), sz);
        if (!engine_) throw std::runtime_error(
            "TRT engine deserialize failed. Delete *.engine and re-run.");
        ctx_ = engine_->createExecutionContext();
        cudaStreamCreate(&stream_);

        int32_t n = engine_->getNbIOTensors();
        d_buffers_.resize(n, nullptr);

        for (int32_t i = 0; i < n; ++i) {
            const char* name = engine_->getIOTensorName(i);
            auto mode        = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            nvinfer1::DataType dt = engine_->getTensorDataType(name);

            size_t vol = 1;
            for (int d = 0; d < dims.nbDims; ++d)
                vol *= std::max(1, static_cast<int>(dims.d[d]));

            size_t elem = (dt == nvinfer1::DataType::kHALF) ? 2
                        : (dt == nvinfer1::DataType::kINT8)  ? 1 : 4;
            cudaMalloc(&d_buffers_[i], vol * elem);
            ctx_->setTensorAddress(name, d_buffers_[i]);

            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                in_idx_   = i;
                in_bytes_ = vol * elem;
            }
        }
    }

    void infer(const std::vector<float>& nchw_input) override {
        cudaMemcpyAsync(d_buffers_[in_idx_], nchw_input.data(),
                        nchw_input.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream_);
        ctx_->enqueueV3(stream_);
        cudaStreamSynchronize(stream_);
    }

    ~TRTEngine() override {
        for (void* b : d_buffers_) if (b) cudaFree(b);
        if (stream_)  cudaStreamDestroy(stream_);
        if (ctx_)     delete ctx_;
        if (engine_)  delete engine_;
        if (runtime_) delete runtime_;
    }
};
#endif

// ──────────────────────────────────────────────────────────────────────────────
// Image list: scan the images directory directly (no COCO JSON needed)
// ──────────────────────────────────────────────────────────────────────────────
static std::vector<std::string> collect_images(const std::string& dir) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        auto ext = entry.path().extension().string();
        // Convert to lowercase for comparison
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png")
            paths.push_back(entry.path().string());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// ──────────────────────────────────────────────────────────────────────────────
// main()
// ──────────────────────────────────────────────────────────────────────────────
int main(int /*argc*/, char** /*argv*/) {

    // ── Configuration from environment ──────────────────────────────────────
    const std::string engine_type = std::getenv("ENGINE")     ? std::getenv("ENGINE")     : "pytorch";
    const std::string precision   = std::getenv("PRECISION")  ? std::getenv("PRECISION")  : "fp16";
    const int         image_size  = std::getenv("IMAGE_SIZE") ? std::stoi(std::getenv("IMAGE_SIZE")) : 640;
    const int         warmup      = std::getenv("WARMUP")     ? std::stoi(std::getenv("WARMUP"))     : 50;
    const std::string model_input = std::getenv("MODEL")      ? std::getenv("MODEL")      : "yolo11s.pt";
    const std::string output_dir  = std::getenv("OUTPUT_DIR") ? std::getenv("OUTPUT_DIR") : "/workspace/results";
    // Directory containing COCO val2017 images
    const std::string images_dir  = std::getenv("COCO_IMAGES_DIR")
                                  ? std::getenv("COCO_IMAGES_DIR")
                                  : "/workspace/datasets/coco/images/val2017";

    // ── GPU info ─────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu_name = prop.name;
    int rt_ver = 0; cudaRuntimeGetVersion(&rt_ver);
    std::string cuda_ver = std::to_string(rt_ver / 1000) + "." +
                           std::to_string((rt_ver % 1000) / 10);

    std::string model_base = model_input;
    if (model_base.size() > 3 && model_base.substr(model_base.size() - 3) == ".pt")
        model_base = model_base.substr(0, model_base.size() - 3);

    std::cout << "========================================\n"
              << "Phase 1 – Dataset Validation (Native C++)\n"
              << "========================================\n"
              << "GPU:        " << gpu_name    << "\n"
              << "Model:      " << model_input << "\n"
              << "Engine:     " << engine_type << "\n"
              << "Precision:  " << precision   << "\n"
              << "Resolution: " << image_size  << "x" << image_size << "\n"
              << "Warmup:     " << warmup      << " images\n"
              << "CUDA:       " << cuda_ver    << "\n"
              << "Images dir: " << images_dir  << "\n"
              << "========================================\n\n";

    // ── Load Engine ──────────────────────────────────────────────────────────
    std::unique_ptr<InferenceEngine> engine;
    std::string model_ext;

    if (engine_type == "pytorch") {
#ifdef ENABLE_TORCH
        engine = std::make_unique<PyTorchEngine>(); model_ext = ".torchscript";
#else
        std::cerr << "LibTorch not enabled at compile time!\n"; return 1;
#endif
    } else if (engine_type == "onnx") {
#ifdef ENABLE_ONNX
        engine = std::make_unique<ONNXEngine>(); model_ext = ".onnx";
#else
        std::cerr << "ONNX Runtime not enabled at compile time!\n"; return 1;
#endif
    } else if (engine_type == "tensorrt") {
#ifdef ENABLE_TRT
        engine = std::make_unique<TRTEngine>(); model_ext = ".engine";
#else
        std::cerr << "TensorRT not enabled at compile time!\n"; return 1;
#endif
    } else {
        std::cerr << "Unknown engine: " << engine_type << "\n"; return 1;
    }

    std::string model_path =
        model_base + "_" + std::to_string(image_size) + "_" + precision + model_ext;
    std::cout << "Loading model: " << model_path << "\n";
    engine->load(model_path, image_size, precision);

    // ── Collect image paths ──────────────────────────────────────────────────
    std::cout << "Scanning images: " << images_dir << "\n";
    std::vector<std::string> image_paths = collect_images(images_dir);
    if (image_paths.empty()) {
        std::cerr << "[ERROR] No images found in " << images_dir << "\n";
        return 1;
    }
    int total = static_cast<int>(image_paths.size());
    std::cout << "Found " << total << " images.\n\n";

    // ── Warmup ───────────────────────────────────────────────────────────────
    std::cout << "Warming up (" << warmup << " images)...\n";
    for (int w = 0; w < warmup && w < total; ++w) {
        cv::Mat img = cv::imread(image_paths[w]);
        if (img.empty()) continue;
        int pw, ph;
        cv::Mat rgb = letterbox(img, image_size, pw, ph);
        auto nchw   = hwc_to_nchw(rgb);
        engine->infer(nchw);
    }
    cudaDeviceSynchronize();
    std::cout << "Warmup complete.\n\n";

    // ── GPU monitoring ────────────────────────────────────────────────────────
    GPUMonitor gpu_monitor;
    gpu_monitor.start();

    // ── Main evaluation loop ──────────────────────────────────────────────────
    std::vector<double> preprocess_ms_vec, infer_ms_vec;
    preprocess_ms_vec.reserve(total);
    infer_ms_vec.reserve(total);

    int processed = 0, skipped = 0;

    for (int idx = 0; idx < total; ++idx) {
        // ── Preprocess (CPU, wall-clock) ─────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();

        cv::Mat img = cv::imread(image_paths[idx]);
        if (img.empty()) { ++skipped; continue; }

        int pw, ph;
        cv::Mat rgb = letterbox(img, image_size, pw, ph);
        auto nchw   = hwc_to_nchw(rgb);

        auto t1 = std::chrono::high_resolution_clock::now();

        // ── Inference (CUDA events for GPU-accurate timing) ───────────────────
        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        cudaEventRecord(ev_start);
        engine->infer(nchw);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);

        float gpu_ms = 0.f;
        cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop);
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);

        double pre_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        preprocess_ms_vec.push_back(pre_ms);
        infer_ms_vec.push_back(static_cast<double>(gpu_ms));

        ++processed;

        if ((idx + 1) % 500 == 0 || idx + 1 == total) {
            double pct = 100.0 * (idx + 1) / total;
            std::cout << "[" << (idx + 1) << "/" << total << " | "
                      << std::fixed << std::setprecision(1) << pct << "%] "
                      << "preprocess=" << std::setprecision(2) << pre_ms
                      << "ms  infer=" << gpu_ms << "ms\n";
        }
    }

    gpu_monitor.stop();
    GPUMetrics gpu_metrics = gpu_monitor.get_results();

    // ── Summary statistics ────────────────────────────────────────────────────
    double pre_mean  = calc_mean(preprocess_ms_vec);
    double inf_mean  = calc_mean(infer_ms_vec);
    double inf_std   = calc_std(infer_ms_vec, inf_mean);
    double inf_p95   = calc_percentile(infer_ms_vec, 95.0);
    double inf_p99   = calc_percentile(infer_ms_vec, 99.0);
    double total_lat = pre_mean + inf_mean;
    double fps_e2e   = (total_lat > 0) ? 1000.0 / total_lat : 0.0;
    double fps_per_w = (gpu_metrics.avg_power_W > 0)
                     ? fps_e2e / gpu_metrics.avg_power_W : 0.0;

    std::cout << "\n================================\n"
              << "PHASE 1 NATIVE RESULT SUMMARY\n"
              << "================================\n"
              << "Images processed : " << processed << " / " << total
              << "  (skipped=" << skipped << ")\n"
              << "Preprocess  (ms) : " << std::fixed << std::setprecision(3) << pre_mean  << "\n"
              << "Inference   (ms) : " << inf_mean   << " ± " << inf_std  << "\n"
              << "Total lat   (ms) : " << total_lat  << "\n"
              << "FPS (e2e)        : " << std::setprecision(2) << fps_e2e   << "\n"
              << "P95 infer   (ms) : " << std::setprecision(3) << inf_p95   << "\n"
              << "P99 infer   (ms) : " << inf_p99   << "\n"
              << "Avg Power   (W)  : " << gpu_metrics.avg_power_W  << "\n"
              << "FPS / Watt       : " << std::setprecision(4) << fps_per_w << "\n";

    // ── Write CSV ─────────────────────────────────────────────────────────────
    fs::create_directories(output_dir);
    std::string gpu_file_str = gpu_name;
    std::replace(gpu_file_str.begin(), gpu_file_str.end(), ' ', '_');
    std::string base_name = gpu_file_str + "_" + model_base + "_" +
                            engine_type + "_" + precision + "_" +
                            std::to_string(image_size) + "_native";
    std::string csv_path = output_dir + "/" + base_name + ".csv";

    std::ofstream csv(csv_path);
    csv << "gpu,model,resolution,precision,engine,"
        << "images_processed,images_skipped,"
        << "preprocess_ms,inference_ms,inference_std_ms,"
        << "total_latency_ms,fps_e2e,"
        << "p95_latency_ms,p99_latency_ms,"
        << "python,torch,cuda,"
        << "avg_power_W,max_power_W,"
        << "gpu_util_mean_pct,gpu_util_max_pct,"
        << "gpu_memory_mean_MB,gpu_memory_max_MB,"
        << "fps_per_watt\n";

    csv << "\"" << gpu_name << "\","
        << model_input      << ","
        << image_size       << ","
        << precision        << ","
        << engine_type      << ","
        << processed        << ","
        << skipped          << ","
        << pre_mean         << ","
        << inf_mean         << ","
        << inf_std          << ","
        << total_lat        << ","
        << fps_e2e          << ","
        << inf_p95          << ","
        << inf_p99          << ","
        << "native_cpp,"
        << "native_cxx17,"
        << cuda_ver         << ","
        << gpu_metrics.avg_power_W        << ","
        << gpu_metrics.max_power_W        << ","
        << gpu_metrics.gpu_util_mean_pct  << ","
        << gpu_metrics.gpu_util_max_pct   << ","
        << gpu_metrics.gpu_memory_mean_MB << ","
        << gpu_metrics.gpu_memory_max_MB  << ","
        << fps_per_w        << "\n";
    csv.close();

    std::cout << "Results CSV : " << csv_path << "\n";
    return 0;
}
