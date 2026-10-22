/**
 * Phase 2 - COCO real-image native inference benchmark.
 *
 * Goal:
 *   Make the native per-engine `inference_ms` as comparable as practical to
 *   Ultralytics `metrics.speed['inference']`: input tensors are prepared before
 *   the timed model/runtime call, and NMS/mAP are intentionally not computed.
 *
 * Reported stages:
 *   image_load_ms   : cv::imread + image decode
 *   preprocess_ms   : resize/letterbox + BGR->RGB + normalize + NCHW packing
 *   input_prepare_ms: CPU->GPU upload and input dtype conversion where needed
 *   inference_ms    : model/runtime call with input resident in its expected
 *                     device memory; backend-specific caveat for ONNX noted below
 *   total_latency_ms: sum of the above stages; native file-to-forward latency,
 *                     not directly equivalent to Ultralytics validation total
 *                     because this program does not run NMS or update mAP.
 *
 * ONNX uses CUDA I/O binding with device input/output buffers. This matches the
 * Ultralytics CUDA I/O-binding path used for static-shape ONNX models. If your
 * installed Ultralytics version ran ONNX without I/O binding, its stored Python
 * inference_ms includes additional transfers and is not strictly comparable.
 * PyTorch native uses TorchScript while the supplied Python PyTorch benchmark
 * uses the original .pt checkpoint; a strict artifact-to-artifact comparison
 * would require a small Python TorchScript timing check, not a full COCO rerun.
 *
 * Required environment variables:
 *   ENGINE=pytorch|onnx|tensorrt
 *   PRECISION=fp32|fp16|int8
 *   IMAGE_SIZE=640|1280|1920
 *   MODEL=yolo11s.pt
 *   WARMUP=50
 *   OUTPUT_DIR=/workspace/results
 *   COCO_IMAGES_DIR=/workspace/datasets/coco/images/val2017
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
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <cctype>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

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
using Clock = std::chrono::steady_clock;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static double elapsed_ms(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

static void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
    }
}

static double calc_mean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

static double calc_std(const std::vector<double>& values, double mean) {
    if (values.empty()) return 0.0;
    double sum_sq = 0.0;
    for (double value : values) sum_sq += (value - mean) * (value - mean);
    return std::sqrt(sum_sq / static_cast<double>(values.size()));
}

static double calc_percentile(std::vector<double> values, double percentile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(percentile / 100.0 * values.size()));
    index = index == 0 ? 0 : index - 1;
    return values[std::min(index, values.size() - 1)];
}

// Correct round-to-nearest-even conversion from IEEE-754 float32 bits to float16 bits.
static uint16_t float_to_half_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;

    if (exponent == 0xffu) {
        if (mantissa == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        // Preserve a nonzero NaN payload.
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        if (payload == 0) payload = 1;
        return static_cast<uint16_t>(sign | 0x7c00u | payload);
    }

    int half_exp = static_cast<int>(exponent) - 127 + 15;
    if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);

    if (half_exp <= 0) {
        if (half_exp < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - half_exp;
        uint32_t rounded = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) ++rounded;
        return static_cast<uint16_t>(sign | rounded);
    }

    // Round mantissa to 10 bits, ties to even.
    uint32_t rounded_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded_mantissa & 1u))) {
        ++rounded_mantissa;
        if (rounded_mantissa == 0x400u) {
            rounded_mantissa = 0;
            ++half_exp;
            if (half_exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(half_exp) << 10) |
                                 rounded_mantissa);
}

#ifdef ENABLE_ONNX
// Avoid the convenience GetShape() wrapper until the ONNX Runtime headers and
// shared library have been verified to match. GetShape() allocates a vector
// using GetDimensionsCount(); a corrupted/ABI-mismatched count can cause
// std::length_error before a useful diagnostic is printed.
template <typename TensorInfoT>
static std::vector<int64_t> get_ort_shape_checked(
        const TensorInfoT& info, const std::string& label) {
    const size_t rank = info.GetDimensionsCount();
    if (rank == 0 || rank > 16) {
        throw std::runtime_error(label + ": implausible ONNX tensor rank=" +
            std::to_string(rank) + ". Check that the ONNX Runtime headers in /opt/onnxruntime/include "
            "match the libonnxruntime.so loaded at runtime, and verify the ONNX artifact.");
    }
    std::vector<int64_t> shape(rank);
    info.GetDimensions(shape.data(), rank);
    return shape;
}

static size_t ort_element_size(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
        default: throw std::runtime_error("Unsupported ONNX output tensor type for preallocation.");
    }
}
#endif

#ifdef ENABLE_TRT
static size_t trt_element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: throw std::runtime_error("Unsupported TensorRT tensor data type.");
    }
}
#endif

// -----------------------------------------------------------------------------
// OpenCV preprocessing
// -----------------------------------------------------------------------------
// Keep pixels uint8/BGR during letterboxing; use one fused CPU pass for channel
// swap, normalization and NCHW packing to avoid cv::split + extra full-image copies.
static cv::Mat letterbox_bgr(const cv::Mat& source, int image_size) {
    if (source.empty()) throw std::runtime_error("Cannot letterbox an empty image.");
    const float scale = std::min(static_cast<float>(image_size) / source.cols,
                                 static_cast<float>(image_size) / source.rows);
    const int new_w = static_cast<int>(std::round(source.cols * scale));
    const int new_h = static_cast<int>(std::round(source.rows * scale));

    cv::Mat resized;
    cv::resize(source, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    const int total_pad_w = image_size - new_w;
    const int total_pad_h = image_size - new_h;
    const int left = total_pad_w / 2;
    const int right = total_pad_w - left;
    const int top = total_pad_h / 2;
    const int bottom = total_pad_h - top;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return padded;
}

static cv::Mat preprocess_to_nchw(const cv::Mat& padded_bgr, int image_size) {
    const int shape[4] = {1, 3, image_size, image_size};
    cv::Mat blob(4, shape, CV_32F);
    float* output = blob.ptr<float>();
    const size_t plane_size = static_cast<size_t>(image_size) * image_size;

    // Single HWC BGR uint8 -> CHW RGB FP32 normalized pass. Rows can be
    // processed independently, and this avoids cv::split plus extra full-image copies.
    cv::parallel_for_(cv::Range(0, image_size), [&](const cv::Range& rows) {
        for (int y = rows.start; y < rows.end; ++y) {
            const cv::Vec3b* source_row = padded_bgr.ptr<cv::Vec3b>(y);
            const size_t row_offset = static_cast<size_t>(y) * image_size;
            for (int x = 0; x < image_size; ++x) {
                const size_t index = row_offset + x;
                const cv::Vec3b pixel = source_row[x];
                output[index] = static_cast<float>(pixel[2]) * (1.0f / 255.0f); // R
                output[plane_size + index] = static_cast<float>(pixel[1]) * (1.0f / 255.0f); // G
                output[2 * plane_size + index] = static_cast<float>(pixel[0]) * (1.0f / 255.0f); // B
            }
        }
    });
    return blob;  // Shape [1,3,H,W], normalized RGB float32.
}

static std::vector<std::string> collect_images(const std::string& directory) {
    std::vector<std::string> paths;
    if (!fs::exists(directory)) {
        throw std::runtime_error("Images directory does not exist: " + directory);
    }
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// -----------------------------------------------------------------------------
// Engine interface
// -----------------------------------------------------------------------------
struct StageTiming {
    double input_prepare_ms = 0.0;
    double inference_ms = 0.0;
};

class InferenceEngine {
public:
    virtual void load(const std::string& model_path, int image_size,
                      const std::string& precision) = 0;
    virtual StageTiming infer(const float* nchw_input, size_t element_count) = 0;
    virtual ~InferenceEngine() = default;
};

// -----------------------------------------------------------------------------
// 1. PyTorch / TorchScript
// -----------------------------------------------------------------------------
#ifdef ENABLE_TORCH
class PyTorchEngine final : public InferenceEngine {
    torch::jit::script::Module module_;
    torch::Device device_{torch::kCUDA, 0};
    torch::ScalarType dtype_{torch::kFloat32};
    int image_size_{640};
    size_t expected_elements_{0};

public:
    void load(const std::string& model_path, int image_size,
              const std::string& precision) override {
        image_size_ = image_size;
        expected_elements_ = 3ULL * image_size_ * image_size_;
        dtype_ = (precision == "fp16") ? torch::kFloat16 : torch::kFloat32;
        module_ = torch::jit::load(model_path, device_);
        module_.eval();
        module_.to(dtype_);
        cuda_check(cudaDeviceSynchronize(), "synchronize after loading TorchScript");
    }

    StageTiming infer(const float* input, size_t element_count) override {
        if (element_count != expected_elements_) {
            throw std::runtime_error("Unexpected PyTorch input element count.");
        }
        torch::NoGradGuard no_grad;
        auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto start_prepare = Clock::now();
        auto cpu_tensor = torch::from_blob(
            const_cast<float*>(input), {1, 3, image_size_, image_size_}, cpu_options);
        auto gpu_tensor = cpu_tensor.to(device_);
        if (dtype_ != torch::kFloat32) gpu_tensor = gpu_tensor.to(dtype_);
        cuda_check(cudaDeviceSynchronize(), "synchronize PyTorch input preparation");
        const double prepare_ms = elapsed_ms(start_prepare, Clock::now());

        const auto start_inference = Clock::now();
        auto output = module_.forward({gpu_tensor});
        cuda_check(cudaDeviceSynchronize(), "synchronize TorchScript inference");
        const double inference_ms = elapsed_ms(start_inference, Clock::now());
        (void)output;  // We measure the model call, not NMS or metric calculation.
        return {prepare_ms, inference_ms};
    }
};
#endif

// -----------------------------------------------------------------------------
// 2. ONNX Runtime with CUDA I/O binding
// -----------------------------------------------------------------------------
#ifdef ENABLE_ONNX
class ONNXEngine final : public InferenceEngine {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "YOLO_Phase2_Native"};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo cuda_memory_{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
    std::unique_ptr<Ort::IoBinding> binding_;
    std::unique_ptr<Ort::Value> input_value_;
    std::vector<Ort::Value> output_values_;
    std::vector<void*> output_buffers_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    std::vector<int64_t> input_shape_;
    ONNXTensorElementDataType input_type_{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
    void* device_input_{nullptr};
    size_t input_elements_{0};
    size_t input_bytes_{0};
    int image_size_{640};
    std::vector<uint16_t> fp16_host_buffer_;

public:
    void load(const std::string& model_path, int image_size,
              const std::string& /*precision*/) override {
        image_size_ = image_size;
        input_elements_ = 3ULL * image_size_ * image_size_;

        std::cerr << "[ONNX:init] creating session from " << model_path << std::endl;
        Ort::SessionOptions options;
        options.SetLogSeverityLevel(3);
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        options.AppendExecutionProvider_CUDA(cuda_options);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
        std::cerr << "[ONNX:init] session created" << std::endl;

        // CUDAExecutionProvider was explicitly configured in SessionOptions above.
        // Avoid Ort::GetAvailableProviders() here: with mismatched ONNX Runtime
        // headers/runtime libraries this wrapper can throw std::length_error
        // while constructing its provider vector. Session creation above is the
        // meaningful availability check for this benchmark's configured provider.

        const size_t input_count = session_->GetInputCount();
        std::cerr << "[ONNX:init] input count=" << input_count << std::endl;
        if (input_count == 0 || input_count > 64) {
            throw std::runtime_error("Implausible ONNX input count=" + std::to_string(input_count) +
                ". Check ONNX Runtime headers/library compatibility.");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        // Newer ONNX Runtime C++ APIs expose the RAII Allocated variants.
        auto raw_input_name = session_->GetInputNameAllocated(0, allocator);
        input_name_ = raw_input_name.get();
        std::cerr << "[ONNX:init] input name=" << input_name_ << std::endl;

        const auto input_type_info = session_->GetInputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_type_ = input_info.GetElementType();
        input_shape_ = get_ort_shape_checked(input_info, "ONNX input");
        std::cerr << "[ONNX:init] input type=" << static_cast<int>(input_type_) << " shape=";
        for (const auto dim : input_shape_) std::cerr << dim << ' ';
        std::cerr << std::endl;
        if (input_shape_.size() != 4) {
            throw std::runtime_error("ONNX model input is not a 4D NCHW tensor.");
        }
        const int64_t desired[4] = {1, 3, image_size_, image_size_};
        for (size_t i = 0; i < input_shape_.size(); ++i) {
            if (input_shape_[i] <= 0) input_shape_[i] = desired[i];
            if (input_shape_[i] != desired[i]) {
                throw std::runtime_error("ONNX input shape does not match configured image size.");
            }
        }
        if (input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
            input_type_ != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            throw std::runtime_error("ONNX input must be FP32 or FP16 for this benchmark.");
        }
        input_bytes_ = input_elements_ *
            (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? sizeof(uint16_t) : sizeof(float));
        cuda_check(cudaMalloc(&device_input_, input_bytes_), "allocate ONNX device input");
        if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            fp16_host_buffer_.resize(input_elements_);
        }

        input_value_ = std::make_unique<Ort::Value>(Ort::Value::CreateTensor(
            cuda_memory_, device_input_, input_bytes_, input_shape_.data(), input_shape_.size(), input_type_));
        binding_ = std::make_unique<Ort::IoBinding>(*session_);
        binding_->BindInput(input_name_.c_str(), *input_value_);

        const size_t output_count = session_->GetOutputCount();
        std::cerr << "[ONNX:init] output count=" << output_count << std::endl;
        if (output_count == 0 || output_count > 64) {
            throw std::runtime_error("Implausible ONNX output count=" + std::to_string(output_count) +
                ". Check ONNX Runtime headers/library compatibility.");
        }
        output_values_.reserve(output_count);
        output_buffers_.reserve(output_count);
        output_names_.reserve(output_count);
        for (size_t i = 0; i < output_count; ++i) {
            auto raw_name = session_->GetOutputNameAllocated(i, allocator);
            std::string name(raw_name.get());
            output_names_.push_back(name);

            const auto output_type_info = session_->GetOutputTypeInfo(i);
            const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
            const auto output_shape = get_ort_shape_checked(output_info, "ONNX output " + std::to_string(i));
            std::cerr << "[ONNX:init] output " << i << " name=" << name << " type="
                      << static_cast<int>(output_info.GetElementType()) << " shape=";
            for (const auto dim : output_shape) std::cerr << dim << ' ';
            std::cerr << std::endl;
            size_t elements = 1;
            bool static_shape = !output_shape.empty();
            for (int64_t dim : output_shape) {
                if (dim <= 0) { static_shape = false; break; }
                elements *= static_cast<size_t>(dim);
            }
            if (!static_shape) {
                // Dynamic outputs are allocated by ORT on the CUDA device.
                binding_->BindOutput(name.c_str(), cuda_memory_);
                output_buffers_.push_back(nullptr);
            } else {
                const size_t bytes = elements * ort_element_size(output_info.GetElementType());
                void* output_device = nullptr;
                cuda_check(cudaMalloc(&output_device, bytes), "allocate ONNX device output");
                output_buffers_.push_back(output_device);
                output_values_.push_back(Ort::Value::CreateTensor(
                    cuda_memory_, output_device, bytes, output_shape.data(), output_shape.size(),
                    output_info.GetElementType()));
                binding_->BindOutput(name.c_str(), output_values_.back());
            }
        }
        cuda_check(cudaDeviceSynchronize(), "synchronize after loading ONNX model");
    }

    StageTiming infer(const float* input, size_t element_count) override {
        if (element_count != input_elements_) {
            throw std::runtime_error("Unexpected ONNX input element count.");
        }
        const auto start_prepare = Clock::now();
        if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            for (size_t i = 0; i < input_elements_; ++i) fp16_host_buffer_[i] = float_to_half_bits(input[i]);
            cuda_check(cudaMemcpy(device_input_, fp16_host_buffer_.data(), input_bytes_,
                                  cudaMemcpyHostToDevice), "copy FP16 ONNX input to GPU");
        } else {
            cuda_check(cudaMemcpy(device_input_, input, input_bytes_, cudaMemcpyHostToDevice),
                       "copy FP32 ONNX input to GPU");
        }
        cuda_check(cudaDeviceSynchronize(), "synchronize ONNX input preparation");
        const double prepare_ms = elapsed_ms(start_prepare, Clock::now());

        const auto start_inference = Clock::now();
        Ort::RunOptions run_options;
        session_->Run(run_options, *binding_);
        binding_->SynchronizeOutputs();
        cuda_check(cudaDeviceSynchronize(), "synchronize ONNX inference");
        const double inference_ms = elapsed_ms(start_inference, Clock::now());
        return {prepare_ms, inference_ms};
    }

    ~ONNXEngine() override {
        // Drop bindings/views before releasing the device memory they reference.
        binding_.reset();
        output_values_.clear();
        input_value_.reset();
        if (device_input_) cudaFree(device_input_);
        for (void* buffer : output_buffers_) if (buffer) cudaFree(buffer);
    }
};
#endif

// -----------------------------------------------------------------------------
// 3. TensorRT
// -----------------------------------------------------------------------------
#ifdef ENABLE_TRT
class TRTLogger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "[TensorRT] " << message << std::endl;
    }
} g_trt_logger;

class TRTEngine final : public InferenceEngine {
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    cudaStream_t stream_{nullptr};
    std::vector<void*> buffers_;
    std::string input_name_;
    size_t input_elements_{0};
    size_t input_bytes_{0};
    nvinfer1::DataType input_dtype_{nvinfer1::DataType::kFLOAT};
    std::vector<uint16_t> fp16_host_buffer_;

public:
    void load(const std::string& model_path, int image_size,
              const std::string& /*precision*/) override {
        input_elements_ = 3ULL * image_size * image_size;
        std::ifstream file(model_path, std::ios::binary);
        if (!file) throw std::runtime_error("TensorRT engine file not found: " + model_path);
        file.seekg(0, std::ios::end);
        const size_t file_size = static_cast<size_t>(file.tellg());
        file.seekg(0, std::ios::beg);
        std::vector<char> serialized(file_size);
        file.read(serialized.data(), static_cast<std::streamsize>(file_size));
        if (!file) throw std::runtime_error("Could not read complete TensorRT engine file.");

        runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
        if (!runtime_) throw std::runtime_error("TensorRT createInferRuntime failed.");
        engine_ = runtime_->deserializeCudaEngine(serialized.data(), file_size);
        if (!engine_) throw std::runtime_error("TensorRT deserializeCudaEngine failed.");
        context_ = engine_->createExecutionContext();
        if (!context_) throw std::runtime_error("TensorRT createExecutionContext failed.");
        cuda_check(cudaStreamCreate(&stream_), "create TensorRT CUDA stream");

        const int io_count = engine_->getNbIOTensors();
        buffers_.assign(static_cast<size_t>(io_count), nullptr);
        for (int i = 0; i < io_count; ++i) {
            const char* name = engine_->getIOTensorName(i);
            const auto mode = engine_->getTensorIOMode(name);
            const auto dims = engine_->getTensorShape(name);
            size_t elements = 1;
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] <= 0) {
                    throw std::runtime_error("Dynamic TensorRT dimensions are unsupported; export with dynamic=false.");
                }
                elements *= static_cast<size_t>(dims.d[d]);
            }
            const auto dtype = engine_->getTensorDataType(name);
            const size_t bytes = elements * trt_element_size(dtype);
            cuda_check(cudaMalloc(&buffers_[static_cast<size_t>(i)], bytes), "allocate TensorRT I/O buffer");
            if (!context_->setTensorAddress(name, buffers_[static_cast<size_t>(i)])) {
                throw std::runtime_error(std::string("TensorRT setTensorAddress failed for ") + name);
            }
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name_ = name;
                input_dtype_ = dtype;
                input_bytes_ = bytes;
                if (elements != input_elements_) {
                    throw std::runtime_error("TensorRT input shape does not match configured image size.");
                }
            }
        }
        if (input_name_.empty()) throw std::runtime_error("TensorRT engine has no input tensor.");
        if (input_dtype_ != nvinfer1::DataType::kFLOAT && input_dtype_ != nvinfer1::DataType::kHALF) {
            throw std::runtime_error("TensorRT input must be FP32 or FP16; this engine exposes another input type.");
        }
        if (input_dtype_ == nvinfer1::DataType::kHALF) fp16_host_buffer_.resize(input_elements_);
        cuda_check(cudaDeviceSynchronize(), "synchronize after loading TensorRT engine");
    }

    StageTiming infer(const float* input, size_t element_count) override {
        if (element_count != input_elements_) {
            throw std::runtime_error("Unexpected TensorRT input element count.");
        }
        const auto start_prepare = Clock::now();
        void* device_input = nullptr;
        for (size_t i = 0; i < buffers_.size(); ++i) {
            // The input pointer is the only tensor whose name equals input_name_.
            // Use the context/engine metadata to resolve it without relying on I/O order.
            const char* name = engine_->getIOTensorName(static_cast<int32_t>(i));
            if (input_name_ == name) { device_input = buffers_[i]; break; }
        }
        if (!device_input) throw std::runtime_error("TensorRT input buffer was not allocated.");

        if (input_dtype_ == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < input_elements_; ++i) fp16_host_buffer_[i] = float_to_half_bits(input[i]);
            cuda_check(cudaMemcpyAsync(device_input, fp16_host_buffer_.data(), input_bytes_,
                                       cudaMemcpyHostToDevice, stream_), "copy FP16 TensorRT input to GPU");
        } else {
            cuda_check(cudaMemcpyAsync(device_input, input, input_bytes_,
                                       cudaMemcpyHostToDevice, stream_), "copy FP32 TensorRT input to GPU");
        }
        cuda_check(cudaStreamSynchronize(stream_), "synchronize TensorRT input preparation");
        const double prepare_ms = elapsed_ms(start_prepare, Clock::now());

        const auto start_inference = Clock::now();
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed.");
        cuda_check(cudaStreamSynchronize(stream_), "synchronize TensorRT inference");
        const double inference_ms = elapsed_ms(start_inference, Clock::now());
        return {prepare_ms, inference_ms};
    }

    ~TRTEngine() override {
        for (void* buffer : buffers_) if (buffer) cudaFree(buffer);
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) delete context_;
        if (engine_) delete engine_;
        if (runtime_) delete runtime_;
    }
};
#endif

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int /*argc*/, char** /*argv*/) {
    try {
        const std::string engine_type = std::getenv("ENGINE") ? std::getenv("ENGINE") : "pytorch";
        const std::string precision = std::getenv("PRECISION") ? std::getenv("PRECISION") : "fp16";
        const int image_size = std::getenv("IMAGE_SIZE") ? std::stoi(std::getenv("IMAGE_SIZE")) : 640;
        const int warmup = std::getenv("WARMUP") ? std::stoi(std::getenv("WARMUP")) : 50;
        const std::string model_input = std::getenv("MODEL") ? std::getenv("MODEL") : "yolo11s.pt";
        const std::string output_dir = std::getenv("OUTPUT_DIR") ? std::getenv("OUTPUT_DIR") : "/workspace/results";
        const std::string images_dir = std::getenv("COCO_IMAGES_DIR")
            ? std::getenv("COCO_IMAGES_DIR") : "/workspace/datasets/coco/images/val2017";

        if (image_size <= 0 || warmup < 0) throw std::runtime_error("IMAGE_SIZE must be positive and WARMUP nonnegative.");
        if (precision != "fp32" && precision != "fp16" && precision != "int8") {
            throw std::runtime_error("PRECISION must be fp32, fp16, or int8.");
        }
        if (engine_type == "pytorch" && precision == "int8") {
            throw std::runtime_error("This PyTorch path supports FP32/FP16 TorchScript only; it does not quantize the model to INT8.");
        }

        cuda_check(cudaSetDevice(0), "select CUDA device 0");
        cudaDeviceProp device_properties{};
        cuda_check(cudaGetDeviceProperties(&device_properties, 0), "read CUDA device properties");
        const std::string gpu_name = device_properties.name;
        int runtime_version = 0;
        cuda_check(cudaRuntimeGetVersion(&runtime_version), "read CUDA runtime version");
        const std::string cuda_version = std::to_string(runtime_version / 1000) + "." +
            std::to_string((runtime_version % 1000) / 10);

        std::string model_base = model_input;
        if (model_base.size() > 3 && model_base.substr(model_base.size() - 3) == ".pt") {
            model_base.resize(model_base.size() - 3);
        }

        std::unique_ptr<InferenceEngine> engine;
        std::string model_extension;
        if (engine_type == "pytorch") {
#ifdef ENABLE_TORCH
            engine = std::make_unique<PyTorchEngine>();
            model_extension = ".torchscript";
#else
            throw std::runtime_error("LibTorch support is not enabled in this build.");
#endif
        } else if (engine_type == "onnx") {
#ifdef ENABLE_ONNX
            engine = std::make_unique<ONNXEngine>();
            model_extension = ".onnx";
#else
            throw std::runtime_error("ONNX Runtime support is not enabled in this build.");
#endif
        } else if (engine_type == "tensorrt") {
#ifdef ENABLE_TRT
            engine = std::make_unique<TRTEngine>();
            model_extension = ".engine";
#else
            throw std::runtime_error("TensorRT support is not enabled in this build.");
#endif
        } else {
            throw std::runtime_error("Unknown ENGINE: " + engine_type + " (use pytorch, onnx, or tensorrt).");
        }

        const std::string model_path = model_base + "_" + std::to_string(image_size) + "_" + precision + model_extension;
        std::cout << "====================================================\n"
                  << "Phase 2 - COCO real-image native benchmark\n"
                  << "====================================================\n"
                  << "GPU:        " << gpu_name << "\n"
                  << "Engine:     " << engine_type << "\n"
                  << "Model:      " << model_input << "\n"
                  << "Artifact:   " << model_path << "\n"
                  << "Precision:  " << precision << "\n"
                  << "Resolution: " << image_size << "x" << image_size << "\n"
                  << "Warmup:     " << warmup << " images\n"
                  << "CUDA:       " << cuda_version << "\n"
                  << "Images:     " << images_dir << "\n"
                  << "====================================================\n\n";

        engine->load(model_path, image_size, precision);
        const auto image_paths = collect_images(images_dir);
        if (image_paths.empty()) throw std::runtime_error("No COCO images found in " + images_dir);
        const int total = static_cast<int>(image_paths.size());
        const size_t input_elements = 3ULL * image_size * image_size;

        std::cout << "Found " << total << " images.\nWarming up (" << warmup << ")...\n";
        int warmed = 0;
        for (int i = 0; i < warmup && i < total; ++i) {
            cv::Mat image = cv::imread(image_paths[static_cast<size_t>(i)], cv::IMREAD_COLOR);
            if (image.empty()) continue;
            cv::Mat padded = letterbox_bgr(image, image_size);
            cv::Mat blob = preprocess_to_nchw(padded, image_size);
            (void)engine->infer(blob.ptr<float>(), input_elements);
            ++warmed;
        }
        cuda_check(cudaDeviceSynchronize(), "synchronize after warmup");
        std::cout << "Warmup complete (" << warmed << " successful images).\n\n";

        GPUMonitor gpu_monitor;
        gpu_monitor.start();

        std::vector<double> image_load_ms_values, preprocess_ms_values;
        std::vector<double> input_prepare_ms_values, inference_ms_values;
        image_load_ms_values.reserve(total);
        preprocess_ms_values.reserve(total);
        input_prepare_ms_values.reserve(total);
        inference_ms_values.reserve(total);
        int processed = 0;
        int skipped = 0;

        for (int idx = 0; idx < total; ++idx) {
            const auto load_start = Clock::now();
            cv::Mat image = cv::imread(image_paths[static_cast<size_t>(idx)], cv::IMREAD_COLOR);
            const double load_ms = elapsed_ms(load_start, Clock::now());
            if (image.empty()) { ++skipped; continue; }

            const auto preprocess_start = Clock::now();
            cv::Mat padded = letterbox_bgr(image, image_size);
            cv::Mat blob = preprocess_to_nchw(padded, image_size);
            const double preprocess_ms = elapsed_ms(preprocess_start, Clock::now());

            const StageTiming timing = engine->infer(blob.ptr<float>(), input_elements);
            image_load_ms_values.push_back(load_ms);
            preprocess_ms_values.push_back(preprocess_ms);
            input_prepare_ms_values.push_back(timing.input_prepare_ms);
            inference_ms_values.push_back(timing.inference_ms);
            ++processed;

            if ((idx + 1) % 500 == 0 || idx + 1 == total) {
                std::cout << "[" << (idx + 1) << "/" << total << "] "
                          << std::fixed << std::setprecision(2)
                          << "load=" << load_ms << " ms, preprocess=" << preprocess_ms
                          << " ms, input_prepare=" << timing.input_prepare_ms
                          << " ms, inference=" << timing.inference_ms << " ms\n";
            }
        }

        gpu_monitor.stop();
        const GPUMetrics gpu_metrics = gpu_monitor.get_results();
        const double image_load_mean = calc_mean(image_load_ms_values);
        const double preprocess_mean = calc_mean(preprocess_ms_values);
        const double input_prepare_mean = calc_mean(input_prepare_ms_values);
        const double inference_mean = calc_mean(inference_ms_values);
        const double inference_std = calc_std(inference_ms_values, inference_mean);
        const double p95 = calc_percentile(inference_ms_values, 95.0);
        const double p99 = calc_percentile(inference_ms_values, 99.0);
        const double total_latency = image_load_mean + preprocess_mean + input_prepare_mean + inference_mean;
        const double fps_e2e = total_latency > 0.0 ? 1000.0 / total_latency : 0.0;
        const double fps_per_watt = gpu_metrics.avg_power_W > 0.0
            ? fps_e2e / gpu_metrics.avg_power_W : 0.0;

        std::cout << "\n====================================================\n"
                  << "PHASE 2 NATIVE RESULT SUMMARY\n"
                  << "====================================================\n"
                  << "Images processed        : " << processed << " / " << total << " (skipped=" << skipped << ")\n"
                  << std::fixed << std::setprecision(3)
                  << "Image load mean (ms)    : " << image_load_mean << "\n"
                  << "CPU preprocess mean (ms): " << preprocess_mean << "\n"
                  << "Input prepare mean (ms) : " << input_prepare_mean << "\n"
                  << "Inference mean (ms)     : " << inference_mean << " +/- " << inference_std << "\n"
                  << "Native total (ms)       : " << total_latency << " (no NMS; includes file loading)\n"
                  << std::setprecision(2)
                  << "Native pipeline FPS     : " << fps_e2e << "\n"
                  << std::setprecision(3)
                  << "P95 inference (ms)      : " << p95 << "\n"
                  << "P99 inference (ms)      : " << p99 << "\n"
                  << "Average power (W)        : " << gpu_metrics.avg_power_W << "\n"
                  << std::setprecision(4)
                  << "FPS / Watt              : " << fps_per_watt << "\n"
                  << "====================================================\n";

        fs::create_directories(output_dir);
        std::string gpu_file = gpu_name;
        std::replace(gpu_file.begin(), gpu_file.end(), ' ', '_');
        const std::string file_base = gpu_file + "_" + model_base + "_" + engine_type + "_" +
                                      precision + "_" + std::to_string(image_size) + "_native";
        const std::string csv_path = (fs::path(output_dir) / (file_base + ".csv")).string();

        std::ofstream csv(csv_path);
        if (!csv) throw std::runtime_error("Cannot open CSV output: " + csv_path);
        csv << "gpu,model,resolution,precision,engine,images_processed,images_skipped,"
            << "image_load_ms,preprocess_ms,input_prepare_ms,inference_ms,inference_std_ms,"
            << "total_latency_ms,fps_e2e,p95_latency_ms,p99_latency_ms,"
            << "python,torch,cuda,avg_power_W,max_power_W,gpu_util_mean_pct,gpu_util_max_pct,"
            << "gpu_memory_mean_MB,gpu_memory_max_MB,fps_per_watt,measurement_scope\n";
        csv << '"' << gpu_name << "\","
            << model_input << ',' << image_size << ',' << precision << ',' << engine_type << ','
            << processed << ',' << skipped << ','
            << std::setprecision(10) << image_load_mean << ',' << preprocess_mean << ','
            << input_prepare_mean << ',' << inference_mean << ',' << inference_std << ','
            << total_latency << ',' << fps_e2e << ',' << p95 << ',' << p99 << ','
            << "native_cpp," << (engine_type == "pytorch" ? "libtorch" : "n/a") << ',' << cuda_version << ','
            << gpu_metrics.avg_power_W << ',' << gpu_metrics.max_power_W << ','
            << gpu_metrics.gpu_util_mean_pct << ',' << gpu_metrics.gpu_util_max_pct << ','
            << gpu_metrics.gpu_memory_mean_MB << ',' << gpu_metrics.gpu_memory_max_MB << ','
            << fps_per_watt << ','
            << '"' << "inference excludes native input preparation; total includes file load and excludes NMS" << '"' << '\n';
        csv.close();
        std::cout << "Saved results to: " << csv_path << std::endl;
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "[ERROR] " << exception.what() << std::endl;
        return 1;
    }
}

