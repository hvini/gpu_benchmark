cmake_minimum_required(VERSION 3.18)
project(native_yolo_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# CUDA Toolkit & NVML
find_package(CUDAToolkit REQUIRED)

# Engine Feature Flags
option(ENABLE_TORCH "Enable PyTorch (LibTorch) Engine" ON)
option(ENABLE_ONNX  "Enable ONNX Runtime Engine" ON)
option(ENABLE_TRT   "Enable TensorRT Engine" ON)

add_executable(benchmark_native benchmark.cpp)

target_link_libraries(benchmark_native PRIVATE 
    ${OpenCV_LIBS} 
    CUDA::cuda_driver 
    CUDA::cudart 
    CUDA::nvml
)

if(ENABLE_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(benchmark_native PRIVATE ENABLE_TORCH)
    target_link_libraries(benchmark_native PRIVATE ${TORCH_LIBRARIES})
endif()

if(ENABLE_ONNX)
    # Explicitly include the ONNX Runtime GPU directories
    target_include_directories(benchmark_native PRIVATE /opt/onnxruntime-linux-x64-gpu-1.19.2/include)
    target_link_directories(benchmark_native PRIVATE /opt/onnxruntime-linux-x64-gpu-1.19.2/lib)
    target_compile_definitions(benchmark_native PRIVATE ENABLE_ONNX)
    target_link_libraries(benchmark_native PRIVATE onnxruntime)
endif()

if(ENABLE_TRT)
    target_compile_definitions(benchmark_native PRIVATE ENABLE_TRT)
    target_link_libraries(benchmark_native PRIVATE nvinfer)
endif()