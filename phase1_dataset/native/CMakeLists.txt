cmake_minimum_required(VERSION 3.18)
project(phase1_native_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Bake runtime paths into the executable so it finds libtorch, etc.
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Find OpenCV & CUDA Toolkit
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

option(ENABLE_TORCH "Enable PyTorch (LibTorch) Engine" ON)
option(ENABLE_ONNX  "Enable ONNX Runtime Engine" ON)
option(ENABLE_TRT   "Enable TensorRT Engine" ON)

add_executable(benchmark_map benchmark_map.cpp)

target_link_libraries(benchmark_map PRIVATE
    ${OpenCV_LIBS}
    CUDA::cuda_driver
    CUDA::cudart
)

# Desktop / x86_64: use NVML for accurate power readings
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    target_link_libraries(benchmark_map PRIVATE CUDA::nvml)
    target_compile_definitions(benchmark_map PRIVATE USE_NVML)
endif()

if(ENABLE_TORCH)
    find_package(Torch REQUIRED)
    target_compile_definitions(benchmark_map PRIVATE ENABLE_TORCH)
    target_link_libraries(benchmark_map PRIVATE ${TORCH_LIBRARIES})
endif()

if(ENABLE_ONNX)
    target_include_directories(benchmark_map PRIVATE /opt/onnxruntime/include)
    target_link_directories(benchmark_map PRIVATE /opt/onnxruntime/lib)
    target_compile_definitions(benchmark_map PRIVATE ENABLE_ONNX)
    target_link_libraries(benchmark_map PRIVATE onnxruntime)
endif()

if(ENABLE_TRT)
    target_compile_definitions(benchmark_map PRIVATE ENABLE_TRT)
    target_link_libraries(benchmark_map PRIVATE nvinfer)
endif()
