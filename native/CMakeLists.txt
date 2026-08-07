cmake_minimum_required(VERSION 3.18)
project(native_yolo_benchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure runtime paths (like python site-packages for torch) are baked into the executable
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Find OpenCV & CUDA Toolkit
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

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
    target_include_directories(benchmark_native PRIVATE /opt/onnxruntime/include)
    target_link_directories(benchmark_native PRIVATE /opt/onnxruntime/lib)
    target_compile_definitions(benchmark_native PRIVATE ENABLE_ONNX)
    target_link_libraries(benchmark_native PRIVATE onnxruntime)
endif()

if(ENABLE_TRT)
    target_compile_definitions(benchmark_native PRIVATE ENABLE_TRT)
    target_link_libraries(benchmark_native PRIVATE nvinfer)
endif()
