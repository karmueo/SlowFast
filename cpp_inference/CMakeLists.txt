cmake_minimum_required(VERSION 3.10)
project(x3d_inference)

set(CMAKE_CXX_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find TensorRT
# Assuming TensorRT is installed in /usr/src/tensorrt or standard system paths
# You might need to adjust these paths based on your environment
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include /usr/local/include /usr/src/tensorrt/include)
find_library(TENSORRT_LIBRARY nvinfer HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/src/tensorrt/lib)

include_directories(${TENSORRT_INCLUDE_DIR})

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Source files
set(SOURCES main.cpp)

# Executable
add_executable(x3d_inference ${SOURCES})

# Link libraries
target_link_libraries(x3d_inference
    ${OpenCV_LIBS}
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
)
