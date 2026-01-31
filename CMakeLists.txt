cmake_minimum_required(VERSION 4.0)
project(sion LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(Torch_DIR "/home/aoi211/libtorch/share/cmake/Torch/")

add_library(sion SHARED)

find_package(Torch REQUIRED)
message(STATUS "Torch libraries found: ${TORCH_LIBRARIES}")
target_link_libraries(sion PRIVATE "${TORCH_LIBRARIES}")

add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(src)

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)
if(BUILD_PYTHON_BINDING)
    add_subdirectory(python)
endif()