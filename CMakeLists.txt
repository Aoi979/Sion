cmake_minimum_required(VERSION 4.0)
project(cuda_ops LANGUAGES CXX CUDA)
include(GNUInstallDirs)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_OPS_CUDA_ARCHITECTURES "80;89" CACHE STRING
    "CUDA architectures for regular CudaOps kernels")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_OPS_CUDA_ARCHITECTURES})
option(CUDA_OPS_BUILD_SM90_KERNELS "Build Hopper SM90 HGEMM kernels" ON)
option(CUDA_OPS_BUILD_MACA
       "Build standalone MACA operators with mxcc"
       OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CUDA_OPS_BUILD_MACA)
    include("${CMAKE_SOURCE_DIR}/cmake/Maca.cmake")
endif()

add_subdirectory(src)

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)

if(BUILD_PYTHON_BINDING)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    add_subdirectory(python)
endif()

add_subdirectory(tests)
add_subdirectory(benchmarks)
