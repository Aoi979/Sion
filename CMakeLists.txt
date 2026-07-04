cmake_minimum_required(VERSION 4.0)
project(sion LANGUAGES CXX CUDA)
include(GNUInstallDirs)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(SION_CUDA_ARCHITECTURES "80;89" CACHE STRING
    "CUDA architectures for regular Sion kernels")
set(CMAKE_CUDA_ARCHITECTURES ${SION_CUDA_ARCHITECTURES})
option(SION_BUILD_SM90_KERNELS "Build Hopper SM90 HGEMM kernels" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)

if(BUILD_PYTHON_BINDING)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    add_subdirectory(python)
endif()

add_subdirectory(tests)
add_subdirectory(benchmarks)
