cmake_minimum_required(VERSION 4.0)
project(sion LANGUAGES CXX CUDA)
include(GNUInstallDirs)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(src)

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)

if(BUILD_PYTHON_BINDING)
    find_package(Python3 COMPONENTS Development REQUIRED)
    add_subdirectory(python)
endif()

add_subdirectory(tests)
# add_subdirectory(bench)