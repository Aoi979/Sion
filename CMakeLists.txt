cmake_minimum_required(VERSION 4.0)
project(sion LANGUAGES CXX CUDA)

enable_testing()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(TORCH_ROOT "" CACHE PATH "Path to libtorch root directory (e.g., /home/user/libtorch)")

if(NOT TORCH_ROOT)
    message(FATAL_ERROR "Must specify libtorch root directory via -DTORCH_ROOT=/path/to/libtorch!")
endif()

list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT}/share/cmake/Torch)

add_library(sion SHARED)

find_package(Torch REQUIRED)
target_link_libraries(sion PUBLIC torch)
find_package(Python3 COMPONENTS Development REQUIRED)

include_directories(${Python3_INCLUDE_DIRS})
link_libraries(${Python3_LIBRARIES})

target_include_directories(sion
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)

if(BUILD_PYTHON_BINDING)
    add_subdirectory(python)
endif()

option(SION_ENABLE_TORCH_BINDINGS
    "Enable PyTorch bindings"
    ON)

add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(src)


install(TARGETS sion
    EXPORT SionTargets
    LIBRARY DESTINATION lib
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT SionTargets
    NAMESPACE Sion::
    DESTINATION lib/cmake/Sion
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake"
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake"
    DESTINATION lib/cmake/Sion
)
