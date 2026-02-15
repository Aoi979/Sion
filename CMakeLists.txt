cmake_minimum_required(VERSION 4.0)
project(sion LANGUAGES CXX CUDA)
include(GNUInstallDirs)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(sion SHARED)

target_sources(sion PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include 
              ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass/include
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/sion/sion.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/felix/felix.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/felix/status.hpp
)

option(SION_ENABLE_TORCH_BINDINGS
    "Enable PyTorch bindings"
    ON)

if(SION_ENABLE_TORCH_BINDINGS)
    find_package(Torch REQUIRED)
    target_link_libraries(sion PUBLIC torch)
endif()

option(BUILD_PYTHON_BINDING "Build Python binding" OFF)

if(BUILD_PYTHON_BINDING)
    find_package(Python3 COMPONENTS Development REQUIRED)
    add_subdirectory(python)
endif()

add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(src)

install(TARGETS sion
    EXPORT SionTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS
)

install(EXPORT SionTargets
    NAMESPACE Sion::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SionConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/SionConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Sion
)

