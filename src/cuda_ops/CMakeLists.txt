add_library(cuda_ops SHARED)

find_package(Torch REQUIRED)

target_sources(cuda_ops PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
    FILES
    ${CMAKE_SOURCE_DIR}/include/cuda_ops/cuda_ops.hpp
)

target_sources(cuda_ops PRIVATE
    operators/flash_attention.cpp
    operators/gemm.cpp
    operators/hgemm.cpp
    operators/sgemm.cpp
    torch/library.cpp
)

target_link_libraries(cuda_ops PUBLIC torch cuda_ops_core)
target_link_options(cuda_ops PRIVATE -static-libstdc++ -static-libgcc)

install(TARGETS cuda_ops
    EXPORT CudaOpsTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS
)

install(EXPORT CudaOpsTargets
    NAMESPACE CudaOps::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOps
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/CudaOpsConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOps
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOps
)
