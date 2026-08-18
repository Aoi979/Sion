add_library(cuda_ops_core SHARED)

find_package(CUDAToolkit REQUIRED)

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE)

list(GET CUDAToolkit_INCLUDE_DIRS 1 CUDA_CCCl_INCLUDE)

target_sources(cuda_ops_core PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include
              ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
              ${CUDA_INCLUDE}
              ${CUDA_CUDNN_INCLUDE}
    FILES
    ${CMAKE_SOURCE_DIR}/include/cuda_ops_core/core.hpp
    ${CMAKE_SOURCE_DIR}/include/cuda_ops_core/status.hpp
)


target_compile_options(cuda_ops_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
target_include_directories(cuda_ops_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(cuda_ops_core PUBLIC CUDA::cuda_driver)
target_link_options(cuda_ops_core PRIVATE -static-libstdc++ -static-libgcc)

add_subdirectory(runtime)
add_subdirectory(gemm)
add_subdirectory(flash_attention)
add_subdirectory(topk)

install(TARGETS cuda_ops_core
    EXPORT CudaOpsCoreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows 可用
    FILE_SET HEADERS
)

install(EXPORT CudaOpsCoreTargets
    NAMESPACE CudaOpsCore::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOpsCore
)

include(CMakePackageConfigHelpers)


configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/CudaOpsCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsCoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOpsCore
)


write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsCoreConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)


install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsCoreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CudaOpsCoreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/CudaOpsCore
)
