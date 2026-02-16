add_library(felix SHARED)

find_package(CUDAToolkit REQUIRED)

list(GET CUDAToolkit_INCLUDE_DIRS 0 CUDA_INCLUDE)

list(GET CUDAToolkit_INCLUDE_DIRS 1 CUDA_CCCl_INCLUDE)

target_sources(felix PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_SOURCE_DIR}/include 
              ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
              ${CUDA_INCLUDE}
              ${CUDA_CUDNN_INCLUDE}
    FILES
    ${CMAKE_SOURCE_DIR}/include/felix/felix.hpp
    ${CMAKE_SOURCE_DIR}/include/felix/status.hpp
)


target_compile_options(felix PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
target_include_directories(felix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(base)
add_subdirectory(gemm)
add_subdirectory(flash_attention)

install(TARGETS felix
    EXPORT FelixTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # Windows 可用
    FILE_SET HEADERS
)

install(EXPORT FelixTargets
    NAMESPACE Felix::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Felix
)

include(CMakePackageConfigHelpers)


configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/FelixConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/FelixConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Felix
)


write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/FelixConfigVersion.cmake
    VERSION 0.1.0
    COMPATIBILITY AnyNewerVersion
)


install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/FelixConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/FelixConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Felix
)
