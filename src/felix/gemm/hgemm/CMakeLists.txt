target_sources(felix PRIVATE
    hgemm.cu
    cute_hgemm_128x128_nn_kernel_launch.cu
    cute_hgemm_128x128_nt_kernel_launch.cu
    sm80_hgemm_128x128_kernel_launch.cu
)

if(SION_BUILD_SM90_KERNELS)
    add_library(felix_sm90_hgemm OBJECT
        sm90_hgemm_kernel_launch.cu
        sm90_hgemm_coop_kernel_launch.cu
    )
    set_target_properties(felix_sm90_hgemm PROPERTIES
        CUDA_ARCHITECTURES 90a
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(felix_sm90_hgemm PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
        ${CUDA_INCLUDE}
        ${CMAKE_SOURCE_DIR}/src/felix
    )
    target_compile_options(felix_sm90_hgemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )
    target_sources(felix PRIVATE $<TARGET_OBJECTS:felix_sm90_hgemm>)
endif()
