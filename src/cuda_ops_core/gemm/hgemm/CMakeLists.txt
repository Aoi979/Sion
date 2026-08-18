target_sources(cuda_ops_core PRIVATE
    launchers/sm80_hgemm_f16_nn_m128n128k64_cute_mma16816.cu
    launchers/sm80_hgemm_f16_nt_m128n128k64_cute_mma16816.cu
    launchers/sm80_hgemm_f16_nn_m128n128k64_fp16acc.cu
    launchers/sm80_hgemm_f16_nn_m128n128k64_fp32acc.cu
    launchers/sm80_hgemm_f16_nn_m128n256k64_fp16acc.cu
    launchers/sm80_hgemm_f16_nn_m128n256k64_fp32acc.cu
    launchers/sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc.cu
    launchers/sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc.cu
    launchers/sm80_hgemm_f16_nn_m128n256k64_streamk_fp32acc.cu
)

if(CUDA_OPS_BUILD_SM90_KERNELS)
    add_library(cuda_ops_core_sm90_hgemm OBJECT
        launchers/sm90_hgemm_f16_nn_m128n128k64_pingpong.cu
        launchers/sm90_hgemm_f16_nn_m128n256k64_cooperative.cu
    )
    set_target_properties(cuda_ops_core_sm90_hgemm PROPERTIES
        CUDA_ARCHITECTURES 90a
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(cuda_ops_core_sm90_hgemm PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
        ${CUDA_INCLUDE}
        ${CMAKE_SOURCE_DIR}/src/cuda_ops_core
    )
    target_compile_options(cuda_ops_core_sm90_hgemm PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    )
    target_sources(cuda_ops_core PRIVATE $<TARGET_OBJECTS:cuda_ops_core_sm90_hgemm>)
endif()
