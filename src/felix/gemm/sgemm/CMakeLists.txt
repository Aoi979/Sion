target_sources(felix PRIVATE
    ampere_sgemm_64x64_nn_kernel_launch.cu
    cute_sgemm_128x128_nn_kernel_launch.cu
    cute_sgemm_128x128_nn_wt_kernel_launch.cu
    cute_sgemm_128x128_nn_wt_db_kernel_launch.cu
    ampere_sgemm_128x128_nn_a1b0_kernel_launch.cu
    sm80_sgemm_stage5_kernel_launch.cu
    cute_sgemm_64x64_nn_kernel_launch.cu
    cute_sgemm_64x64_nn_swizzle_kernel_launch.cu
)
