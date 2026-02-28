target_sources(felix PRIVATE
    hgemm.cu
    cute_hgemm_128x128_nn_kernel_launch.cu
    cute_hgemm_128x128_nt_kernel_launch.cu
)