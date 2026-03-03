add_subdirectory(sgemm)
add_subdirectory(hgemm)
add_subdirectory(flash_attention)

add_custom_target(bench)
add_dependencies(bench
    sgemm_bench
    cublas_bench
    hgemm_bench
    cublas_hgemm_bench
    fa_bench
)
