target_sources(sion PRIVATE
    gemm.cpp
)
add_subdirectory(sgemm)
add_subdirectory(hgemm)
