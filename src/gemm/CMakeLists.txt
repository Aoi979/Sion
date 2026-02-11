if(SION_ENABLE_TORCH_BINDINGS)
    target_sources(sion PRIVATE
        gemm.cpp
    )
endif()

add_subdirectory(sgemm)
add_subdirectory(hgemm)
