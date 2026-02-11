target_sources(sion PRIVATE
    sgemm.cu
)

if(SION_ENABLE_TORCH_BINDINGS)
    target_sources(sion PRIVATE sgemm.cpp)
endif()
