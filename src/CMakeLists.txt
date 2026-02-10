add_subdirectory(flash_attention)
add_subdirectory(gemm)
add_subdirectory(base)
target_compile_options(sion PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)
target_include_directories(sion PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_include_directories(sion PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)