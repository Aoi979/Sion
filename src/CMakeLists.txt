add_subdirectory(flash_attention)
add_subdirectory(sgemm)
target_compile_options(sion PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
)