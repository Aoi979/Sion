target_sources(sion PRIVATE
    flash_attn.cu
)
if (SION_ENABLE_TORCH_BINDINGS)
  target_sources(sion PRIVATE flash_attn.cpp)
endif()