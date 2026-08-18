target_sources(cuda_ops_core PRIVATE
    launchers/sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2.cu
    launchers/sm80_flash_attn_f16_hd128_bq128_bk64_mma16816_v2.cu
    launchers/sm80_flash_attn_f16_hd64_bq128_bk64_fav2_best_experimental.cu
    launchers/sm80_flash_attn_f16_hd128_bq128_bk64_fav2_best_experimental.cu
)
