target_sources(felix PRIVATE
    launchers/sm80_sgemm_f32_nn_m64n64k8_basic.cu
    launchers/sm80_sgemm_f32_nn_m64n64k8_cute.cu
    launchers/sm80_sgemm_f32_nn_m64n64k8_cute_swizzle.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling_db.cu
    launchers/sm80_sgemm_f32_nn_m128n128k16_a1b0.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5.cu
)
