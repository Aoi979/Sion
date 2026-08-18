target_sources(cuda_ops_core PRIVATE
    launchers/sm80_sgemm_f32_nn_m64n64k8_basic.cu
    launchers/sm80_sgemm_f32_nn_m64n64k8_cute.cu
    launchers/sm80_sgemm_f32_nn_m64n64k8_cute_swizzle.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling_db.cu
    launchers/sm80_sgemm_f32_nn_m128n128k16_a1b0.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule.cu
    launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order.cu
)
