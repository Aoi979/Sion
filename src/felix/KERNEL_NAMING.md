# Felix Kernel Naming

Internal kernel registry names use:

```text
<arch>_<op>_<dtype>_<layout-or-shape>_<tile>_<variant>
```

Examples:

```text
sm80_sgemm_f32_nn_m128n128k8_stage5
sm80_hgemm_f16_nn_m128n128k64_fp32acc
sm90_hgemm_f16_nn_m128n128k64_pingpong
sm80_flash_attn_f16_hd64_bq64_bk64_mma16168_s2_1d
cuda_topk_f32_radix_select
```

Rules:

- Public PyTorch operator names stay short: `sion::gemm`, `sion::hgemm`,
  `sion::sgemm`, `sion::flash_attention`.
- Registry names describe implementation choices and are for internal
  dispatch, benchmarks, and debug paths.
- Names must be unique at registry level. If one kernel body has multiple typed
  specializations, encode the differentiating semantic shape in the registry
  name, for example FlashAttention `hd64` and `hd128`.
- `launchers/` contains `.cu` files that register and launch kernels. When one
  launcher file registers one implementation, its stem should match the
  registry name. When it registers several typed specializations, use the
  shared launcher stem and keep each registered name unique.
- `kernels/` contains `.cuh` files with CUDA kernel bodies. A one-launcher
  kernel should use the launcher stem. Shared kernel bodies should use the
  shared stem without layout when the same body serves multiple layouts, for
  example `sm80_hgemm_f16_m128n128k64_cute_mma16816.cuh`.
- `detail/` contains reusable device helpers, traits, schedulers, barriers,
  swizzles, and epilogue/mainloop building blocks. Files in `detail/` should
  not register kernels or expose Felix launch functions.
- Top-level `__global__` kernel symbols should end in `_kernel` and use the same
  stem as their `.cuh` file.
