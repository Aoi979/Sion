# CudaOpsCore Kernel Naming

Internal kernel registry names use:

```text
<arch>_<op>_<dtype>_<layout-or-shape>_<tile>_<variant>
```

Examples:

```text
sm80_sgemm_f32_nn_m128n128k8_stage5
sm80_GA10x_sgemm_f32_nn_m128n128k8_stage5
sm80_hgemm_f16_nn_m128n128k64_fp32acc
sm90_hgemm_f16_nn_m128n128k64_pingpong
sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2
cuda_topk_f32_radix_select
```

Rules:

- `src/cuda_ops/` is the PyTorch integration layer. Tensor validation and ATen
  wrappers live in `src/cuda_ops/operators/`; dispatcher schema and backend
  registration live in `src/cuda_ops/torch/`; shared wrapper helpers live in
  `src/cuda_ops/detail/`.
- `src/cuda_ops_core/runtime/` owns the low-level launch API, registry, dispatch
  selection, and status handling. It should not contain CUDA kernel bodies.
- Public PyTorch operator names stay short: `cuda_ops::gemm`, `cuda_ops::hgemm`,
  `cuda_ops::sgemm`, `cuda_ops::flash_attention`.
- Registry names describe implementation choices and are for internal
  dispatch, benchmarks, and debug paths.
- Names must be unique at registry level. If one algorithm has multiple typed
  specializations, encode the differentiating semantic shape in the registry
  name, for example FlashAttention `hd64` and `hd128`.
- `launchers/` contains `.cu` files that register and launch kernels. Default to
  one registered implementation per `.cu`; the file stem should match the
  registry name.
- `kernels/` contains `.cuh` files with CUDA kernel bodies. Default to one
  top-level `__global__` kernel per `.cuh`; the file stem should match the
  registry name without the launcher suffix.
- Non-CUDA backend artifacts are kept under `kernels/non_cuda/<backend>/`.
  These files are retained for reference or future porting only: they must not
  be added to a CMake source list, public header set, launcher, registry, or
  API until that backend is explicitly supported.
- `detail/` contains reusable device helpers, traits, schedulers, barriers,
  swizzles, shared launch glue, and epilogue/mainloop building blocks. Files in
  `detail/` should not register kernels.
- Top-level `__global__` kernel symbols should end in `_kernel` and use the same
  stem as their `.cuh` file.
- The `smXX` prefix is a physical compute-capability specialization, not a
  generic performance label. In particular, `sm80_GA10x` denotes the physical
  4060-class path tuned for runtime CC 8.9; it must remain separate from the
  plain `sm80` A100-class path because their FP32 throughput balance is
  different.
