#include "../kernels/sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc.cuh"
#include <cuda_ops_core/core.hpp>
#include <cuda_ops_core/registry.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace cuda_ops_core {
namespace {

namespace atomic_splitk =
    detail::sm80::atomic_splitk;
using AtomicSplitKShape =
    detail::sm80::tile::shape_mnk_n256;

Status validate_atomic_splitk_request(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float beta,
    half const *A, half const *B, half *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % AtomicSplitKShape::M != 0 ||
      N % AtomicSplitKShape::N != 0 || K % AtomicSplitKShape::K != 0) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM m128n256k64_splitk_atomic_fp32acc requires non-null "
        "A/B/C and aligned M/N/K");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return Status{};
}

int choose_atomic_split_factor(uint32_t M, uint32_t N, uint32_t K,
                              int sm_count) {
  int const k_tiles = static_cast<int>(K / AtomicSplitKShape::K);
  uint64_t const output_tiles =
      static_cast<uint64_t>(M / AtomicSplitKShape::M) *
      (N / AtomicSplitKShape::N);
  uint64_t const target =
      (static_cast<uint64_t>(sm_count) + output_tiles - 1) / output_tiles;

  int split_factor = 1;
  for (int candidate = 2; candidate <= 64; candidate *= 2) {
    if (candidate > k_tiles || candidate > target || k_tiles % candidate != 0) {
      break;
    }
    split_factor = candidate;
  }
  return split_factor;
}

int choose_block_swizzle(uint32_t N) {
  return N / AtomicSplitKShape::N <= 16 ? 1 : 8;
}

Status convert_atomic_splitk_error(cudaError_t error) {
  if (error == cudaSuccess) {
    return {};
  }
  return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, error);
}

Status allocate_workspace(void **ptr, size_t bytes, cudaStream_t stream,
                               const char *description) {
  cudaError_t error = cudaMallocAsync(ptr, bytes, stream);
  if (error != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED, error,
                             std::string("failed to allocate ") +
                                 description);
  }
  return {};
}

Status release_workspace(float *accumulator, int *tile_turns,
                              cudaStream_t stream, Status status) {
  cudaError_t free_error = cudaSuccess;
  if (accumulator != nullptr) {
    free_error = cudaFreeAsync(accumulator, stream);
  }
  if (tile_turns != nullptr) {
    cudaError_t error = cudaFreeAsync(tile_turns, stream);
    if (free_error == cudaSuccess) {
      free_error = error;
    }
  }
  if (status.ok() && free_error != cudaSuccess) {
    return Status::make(Status::Type::KERNEL_LAUNCH_FAILED,
                             free_error,
                             "failed to release atomic Split-K workspace");
  }
  return status;
}

} // namespace

Status sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status =
      validate_atomic_splitk_request(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return Status::make(Status::Type::API_ERROR, error);
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return Status::make(Status::Type::API_ERROR, error);
  }

  int const split_factor = choose_atomic_split_factor(
      M, N, K, properties.multiProcessorCount);
  uint64_t const output_elements = static_cast<uint64_t>(M) * N;
  uint64_t const tile_count =
      static_cast<uint64_t>(M / AtomicSplitKShape::M) *
      (N / AtomicSplitKShape::N);
  if (output_elements > std::numeric_limits<size_t>::max() ||
      tile_count > std::numeric_limits<size_t>::max() ||
      output_elements >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      tile_count > std::numeric_limits<size_t>::max() / sizeof(int)) {
    return Status::make(
        Status::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM atomic Split-K workspace size overflows");
  }

  float *accumulator = nullptr;
  int *tile_turns = nullptr;
  status = allocate_workspace(
      reinterpret_cast<void **>(&accumulator),
      static_cast<size_t>(output_elements) * sizeof(float), stream,
      "atomic Split-K accumulator");
  if (!status.ok()) {
    return status;
  }
  status = allocate_workspace(
      reinterpret_cast<void **>(&tile_turns),
      static_cast<size_t>(tile_count) * sizeof(int), stream,
      "atomic Split-K tile turns");
  if (!status.ok()) {
    return release_workspace(accumulator, nullptr, stream, status);
  }

  error = cudaMemsetAsync(accumulator, 0,
                          static_cast<size_t>(output_elements) * sizeof(float),
                          stream);
  if (error == cudaSuccess) {
    error = cudaMemsetAsync(tile_turns, 0,
                            static_cast<size_t>(tile_count) * sizeof(int),
                            stream);
  }
  if (error == cudaSuccess) {
    error = atomic_splitk::launch_hgemm_128x256_splitk_atomic_fp32acc(
        A, B, accumulator, tile_turns, C, static_cast<int>(M),
        static_cast<int>(N), static_cast<int>(K), split_factor,
        choose_block_swizzle(N), stream);
  }

  return release_workspace(accumulator, tile_turns, stream,
                           convert_atomic_splitk_error(error));
}

} // namespace cuda_ops_core

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc,
    cuda_ops_core::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc",
        cuda_ops_core::sm80_hgemm_f16_nn_m128n256k64_splitk_atomic_fp32acc_launch,
        false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 80,
         .required_dynamic_smem_bytes =
             cuda_ops_core::detail::sm80::atomic_splitk::
                 kAtomicSplitKSharedStorageBytes128x256,
         .required_threads_per_block =
             cuda_ops_core::detail::sm80::atomic_splitk::kAtomicSplitKThreads},
        {.layout = cuda_ops_core::KernelLayout::NN,
         .align_m = 128,
         .align_n = 256,
         .align_k = 64,
         .requires_alpha_one_beta_zero = true}));
