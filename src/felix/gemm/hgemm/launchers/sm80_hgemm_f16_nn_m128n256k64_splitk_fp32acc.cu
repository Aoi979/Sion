#include "../kernels/sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace felix {
namespace {

namespace splitk = detail::sm80_hgemm_128x256_splitk::n256_splitk;
using SplitKShape = detail::sm80_hgemm_128x256_splitk::shape_mnk_n256;

FelixStatus validate_splitk_request(uint32_t M, uint32_t N, uint32_t K,
                                    float alpha, float beta, half const *A,
                                    half const *B, half *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % SplitKShape::M != 0 || N % SplitKShape::N != 0 ||
      K % SplitKShape::K != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM m128n256k64_splitk_fp32acc requires non-null A/B/C and "
        "aligned M/N/K");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return FelixStatus{};
}

int choose_split_factor(uint32_t M, uint32_t N, uint32_t K,
                        int sm_count) {
  int const k_tiles = static_cast<int>(K / SplitKShape::K);
  uint64_t const output_tiles =
      static_cast<uint64_t>(M / SplitKShape::M) * (N / SplitKShape::N);
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
  return N / SplitKShape::N <= 16 ? 1 : 8;
}

FelixStatus convert_splitk_error(cudaError_t error) {
  if (error == cudaSuccess) {
    return {};
  }
  return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error);
}

FelixStatus allocate_partial_workspace(float **partial, size_t elements,
                                       cudaStream_t stream) {
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "sm80 HGEMM Split-K workspace size overflows");
  }
  cudaError_t error = cudaMallocAsync(
      reinterpret_cast<void **>(partial), elements * sizeof(float), stream);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error,
                             "failed to allocate Split-K workspace");
  }
  return {};
}

FelixStatus release_partial_workspace(float *partial, cudaStream_t stream,
                                      FelixStatus status) {
  if (partial == nullptr) {
    return status;
  }
  cudaError_t error = cudaFreeAsync(partial, stream);
  if (status.ok() && error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, error,
                             "failed to release Split-K workspace");
  }
  return status;
}

} // namespace

FelixStatus sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_splitk_request(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }

  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, error);
  }
  cudaDeviceProp properties{};
  error = cudaGetDeviceProperties(&properties, device);
  if (error != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR, error);
  }

  int const split_factor =
      choose_split_factor(M, N, K, properties.multiProcessorCount);
  uint64_t const workspace_elements =
      static_cast<uint64_t>(split_factor) * M * N;
  if (workspace_elements > std::numeric_limits<size_t>::max()) {
    return FelixStatus::make(FelixStatus::Type::API_ERROR,
                             cudaErrorInvalidValue,
                             "sm80 HGEMM Split-K workspace size overflows");
  }

  float *partial = nullptr;
  status = allocate_partial_workspace(
      &partial, static_cast<size_t>(workspace_elements), stream);
  if (!status.ok()) {
    return status;
  }

  error = splitk::launch_hgemm_128x256_splitk_fp32acc(
      A, B, partial, static_cast<int>(M), static_cast<int>(N),
      static_cast<int>(K), split_factor, choose_block_swizzle(N), stream);
  if (error == cudaSuccess) {
    error = splitk::launch_hgemm_128x256_splitk_reduce(
        partial, C, static_cast<int>(M), static_cast<int>(N), split_factor,
        stream);
  }
  return release_partial_workspace(partial, stream,
                                   convert_splitk_error(error));
}

} // namespace felix

REGISTER_KERNEL(
    sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc,
    felix::make_hgemm_kernel(
        "sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc",
        felix::sm80_hgemm_f16_nn_m128n256k64_splitk_fp32acc_launch, false,
        {.min_cc = 80,
         .max_cc = 89,
         .priority = 85,
         .required_dynamic_smem_bytes =
             felix::detail::sm80_hgemm_128x256_splitk::
                 n256_splitk::kSplitKSharedStorageBytes128x256,
         .required_threads_per_block =
             felix::detail::sm80_hgemm_128x256_splitk::n256_splitk::
                 kSplitKThreads},
        {.layout = felix::KernelLayout::NN,
         .align_m = 128,
         .align_n = 256,
         .align_k = 64,
         .requires_alpha_one_beta_zero = true}));
