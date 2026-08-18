#include "../detail/tensor_utils.hpp"

#include <cuda_ops_core/core.hpp>
namespace cuda_ops {
namespace {

torch::Tensor hgemm_impl(const torch::Tensor &A, const torch::Tensor &B,
                         float alpha, float beta,
                         const std::string *kernel_name, bool nt_layout) {
  detail::check_same_cuda_device(A, B, "A", "B");
  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = nt_layout ? B.size(0) : B.size(1);

  if (nt_layout) {
    TORCH_CHECK(B.size(1) == k, "B.size(1) must match A.size(1) for NT GEMM");
  } else {
    TORCH_CHECK(B.size(0) == k, "B.size(0) must match A.size(1)");
  }

  uint32_t M = detail::checked_u32(m, "M");
  uint32_t K = detail::checked_u32(k, "K");
  uint32_t N = detail::checked_u32(n, "N");

  auto C = beta == 0.0f ? torch::empty({m, n}, A.options())
                        : torch::zeros({m, n}, A.options());

  const __half *ptrA = reinterpret_cast<const __half *>(A.data_ptr<at::Half>());
  const __half *ptrB = reinterpret_cast<const __half *>(B.data_ptr<at::Half>());
  __half *ptrC = reinterpret_cast<__half *>(C.data_ptr<at::Half>());

  const at::cuda::OptionalCUDAGuard device_guard(A.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cuda_ops_core::Status status;
  if (kernel_name != nullptr) {
    status = cuda_ops_core::hgemm_f16_launch_by_name(M, N, K, alpha, ptrA, ptrB, beta,
                                             ptrC, stream, *kernel_name);
  } else if (nt_layout) {
    status = cuda_ops_core::hgemm_f16_nt_launch(M, N, K, alpha, ptrA, ptrB, beta, ptrC,
                                        stream);
  } else {
    status =
        cuda_ops_core::hgemm_f16_launch(M, N, K, alpha, ptrA, ptrB, beta, ptrC, stream);
  }

  TORCH_CHECK(status.ok(), "HGEMM launch failed: ", status.str());

  return C;
}

} // namespace

torch::Tensor hgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  return hgemm_impl(A, B, alpha, beta, nullptr, false);
}

torch::Tensor hgemm_by_name(const torch::Tensor &A, const torch::Tensor &B,
                            float alpha, float beta,
                            const std::string &kernel_name) {
  return hgemm_impl(A, B, alpha, beta, &kernel_name, false);
}

torch::Tensor hgemm_nt(const torch::Tensor &A, const torch::Tensor &B,
                       float alpha, float beta) {
  return hgemm_impl(A, B, alpha, beta, nullptr, true);
}
} // namespace cuda_ops
