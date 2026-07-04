#include "../detail/tensor_utils.hpp"

#include <felix/felix.hpp>
namespace sion {
namespace {

torch::Tensor sgemm_impl(const torch::Tensor &A, const torch::Tensor &B,
                         float alpha, float beta,
                         const std::string *kernel_name) {
  detail::check_same_cuda_device(A, B, "A", "B");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

  const int64_t m = A.size(0);
  const int64_t k = A.size(1);
  const int64_t n = B.size(1);

  TORCH_CHECK(B.size(0) == k, "B.size(0) must match A.size(1)");

  uint32_t M = detail::checked_u32(m, "M");
  uint32_t K = detail::checked_u32(k, "K");
  uint32_t N = detail::checked_u32(n, "N");

  auto C = beta == 0.0f ? torch::empty({m, n}, A.options())
                        : torch::zeros({m, n}, A.options());

  const float *ptrA = A.data_ptr<float>();
  const float *ptrB = B.data_ptr<float>();
  float *ptrC = C.data_ptr<float>();

  const at::cuda::OptionalCUDAGuard device_guard(A.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto status =
      kernel_name == nullptr
          ? felix::sgemm_f32_launch(M, N, K, alpha, ptrA, ptrB, beta, ptrC,
                                    stream)
          : felix::sgemm_f32_launch_by_name(M, N, K, alpha, ptrA, ptrB, beta,
                                            ptrC, stream, *kernel_name);

  TORCH_CHECK(status.ok(), "SGEMM launch failed: ", status.str());

  return C;
}

} // namespace

torch::Tensor sgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  return sgemm_impl(A, B, alpha, beta, nullptr);
}

torch::Tensor sgemm_by_name(const torch::Tensor &A, const torch::Tensor &B,
                            float alpha, float beta,
                            const std::string &kernel_name) {
  return sgemm_impl(A, B, alpha, beta, &kernel_name);
}
} // namespace sion
