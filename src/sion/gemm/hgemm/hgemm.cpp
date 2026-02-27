#include "../../common.hpp"
#include <felix/felix.hpp>
namespace sion {
torch::Tensor hgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");

  TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
  TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

  TORCH_CHECK(A.dim() == 2, "A must be 2D");
  TORCH_CHECK(B.dim() == 2, "B must be 2D");

  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t N = B.size(1);

  TORCH_CHECK(B.size(0) == K, "B.size(0) must match A.size(1)");

  auto C = torch::empty({M, N}, A.options());

  auto B_t = B.transpose(0, 1).contiguous();

  const __half *ptrA = reinterpret_cast<const __half *>(A.data_ptr<at::Half>());

  const __half *ptrB =
      reinterpret_cast<const __half *>(B_t.data_ptr<at::Half>());

  __half *ptrC = reinterpret_cast<__half *>(C.data_ptr<at::Half>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto status = felix::ampere_hgemm_launch(
      static_cast<uint32_t>(M), static_cast<uint32_t>(N),
      static_cast<uint32_t>(K), alpha, ptrA, ptrB, beta, ptrC, stream);

  TORCH_CHECK(status.ok(), "HGEMM launch failed: ", status.str());

  return C;
}
} // namespace sion
