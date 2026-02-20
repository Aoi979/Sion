#include "../../common.hpp"
#include <felix/felix.hpp>
namespace sion {

torch::Tensor sgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

  uint32_t M = (uint32_t)A.size(0);
  uint32_t K = (uint32_t)A.size(1);
  uint32_t N = (uint32_t)B.size(1);

  TORCH_CHECK(B.size(0) == K, "B.size(0) must match A.size(1)");

  auto C = torch::empty({M, N}, A.options());

  const float *ptrA = A.data_ptr<float>();
  const float *ptrB = B.data_ptr<float>();
  float *ptrC = C.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto status = felix::ampere_sgemm_launch(M, N, K, alpha, ptrA, ptrB, beta,
                                           ptrC, stream);

  TORCH_CHECK(status.ok(), "SGEMM launch failed: ", status.str());

  return C;
}
} // namespace sion