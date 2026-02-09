#include <torch/torch.h>
#include "../../common.hpp"
#include "ampere_sgemm_64x64.cuh"
namespace sion {

torch::Tensor sgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
  TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

  int M = (int)A.size(0);
  int K = (int)A.size(1);
  int N = (int)B.size(1);
  TORCH_CHECK(B.size(0) == K, "B.size(0) must match A.size(1)");
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 8;
  auto C = torch::empty({M, N}, A.options());
  const float *ptrA = A.data_ptr<float>();
  const float *ptrB = B.data_ptr<float>();
  float *ptrC = C.data_ptr<float>();

  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(64);

  cuda_check(cudaGetLastError(), "CUDA pre-launch error");

  ampere_sgemm_64x64<<<grid, block>>>(M, N, K, alpha, ptrA, ptrB, beta, ptrC);

  cuda_check(cudaGetLastError(), "Kernel launch failed");

  cuda_check(cudaDeviceSynchronize(),
             "Kernel execution failed (runtime error)");

  cuda_check(cudaGetLastError(), "CUDA post-sync error");

  return C;
}
} // namespace sion