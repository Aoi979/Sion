#include "sgemm.cuh"
#include <torch/torch.h>
#include "../common.hpp"
namespace sion {


torch::Tensor sgemm(const torch::Tensor& A, const torch::Tensor& B,
                    float alpha, float beta) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(1);
    TORCH_CHECK(B.size(0) == K, "B.size(0) must match A.size(1)");

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int bK = 16;

    TORCH_CHECK(M % BM == 0, "M must be multiple of 128");
    TORCH_CHECK(N % BN == 0, "N must be multiple of 128");
    TORCH_CHECK(K % bK == 0, "K must be multiple of 16");

    auto C = torch::empty({M, N}, A.options());

    const float* ptrA = A.data_ptr<float>();
    const float* ptrB = B.data_ptr<float>();
    float* ptrC = C.data_ptr<float>();

    dim3 grid(N / BN, M / BM);
    dim3 block(256);

    cuda_check(cudaGetLastError(), "CUDA pre-launch error");

    sgemm_kernel<<<grid, block>>>(
        M, N, K, alpha, ptrA, ptrB, beta, ptrC
    );

    cuda_check(cudaGetLastError(), "Kernel launch failed");

    cuda_check(cudaDeviceSynchronize(), "Kernel execution failed (runtime error)");

    cuda_check(cudaGetLastError(), "CUDA post-sync error");

    return C;
}
}