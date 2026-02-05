#include <torch/torch.h>
#include "../common.hpp"
#include "ampere_hgemm_wmma_128x128.cuh"
namespace sion {
    torch::Tensor hgemm(const torch::Tensor& A,
                    const torch::Tensor& B,
                    float alpha,
                    float beta) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");

    int M = (int)A.size(0);
    int K = (int)A.size(1);
    int N = (int)B.size(1);
    TORCH_CHECK(B.size(0) == K, "B.size(0) must match A.size(1)");

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int bK = 32;

    TORCH_CHECK(M % BM == 0, "M must be multiple of 128");
    TORCH_CHECK(N % BN == 0, "N must be multiple of 128");
    TORCH_CHECK(K % bK == 0, "K must be multiple of 32");

    auto C = torch::empty({M, N}, A.options());

    const at::Half* A_half = A.data_ptr<at::Half>();
    const at::Half* B_half = B.data_ptr<at::Half>();
    at::Half* C_half = C.data_ptr<at::Half>();

    const __half* ptrA = reinterpret_cast<const __half*>(A_half);
    const __half* ptrB = reinterpret_cast<const __half*>(B_half);
    __half* ptrC = reinterpret_cast<__half*>(C_half);

    dim3 grid(N / BN, M / BM);
    dim3 block(128);

    cuda_check(cudaGetLastError(), "CUDA pre-launch error");

    cudaFuncSetAttribute(
        ampere_hgemm_wmma_128x128<>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        37888
    );

    ampere_hgemm_wmma_128x128<<<grid, block>>>(
        ptrA, ptrB, ptrC, M, N, K
    );

    cuda_check(cudaGetLastError(), "Kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "Kernel execution failed");
    cuda_check(cudaGetLastError(), "CUDA post-sync error");

    return C;
}


}