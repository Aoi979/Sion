#include <torch/torch.h>

namespace sion {
    torch::Tensor sgemm(const torch::Tensor& A, const torch::Tensor& B,
                    float alpha, float beta);

    torch::Tensor hgemm(const torch::Tensor& A, const torch::Tensor& B,
                    float alpha, float beta);

    torch::Tensor gemm(const torch::Tensor& A, const torch::Tensor& B,
                   float alpha, float beta) {
                    
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");

    TORCH_CHECK(A.dtype() == B.dtype(),
                "A and B must have the same dtype");

    switch (A.scalar_type()) {
        case torch::kFloat32:
            return sgemm(A, B, alpha, beta);

        case torch::kFloat16:
            return hgemm(A, B, alpha, beta);

        default:
            TORCH_CHECK(false,
                        "Unsupported dtype for gemm: ",
                        A.dtype());
    }
}

}