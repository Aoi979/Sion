#include <sion/sion.hpp>

namespace sion {
torch::Tensor gemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                   float beta, const std::string &kernel_name) {
                    
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");

    TORCH_CHECK(A.dtype() == B.dtype(),
                "A and B must have the same dtype");

    switch (A.scalar_type()) {
        case torch::kFloat32:
            return kernel_name.empty() ? sgemm(A, B, alpha, beta)
                                       : sgemm(A, B, alpha, beta, kernel_name);

        case torch::kFloat16:
            return kernel_name.empty() ? hgemm(A, B, alpha, beta)
                                       : hgemm(A, B, alpha, beta, kernel_name);

        default:
            TORCH_CHECK(false,
                        "Unsupported dtype for gemm: ",
                        A.dtype());
    }
}

}
