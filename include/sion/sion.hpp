#include <torch/torch.h>
#include <string>

namespace sion {

torch::Tensor flash_attention(const torch::Tensor &query,
                              const torch::Tensor &key,
                              const torch::Tensor &value);
torch::Tensor sgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta,
                    const std::string &kernel_name = "cute_sgemm_64x64_nn");
torch::Tensor hgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta,
                    const std::string &kernel_name = "cute_hgemm_128x128_nn");
torch::Tensor hgemm_nt(const torch::Tensor &A, const torch::Tensor &B,
                       float alpha, float beta,
                       const std::string &kernel_name = "cute_hgemm_128x128_nt");
torch::Tensor gemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                   float beta, const std::string &kernel_name = "");

} // namespace sion
