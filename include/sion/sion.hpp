#include <ATen/core/TensorBody.h>
#include <torch/torch.h>
namespace sion {

    torch::Tensor flash_attention(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value);
    torch::Tensor sgemm(const torch::Tensor& A, const torch::Tensor& B, float alpha, float beta);
    torch::Tensor gemm(const torch::Tensor& A, const torch::Tensor& B, float alpha, float beta);

}