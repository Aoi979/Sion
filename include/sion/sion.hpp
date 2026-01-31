#include <torch/torch.h>
namespace sion {

    torch::Tensor flash_attention(torch::Tensor &query, torch::Tensor &key, torch::Tensor &value);
    torch::Tensor sgemm(const torch::Tensor& A, const torch::Tensor& B, float alpha, float beta);

}