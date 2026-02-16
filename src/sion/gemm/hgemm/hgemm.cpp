#include <torch/torch.h>
namespace sion {
torch::Tensor hgemm(const torch::Tensor &A, const torch::Tensor &B, float alpha,
                    float beta) {
  return torch::empty({0});
}
} // namespace sion
