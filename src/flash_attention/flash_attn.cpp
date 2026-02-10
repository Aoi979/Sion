#include "../common.hpp"
#include <felix/felix.hpp>

namespace sion {

namespace detail {
template <int HEAD_DIM, int STAGE>
void launch_flash_attn_mma_stages_1D(const torch::Tensor &Q,
                                     const torch::Tensor &K,
                                     const torch::Tensor &V, torch::Tensor &O) {
  TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && O.is_cuda(),
              "All tensors must be CUDA tensors");
  TORCH_CHECK(Q.dtype() == torch::kHalf && K.dtype() == torch::kHalf &&
                  V.dtype() == torch::kHalf && O.dtype() == torch::kHalf,
              "All tensors must be half");
  TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4 && O.dim() == 4,
              "All tensors must be 4D");

  half *dQ = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
  half *dK = reinterpret_cast<half *>(K.data_ptr<at::Half>());
  half *dV = reinterpret_cast<half *>(V.data_ptr<at::Half>());
  half *dO = reinterpret_cast<half *>(O.data_ptr<at::Half>());

  at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = current_stream.stream();

  auto status =
      felix::ampere_flash_attn_mma16168_64_1D_warp_tiling_kernel_launch<
          HEAD_DIM, STAGE>(dQ, dK, dV, dO, Q.size(1), Q.size(2), stream);

  TORCH_CHECK(status.ok(),
              "flash_attention: kernel launch failed: ", status.str());
}

} // namespace detail

torch::Tensor flash_attention(const torch::Tensor &query,
                              const torch::Tensor &key,
                              const torch::Tensor &value) {
  auto B = query.size(0);
  auto H = query.size(1);
  auto N = query.size(2);
  auto D = query.size(3);
  auto opt = query.options();
  auto O = torch::empty({B, H, N, D}, opt);

  switch (D) {
  case 64:
    detail::launch_flash_attn_mma_stages_1D<64, 2>(query, key, value, O);
    break;
  case 128:
    detail::launch_flash_attn_mma_stages_1D<128, 2>(query, key, value, O);
    break;
  default:
    TORCH_CHECK(false, "flash_attention: unsupported head dimension ", D);
  }

  return O;
}

} // namespace sion