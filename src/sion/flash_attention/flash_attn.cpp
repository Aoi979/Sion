#include "../common.hpp"
#include <cstdint>
#include <felix/felix.hpp>

namespace sion {

namespace detail {
template <int HEAD_DIM>
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

  auto batch_size = Q.size(0);
  auto heads = Q.size(1);
  auto QKV_seqlen = Q.size(2);
  // TODO: support tail tiles when seq_len is not divisible by Br (Br=64).
  TORCH_CHECK((QKV_seqlen % 64) == 0,
              "flash_attention: seq_len must be divisible by 64 for the "
              "current kernel; tail handling is not implemented yet");
  auto status = felix::ampere_flash_attn_launch<HEAD_DIM, 64>(
      dQ, dK, dV, dO, heads, batch_size, QKV_seqlen, stream,
      "ampere_flash_attn_mma16168_64_1D_warp_tiling");

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
    detail::launch_flash_attn_mma_stages_1D<64>(query, key, value, O);
    break;
  case 128:
    detail::launch_flash_attn_mma_stages_1D<128>(query, key, value, O);
    break;
  default:
    TORCH_CHECK(false, "flash_attention: unsupported head dimension ", D);
  }

  return O;
}

} // namespace sion
