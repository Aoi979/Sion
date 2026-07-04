#include "../detail/tensor_utils.hpp"

#include <felix/felix.hpp>

namespace sion {

namespace detail {
template <int HEAD_DIM>
void launch_flash_attn_sm80_v2(const torch::Tensor &Q, const torch::Tensor &K,
                               const torch::Tensor &V, torch::Tensor &O) {
  half *dQ = reinterpret_cast<half *>(Q.data_ptr<at::Half>());
  half *dK = reinterpret_cast<half *>(K.data_ptr<at::Half>());
  half *dV = reinterpret_cast<half *>(V.data_ptr<at::Half>());
  half *dO = reinterpret_cast<half *>(O.data_ptr<at::Half>());

  const at::cuda::OptionalCUDAGuard device_guard(Q.device());
  at::cuda::CUDAStream current_stream = at::cuda::getCurrentCUDAStream();
  cudaStream_t stream = current_stream.stream();

  auto batch_size = detail::checked_u32(Q.size(0), "batch_size");
  auto heads = detail::checked_u32(Q.size(1), "heads");
  auto QKV_seqlen = detail::checked_u32(Q.size(2), "seq_len");
  // TODO: support tail tiles when seq_len is not divisible by the active tile.
  TORCH_CHECK((QKV_seqlen % 128) == 0,
              "flash_attention: seq_len must be divisible by 128 for the "
              "current kernel; tail handling is not implemented yet");
  auto status = felix::flash_attn_f16_launch<HEAD_DIM, 64>(
      dQ, dK, dV, dO, heads, batch_size, QKV_seqlen, stream);

  TORCH_CHECK(status.ok(),
              "flash_attention: kernel launch failed: ", status.str());
}

} // namespace detail

torch::Tensor flash_attention(const torch::Tensor &query,
                              const torch::Tensor &key,
                              const torch::Tensor &value) {
  TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
              "query, key and value must be CUDA tensors");
  TORCH_CHECK(query.device() == key.device() &&
                  query.device() == value.device(),
              "query, key and value must be on the same CUDA device");
  TORCH_CHECK(query.dtype() == torch::kHalf && key.dtype() == torch::kHalf &&
                  value.dtype() == torch::kHalf,
              "query, key and value must be float16");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
              "query, key and value must be 4D tensors");
  TORCH_CHECK(query.sizes() == key.sizes() && query.sizes() == value.sizes(),
              "query, key and value must have the same shape");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
  TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

  auto B = query.size(0);
  auto H = query.size(1);
  auto N = query.size(2);
  auto D = query.size(3);
  auto opt = query.options();
  auto O = torch::empty({B, H, N, D}, opt);

  switch (D) {
  case 64:
    detail::launch_flash_attn_sm80_v2<64>(query, key, value, O);
    break;
  case 128:
    detail::launch_flash_attn_sm80_v2<128>(query, key, value, O);
    break;
  default:
    TORCH_CHECK(false, "flash_attention: unsupported head dimension ", D);
  }

  return O;
}

} // namespace sion
