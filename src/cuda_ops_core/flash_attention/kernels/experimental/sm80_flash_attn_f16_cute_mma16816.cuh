#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/pointer_flagged.hpp"
#include "cute/tensor_impl.hpp"
#include "../../detail/softmax.cuh"
#include <cute/tensor.hpp>
#include <limits>

struct BlockInfo {

  template <typename Params>
  __device__ BlockInfo(const Params &params, const int /*bidb*/)
      : actual_seqlen_q(params.seqlen_q), actual_seqlen_k(params.seqlen_k) {}

  template <typename index_t>
  __forceinline__ __device__ index_t q_offset(index_t batch_stride,
                                              index_t /*row_stride*/,
                                              int bidb) const {
    return bidb * batch_stride;
  }

  template <typename index_t>
  __forceinline__ __device__ index_t k_offset(index_t batch_stride,
                                              index_t /*row_stride*/,
                                              int bidb) const {
    return bidb * batch_stride;
  }

  const int actual_seqlen_q;
  const int actual_seqlen_k;
};

template <typename Kernel_traits, typename Params, bool Is_causal>
inline __device__ void cute_flash_attn_mma16816(const Params &params,
                                                const int bidb, const int bidh,
                                                const int m_block) {
  using namespace cute;
  using namespace cuda_ops_core::flash_attn::utils;

  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;
  const BlockInfo binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q)
    return;

  const int n_block_min = 0;

  int n_block_max = ceil_div(binfo.actual_seqlen_k, kBlockN);
  if constexpr (Is_causal) {
    n_block_max = std::min(n_block_max, ceil_div((m_block + 1) * kBlockM +
                                                     binfo.actual_seqlen_k -
                                                     binfo.actual_seqlen_q,
                                                 kBlockN));
  }
  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) +
                                binfo.q_offset(params.q_batch_stride,
                                               params.q_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));

  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                         make_coord(m_block, 0)); // (kBlockM, kHeadDim)
  Tensor mK =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) +
                                binfo.k_offset(params.k_batch_stride,
                                               params.k_row_stride, bidb)),
                  make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                  make_stride(params.k_row_stride, params.k_head_stride, _1{}));

  Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                         Shape<Int<kBlockN>, Int<kHeadDim>>{},
                         make_coord(_, 0)); // (kBlockN, kHeadDim, nblocksN)
  
                         
}
