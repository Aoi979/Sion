#pragma once
#include <cstdint>
#include "../utils/utils.h"

__global__ void v1_fwd_kernel_naive(float const *Q, float const *K,
                                    float const *V, int const target_seq_len,
                                    int const src_seq_len, int const d,
                                    int const Tc, int const Tr, int const Bc,
                                    int const Br, float const softmax_scale,
                                    float *l, float *m, float *O) {
  uint32_t kv_offset = (blockIdx.x * gridDim.y * src_seq_len * d) +
                       (blockIdx.y * src_seq_len * d);
  uint32_t q_offset = (blockIdx.x * gridDim.y * target_seq_len * d) +
                      (blockIdx.y * target_seq_len * d);

  uint32_t lm_offset =
      (blockIdx.x * gridDim.y * target_seq_len) + (blockIdx.y * target_seq_len);

  extern __shared__ float smem[];

  int32_t tile_size = Bc * d;
  float *Qi = smem;
  float *Kj = &smem[tile_size];
  float *Vj = &smem[tile_size * 2];
  float *S = &smem[tile_size * 3];

  for (uint32_t j = 0; j < Tc; j++) {
    for (uint32_t x = 0; x < d; x++) {
      Kj[threadIdx.x * d + x] =
          K[kv_offset + (tile_size * j) + (threadIdx.x * d) + x];
      Vj[threadIdx.x * d + x] =
          V[kv_offset + (tile_size * j) + (threadIdx.x * d) + x];
    }
    __syncthreads();
    for (uint32_t i = 0; i < Tr; i++) {
      if (threadIdx.x < Br) {
        for (uint32_t x = 0; x < d; x++) {
          Qi[threadIdx.x * d + x] =
              Q[q_offset + (tile_size * i) + (threadIdx.x * d) + x];
        }
        float row_m_prev = m[lm_offset + (Br * i) + threadIdx.x];
        float row_l_prev = l[lm_offset + (Br * i) + threadIdx.x];

        float row_m = -INFINITY;

        for (int y = 0; y < Bc; y++) {
          float sum = 0;
          for (uint32_t x = 0; x < d; x++) {
            sum += Qi[(threadIdx.x * d) + x] * Kj[(y * d) + x];
          }
          sum *= softmax_scale;
          S[(Bc * threadIdx.x) + y] = sum;

          if (sum > row_m) {
            row_m = sum;
          }
        }

        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
          S[(Bc * threadIdx.x) + y] = __expf(S[(Bc * threadIdx.x) + y] - row_m);
          row_l += S[(Bc * threadIdx.x) + y];
        }

        float row_m_new = max(row_m_prev, row_m);
        float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) +
                          (__expf(row_m - row_m_new) * row_l);

        for (int x = 0; x < d; x++) {
          float pv = 0;
          for (int y = 0; y < Bc; y++) {
            pv += S[(Bc * threadIdx.x) + y] * Vj[(y * d) + x];
          }
          O[q_offset + (tile_size * i) + (threadIdx.x * d) + x] =
              (1 / row_l_new) *
              ((row_l_prev * __expf(row_m_prev - row_m_new) *
                O[q_offset + (tile_size * i) + (threadIdx.x * d) + x]) +
               (__expf(row_m - row_m_new) * pv));
        }
        m[lm_offset + (Br * i) + threadIdx.x] = row_m_new;
        l[lm_offset + (Br * i) + threadIdx.x] = row_l_new;
      }
    }
    __syncthreads();
  }
}


