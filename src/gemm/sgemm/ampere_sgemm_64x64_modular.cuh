#include <cstdint>
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer)                                            \
  (reinterpret_cast<const float4 *>(&(pointer))[0])


namespace felix {

struct Sgemm64x64Config {
  static constexpr uint32_t WARP_SIZE = 32;

  static constexpr uint32_t kBM = 64;
  static constexpr uint32_t kBN = 64;
  static constexpr uint32_t kBK = 8;
  static constexpr uint32_t kWM = 32;
  static constexpr uint32_t kWN = 64;
  static constexpr uint32_t kTM = 4;
  static constexpr uint32_t kTN = 4;
  static constexpr uint32_t kWM_ITER = 2;
  static constexpr uint32_t kWN_ITER = 2;
  static constexpr uint32_t kStages = 2;

  static constexpr uint32_t kThreads = 64;
  static constexpr uint32_t kWarps = kThreads / WARP_SIZE;

  static constexpr uint32_t load_smem_A_row_stride = kThreads / kBK;
  static constexpr uint32_t load_smem_B_row_stride = kThreads / kBN;
  static constexpr uint32_t load_a_iters = kBM / load_smem_A_row_stride;
  static constexpr uint32_t load_b_iters = kBK / load_smem_B_row_stride;

  static constexpr uint32_t warps_per_row = kBN / kWN;
  static constexpr uint32_t warpMMATileM = kWM / kWM_ITER;
  static constexpr uint32_t warpMMATileN = kWN / kWN_ITER;
};

template <typename Config>
struct ThreadInfo {
  uint32_t tid;
  uint32_t warp_id;
  uint32_t lane_id;
  uint32_t warp_row;
  uint32_t warp_col;
  uint32_t thread_row;
  uint32_t thread_col;

  __device__ ThreadInfo() {
    tid = threadIdx.x;
    warp_id = tid / Config::WARP_SIZE;
    lane_id = tid % Config::WARP_SIZE;
    warp_row = warp_id / Config::warps_per_row;
    warp_col = warp_id % Config::warps_per_row;
    thread_row = lane_id / (Config::warpMMATileN / Config::kTN);
    thread_col = lane_id % (Config::warpMMATileN / Config::kTN);
  }
};

template <typename Config>
struct BlockOffset {
  uint32_t m;
  uint32_t n;
};

template <typename Config>
struct SharedStorage {
  float a[Config::kStages][Config::kBK * Config::kBM];
  float b[Config::kStages][Config::kBK * Config::kBN];
};

template <typename Config>
struct GmemIteratorA {
  float const *A_block;
  uint32_t M;
  uint32_t K;
  uint32_t block_m;
  uint32_t load_smem_a_row;
  uint32_t load_smem_a_col;

  __device__ GmemIteratorA(float const *A_block_, uint32_t M_, uint32_t K_,
                           uint32_t block_m_, ThreadInfo<Config> const &thread)
      : A_block(A_block_), M(M_), K(K_), block_m(block_m_) {
    load_smem_a_row = thread.tid / Config::kBK;
    load_smem_a_col = thread.tid % Config::kBK;
  }

  __device__ void load(float (&ldg)[Config::load_a_iters],
                       uint32_t k_tile) const {
#pragma unroll
    for (uint32_t i = 0; i < Config::load_a_iters; i++) {
      uint32_t gk = k_tile * Config::kBK + load_smem_a_col;
      uint32_t gm =
          block_m + i * Config::load_smem_A_row_stride + load_smem_a_row;
      if (gk >= K || gm >= M) {
        ldg[i] = 0.0f;
      } else {
        uint32_t load_gmem_a =
            (i * Config::load_smem_A_row_stride + load_smem_a_row) * K + gk;
        ldg[i] = A_block[load_gmem_a];
      }
    }
  }

  __device__ void store(SharedStorage<Config> &shared,
                        float (&ldg)[Config::load_a_iters],
                        uint32_t stage) const {
#pragma unroll
    for (uint32_t i = 0; i < Config::load_a_iters; i++) {
      uint32_t load_smem_a = load_smem_a_col * Config::kBM +
                             load_smem_a_row +
                             i * Config::load_smem_A_row_stride;
      shared.a[stage][load_smem_a] = ldg[i];
    }
  }
};

template <typename Config>
struct GmemIteratorB {
  float const *B_block;
  uint32_t N;
  uint32_t K;
  uint32_t block_n;
  uint32_t load_smem_b_row;
  uint32_t load_smem_b_col;

  __device__ GmemIteratorB(float const *B_block_, uint32_t N_, uint32_t K_,
                           uint32_t block_n_, ThreadInfo<Config> const &thread)
      : B_block(B_block_), N(N_), K(K_), block_n(block_n_) {
    load_smem_b_row = thread.tid / Config::kBN;
    load_smem_b_col = thread.tid % Config::kBN;
  }

  __device__ void load(float (&ldg)[Config::load_b_iters],
                       uint32_t k_tile) const {
#pragma unroll
    for (uint32_t i = 0; i < Config::load_b_iters; i++) {
      uint32_t gk = k_tile * Config::kBK +
                    (i * Config::load_smem_B_row_stride + load_smem_b_row);
      uint32_t gn = block_n + load_smem_b_col;
      if (gk >= K || gn >= N) {
        ldg[i] = 0.0f;
      } else {
        uint32_t load_gmem_b = gk * N + load_smem_b_col;
        ldg[i] = B_block[load_gmem_b];
      }
    }
  }

  __device__ void store(SharedStorage<Config> &shared,
                        float (&ldg)[Config::load_b_iters],
                        uint32_t stage) const {
#pragma unroll
    for (uint32_t i = 0; i < Config::load_b_iters; i++) {
      uint32_t load_smem_b =
          (load_smem_b_row + i * Config::load_smem_B_row_stride) *
              Config::kBN +
          load_smem_b_col;
      shared.b[stage][load_smem_b] = ldg[i];
    }
  }
};

template <typename Config>
struct WarpMma {
  ThreadInfo<Config> thread;
  float aFragment[Config::kStages][Config::kWM_ITER * Config::kTM];
  float bFragment[Config::kStages][Config::kWN_ITER * Config::kTN];
  float accum[Config::kWM_ITER * Config::kWN_ITER * Config::kTM * Config::kTN];

  __device__ WarpMma(ThreadInfo<Config> const &thread_) : thread(thread_) {
#pragma unroll
    for (uint32_t i = 0;
         i < Config::kWM_ITER * Config::kWN_ITER * Config::kTM * Config::kTN;
         i++) {
      accum[i] = 0.0f;
    }
  }

  __device__ void load_fragments(SharedStorage<Config> const &shared,
                                 uint32_t stage, uint32_t k_inner) {
    float const *aSmemWarpTile = &shared.a[stage][thread.warp_row * Config::kWM];
    float const *bSmemWarpTile = &shared.b[stage][thread.warp_col * Config::kWN];

    uint32_t frag_idx = k_inner & 1u;
#pragma unroll
    for (int a = 0; a < static_cast<int>(Config::kWM_ITER); a++) {
      FETCH_FLOAT4(aFragment[frag_idx][a * Config::kTM]) = FETCH_CONST_FLOAT4(
          aSmemWarpTile[thread.thread_row * Config::kTM +
                        k_inner * Config::kBM + a * Config::warpMMATileM]);
    }
#pragma unroll
    for (int b = 0; b < static_cast<int>(Config::kWN_ITER); b++) {
      FETCH_FLOAT4(bFragment[frag_idx][b * Config::kTN]) = FETCH_CONST_FLOAT4(
          bSmemWarpTile[thread.thread_col * Config::kTN +
                        k_inner * Config::kBN + b * Config::warpMMATileN]);
    }
  }

  __device__ void mma(uint32_t frag_idx) {
#pragma unroll
    for (uint32_t a = 0; a < Config::kWM_ITER; a++) {
#pragma unroll
      for (uint32_t b = 0; b < Config::kWN_ITER; b++) {
#pragma unroll
        for (uint32_t reg_idx_a = 0; reg_idx_a < Config::kTM; reg_idx_a++) {
#pragma unroll
          for (uint32_t reg_idx_b = 0; reg_idx_b < Config::kTN; reg_idx_b++) {
            uint32_t acc_idx = (a * Config::kTM + reg_idx_a) *
                                   Config::kWN_ITER * Config::kTN +
                               b * Config::kTN + reg_idx_b;
            accum[acc_idx] +=
                aFragment[frag_idx][a * Config::kTM + reg_idx_a] *
                bFragment[frag_idx][b * Config::kTN + reg_idx_b];
          }
        }
      }
    }
  }
};

template <typename Config>
struct MmaPipeline {
  SharedStorage<Config> &shared;
  GmemIteratorA<Config> iterA;
  GmemIteratorB<Config> iterB;
  WarpMma<Config> &mma;
  float ldg_a[Config::load_a_iters];
  float ldg_b[Config::load_b_iters];

  __device__ MmaPipeline(SharedStorage<Config> &shared_,
                         GmemIteratorA<Config> const &iterA_,
                         GmemIteratorB<Config> const &iterB_,
                         WarpMma<Config> &mma_)
      : shared(shared_), iterA(iterA_), iterB(iterB_), mma(mma_) {}

  __device__ void run(uint32_t k_iter) {
    // Prologue: load k=0 tile to smem and prime fragments.
    iterA.load(ldg_a, 0);
    iterB.load(ldg_b, 0);
    iterA.store(shared, ldg_a, 0);
    iterB.store(shared, ldg_b, 0);
    __syncthreads();
    mma.load_fragments(shared, 0, 0);

    uint32_t write_stage_idx = 1;
    for (uint32_t k = 1; k < k_iter; k++) {
      iterA.load(ldg_a, k);
      iterB.load(ldg_b, k);

      uint32_t load_stage_idx = write_stage_idx ^ 1;
#pragma unroll
      for (uint32_t dot_product_idx = 1; dot_product_idx < Config::kBK;
           ++dot_product_idx) {
        mma.load_fragments(shared, load_stage_idx, dot_product_idx);
        mma.mma((dot_product_idx - 1) & 1u);
      }

      iterA.store(shared, ldg_a, write_stage_idx);
      iterB.store(shared, ldg_b, write_stage_idx);
      __syncthreads();

      mma.load_fragments(shared, write_stage_idx, 0);
      write_stage_idx ^= 1;
      mma.mma((Config::kBK - 1) & 1u);
    }

    uint32_t load_stage_idx = (k_iter - 1) & 1u;
#pragma unroll
    for (uint32_t dot_product_idx = 1; dot_product_idx < Config::kBK;
         ++dot_product_idx) {
      mma.load_fragments(shared, load_stage_idx, dot_product_idx);
      mma.mma((dot_product_idx - 1) & 1u);
    }
    mma.mma((Config::kBK - 1) & 1u);
  }
};

template <typename Config>
struct Epilogue {
  ThreadInfo<Config> thread;

  __device__ Epilogue(ThreadInfo<Config> const &thread_) : thread(thread_) {}

  __device__ void store(float *C_block, uint32_t M, uint32_t N,
                        BlockOffset<Config> const &block, float alpha,
                        float beta,
                        float const (&accum)[Config::kWM_ITER *
                                             Config::kWN_ITER * Config::kTM *
                                             Config::kTN]) const {
    float *final_C =
        &C_block[thread.warp_row * Config::kWM * N +
                 thread.warp_col * Config::kWN];
#pragma unroll
    for (uint32_t a = 0; a < Config::kWM_ITER; a++) {
#pragma unroll
      for (uint32_t b = 0; b < Config::kWN_ITER; b++) {
#pragma unroll
        for (uint32_t reg_idx_a = 0; reg_idx_a < Config::kTM; reg_idx_a++) {
#pragma unroll
          for (uint32_t reg_idx_b = 0; reg_idx_b < Config::kTN;
               reg_idx_b++) {
            uint32_t row_offset =
                a * Config::warpMMATileM + thread.thread_row * Config::kTM +
                reg_idx_a;
            uint32_t col_offset =
                b * Config::warpMMATileN + thread.thread_col * Config::kTN +
                reg_idx_b;
            uint32_t global_row = block.m + thread.warp_row * Config::kWM +
                                  row_offset;
            uint32_t global_col = block.n + thread.warp_col * Config::kWN +
                                  col_offset;
            if (global_row < M && global_col < N) {
              uint32_t acc_idx =
                  (a * Config::kTM + reg_idx_a) * Config::kWN_ITER *
                      Config::kTN +
                  b * Config::kTN + reg_idx_b;
              final_C[row_offset * N + col_offset] =
                  alpha * accum[acc_idx] +
                  beta * final_C[row_offset * N + col_offset];
            }
          }
        }
      }
    }
  }
};

} // namespace felix


__global__ void ampere_sgemm_64x64_nn(
    uint32_t M, uint32_t N, uint32_t K, float alpha, float const *__restrict__ A,
    float const *__restrict__ B, float beta, float *__restrict__ C) {
  using Config = felix::Sgemm64x64Config;
  using ThreadInfo = felix::ThreadInfo<Config>;
  using BlockOffset = felix::BlockOffset<Config>;
  using SharedStorage = felix::SharedStorage<Config>;
  using GmemIteratorA = felix::GmemIteratorA<Config>;
  using GmemIteratorB = felix::GmemIteratorB<Config>;
  using WarpMma = felix::WarpMma<Config>;
  using MmaPipeline = felix::MmaPipeline<Config>;
  using Epilogue = felix::Epilogue<Config>;

  ThreadInfo thread;
  BlockOffset block{blockIdx.y * Config::kBM,
                    blockIdx.x * Config::kBN};

  float const *A_block = A + block.m * K;
  float const *B_block = B + block.n;
  float *C_block = C + block.m * N + block.n;

  __shared__ SharedStorage shared;
  GmemIteratorA iterA(A_block, M, K, block.m, thread);
  GmemIteratorB iterB(B_block, N, K, block.n, thread);
  WarpMma mma(thread);
  MmaPipeline pipeline(shared, iterA, iterB, mma);

  uint32_t k_iter =
      (K + Config::kBK - 1) / Config::kBK;
  pipeline.run(k_iter);

  Epilogue epilogue(thread);
  epilogue.store(C_block, M, N, block, alpha, beta, mma.accum);
}
