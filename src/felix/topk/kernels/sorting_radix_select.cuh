// Reference:
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/SortingRadixSelect.cuh
#include <cassert>
#include <cstdint>
#define WARP_BALLOT(pred, mask) __ballot_sync(mask, pred)

template <typename T> struct Bitfield {
  __device__ __forceinline__ static T getBitfield(T val, int pos, int bits) {
    return (val >> pos) & ((((T)1) << bits) - 1);
  }
  __device__ __forceinline__ static T setBitfield(T value, T toInsert, int pos,
                                                  int width) {
    T mask = ((T(1) << width) - 1) << pos;
    value &= ~mask;
    value |= (toInsert << pos) & mask;
    return value;
  }
};

template <typename scalar_t> struct TopKTypeConfig {};

template <> struct TopKTypeConfig<float> {
  using RadixType = uint32_t;
  static inline __device__ RadixType convert(float v) {
    RadixType x = __float_as_int(v);
    RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;

    return (v == v) ? (x ^ mask) : 0xffffffff;
  }

  static inline __device__ float deconvert(RadixType v) {
    RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;

    return __int_as_float(v ^ mask);
  }
};

constexpr int RADIX_BITS = 2;
constexpr int RADIX_SIZE = 4;
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

template <typename scalar_t, typename bitwise_t, typename index_t,
          typename CountType, int RadixSize, int RadixBits>
__device__ void
countRadixUsingMask(CountType counts[RadixSize], CountType *smem,
                    bitwise_t desired, bitwise_t desiredMask, int radixDigitPos,
                    index_t sliceSize, index_t withinSliceStride,
                    const scalar_t *data) {
#pragma unroll
  for (int i = 0; i < RadixSize; i++) {
    counts[i] = 0;
  }
  if (threadIdx.x < RadixSize) {
    smem[threadIdx.x] = 0;
  }
  __syncthreads();

  uint32_t mask = WARP_BALLOT(threadIdx.x < sliceSize, 0xffffffff);

  for (index_t i = threadIdx.x; i < sliceSize;) {
    bitwise_t val =
        TopKTypeConfig<scalar_t>::convert(data[i * withinSliceStride]);
    bool hasVal = ((val & desiredMask) == desired);

    bitwise_t digitInRadix =
        Bitfield<bitwise_t>::getBitfield(val, radixDigitPos, RadixBits);

    for (uint32_t j = 0; j < RadixSize; j++) {
      bool vote = hasVal && (digitInRadix == j);
      counts[j] += __popc(WARP_BALLOT(vote, mask));
    }
    i += blockDim.x;
    mask = WARP_BALLOT(i < sliceSize, mask);
  }
  if (threadIdx.x % 32 == 0) {
    for (uint32_t i = 0; i < RadixSize; i++) {
      atomicAdd(&smem[i], counts[i]);
    }
  }
  __syncthreads();

  for (uint32_t i = 0; i < RadixSize; i++) {
    counts[i] = smem[i];
  }
  __syncthreads();
}

template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ scalar_t findPattern(scalar_t *smem, const scalar_t *data,
                                index_t sliceSize, index_t withinSliceStride,
                                bitwise_t desired, bitwise_t desiredMask) {
  // found  : smem[0]
  // val    : smem[1]
  if (threadIdx.x < 2) {
    smem[threadIdx.x] = static_cast<scalar_t>(0);
  }
  __syncthreads();
  index_t blockSize = static_cast<index_t>(blockDim.x);

  index_t numIterations = ((sliceSize + blockSize - 1) / blockSize) * blockSize;

  for (index_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
    bool inRange = (i < sliceSize);
    scalar_t v =
        inRange ? data[i * withinSliceStride] : static_cast<scalar_t>(0);
    if (inRange &&
        ((TopKTypeConfig<scalar_t>::convert(v) & desiredMask) == desired)) {
      // There should not be conflicts if we are using findPattern,
      // since the result is unique
      smem[0] = static_cast<scalar_t>(1);
      smem[1] = v; // can't use val as the flag, since it could be 0
    }
    __syncthreads();

    scalar_t found = smem[0];
    scalar_t val = smem[1];

    __syncthreads();

    if (found != static_cast<scalar_t>(0)) {
      return val;
    }
  }
  assert(false);
  return static_cast<scalar_t>(0);
}

template <typename scalar_t, typename bitwise_t, typename index_t>
__device__ void radixSelect(const scalar_t *data, index_t k, bool largest,
                            index_t sliceSize, index_t withinSliceStride,
                            int *smem, scalar_t *topK) {
  int counts[RADIX_SIZE];
  bitwise_t desired = 0;
  bitwise_t desiredMask = 0;

  int kToFind = k;

  for (int digitPos = sizeof(scalar_t) * 8 - RADIX_BITS; digitPos >= 0;
       digitPos -= RADIX_BITS) {
    countRadixUsingMask<scalar_t, bitwise_t, index_t, int, RADIX_SIZE,
                        RADIX_BITS>(counts, smem, desired, desiredMask,
                                    digitPos, sliceSize, withinSliceStride,
                                    data);
    auto found_unique = [&](int i, int count) -> bool {
      if (count == 1 && kToFind == 1) {
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(desiredMask, RADIX_MASK,
                                                       digitPos, RADIX_BITS);
        *topK = findPattern((scalar_t *)smem, data, sliceSize,
                            withinSliceStride, desired, desiredMask);
        return true;
      }
      return false;
    };
    auto found_non_unique = [&](int i, int count) -> bool {
      if (count >= kToFind) {
        desired =
            Bitfield<bitwise_t>::setBitfield(desired, i, digitPos, RADIX_BITS);
        desiredMask = Bitfield<bitwise_t>::setBitfield(desiredMask, RADIX_MASK,
                                                       digitPos, RADIX_BITS);
        return true;
      }
      kToFind -= count;
      return false;
    };
    if (largest) {
      for (int i = RADIX_SIZE - 1; i >= 0; --i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    } else {
      for (int i = 0; i < RADIX_SIZE; ++i) {
        int count = counts[i];
        if (found_unique(i, count)) {
          return;
        }
        if (found_non_unique(i, count)) {
          break;
        }
      }
    }
  }
  *topK = TopKTypeConfig<scalar_t>::deconvert(desired);
}

__global__ void sorting_radix_select_kernel(const float *data, float *out,
                                            uint32_t num_slices,
                                            uint32_t slice_size, uint32_t k,
                                            bool largest) {
  const uint32_t slice = static_cast<uint32_t>(blockIdx.x);
  if (slice >= num_slices) {
    return;
  }

  const float *slice_data = data + slice * slice_size;

  __shared__ int smem[RADIX_SIZE];
  __shared__ float topk;

  radixSelect<float, TopKTypeConfig<float>::RadixType, int>(
      slice_data, static_cast<int>(k), largest, static_cast<int>(slice_size),
      /*withinSliceStride=*/1, smem, &topk);

  __syncthreads();
  if (threadIdx.x == 0) {
    out[slice] = topk;
  }
}
