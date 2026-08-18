#include <type_traits>
#include <cuda_runtime.h>

template <int B, int M, int S>
struct Swizzle
{
    static_assert(B >= 0, "B must be >= 0");
    static_assert(M >= 0, "M must be >= 0");

    static constexpr int mask = (1 << B) - 1;

    static constexpr int hi_shift = M + (S > 0 ? S : 0);
    static constexpr int lo_shift = M - (S < 0 ? S : 0);

    /// Swizzle a linear element index
    template <typename Index>
    __host__ __device__ static constexpr Index apply(Index i)
    {
        static_assert(std::is_integral<Index>::value,
                      "Index must be an integral type");

        return i ^ (((i >> hi_shift) & mask) << lo_shift);
    }
};
