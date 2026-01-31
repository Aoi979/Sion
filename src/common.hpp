#pragma once
namespace sion {
static inline void cuda_check(cudaError_t e, const char* msg) {
    TORCH_CHECK(e == cudaSuccess, msg, ": ", cudaGetErrorString(e));
}    
}
