#pragma once
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
namespace sion {
static inline void cuda_check(cudaError_t e, const char* msg) {
    TORCH_CHECK(e == cudaSuccess, msg, ": ", cudaGetErrorString(e));
}    
}
