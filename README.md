# CudaOps

CudaOps is an experimental CUDA operator library for exploring handwritten and CuTe-based GPU kernels.

## Build

Requirements: C++20, CUDA 13.1+, CMake 4.0+, Ninja, and LibTorch.

```bash
git clone https://github.com/Aoi979/CudaOps.git
cd CudaOps
cmake -S . -B build -G Ninja
cmake --build build
```

To build the Python bindings:

```bash
cmake -S . -B build -G Ninja -DBUILD_PYTHON_BINDING=ON
cmake --build build
```

## API

The Python API follows the PyTorch tensor style:

```python
import torch
import cuda_ops

A = torch.randn(128, 256, device="cuda")
B = torch.randn(256, 512, device="cuda")
C = cuda_ops.sgemm(A, B)
```

Available operators include `gemm`, `sgemm`, `hgemm`, `hgemm_nt`, and `flash_attention`.

For C++:

```cmake
find_package(CudaOps REQUIRED)
target_link_libraries(your_target PRIVATE CudaOps::cuda_ops)
```

The C++ API accepts and returns LibTorch tensors:

```cpp
#include <cuda_ops/cuda_ops.hpp>

torch::Tensor a = torch::randn(
    {128, 256},
    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));
torch::Tensor b = torch::randn(
    {256, 512},
    torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

torch::Tensor c = cuda_ops::sgemm(a, b, 1.0f, 0.0f);
```
