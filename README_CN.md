# CudaOps

CudaOps 是一个用于探索手写 CUDA 与 CuTe kernel 的实验性算子库。

## 编译

环境要求：C++20、CUDA 13.1+、CMake 4.0+、Ninja 和 LibTorch。

```bash
git clone https://github.com/Aoi979/CudaOps.git
cd CudaOps
cmake -S . -B build -G Ninja
cmake --build build
```

编译 Python 绑定：

```bash
cmake -S . -B build -G Ninja -DBUILD_PYTHON_BINDING=ON
cmake --build build
```

## API

Python API 遵循 PyTorch Tensor 风格：

```python
import torch
import cuda_ops

A = torch.randn(128, 256, device="cuda")
B = torch.randn(256, 512, device="cuda")
C = cuda_ops.sgemm(A, B)
```

当前提供 `gemm`、`sgemm`、`hgemm`、`hgemm_nt` 和 `flash_attention`。

C++ 接口：

```cmake
find_package(CudaOps REQUIRED)
target_link_libraries(your_target PRIVATE CudaOps::cuda_ops)
```

C++ API 接受并返回 LibTorch Tensor：

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
