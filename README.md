# Sion

Sion is a high-performance CUDA AI operator library, focusing on GPU implementations of core deep learning operators. It aims for extreme performance and numerical stability.

> ⚠️ Early development stage. Features are limited and high performance is not guaranteed.

💡 The name is inspired by the character Sion from the game *Eden**.

🌐 [中文版 README_CN.md](README_CN.md)

## Supported

- **SGEMM** (**SIMT**)

## Partially Supported

- **Flash Attention** (**Ampere**)  
  Currently only supports FP16. Shapes must be aligned. Features like **mask** are not supported.

## Requirements

- **C++20**
- **CUDA 13.1+**
- **Libtorch** (PyTorch C++ API)
- **CMake 4.0+**

## Build Instructions

```bash
git clone https://github.com/Aoi979/Sion.git
cd sion
mkdir build && cd build
cmake -G Ninja ..
ninja
```
To enable Python bindings:
```bash
cmake -G Ninja -DBUILD_PYTHON_BINDING=ON ..
```

## Installation
After building Sion, install it with:
```bash
ninja install
```

## Usage
### C++
```CMake
find_package(Sion REQUIRED)

target_link_libraries(your_target
    PRIVATE
        Sion::sion
)
```
### Python
Build with `-DBUILD_PYTHON_BINDING=ON`.

Example:
```python
import torch
import sion

A = torch.randn(128, 256, device="cuda")
B = torch.randn(256, 512, device="cuda")

C = sion.sgemm(A, B)
```
## Contributing

Sion is currently a small experimental project.

If you're interested in GPU kernel design or high-performance operator implementation,
feel free to contribute.

Implementations can be:

- Pure handwritten CUDA kernels
- Built with CuTe

Please avoid wrapping existing high-level libraries (e.g. cuBLAS, cuDNN, etc.).
The goal is to explore operator implementations from scratch.

Discussions, ideas, and improvements are all welcome.

## License
Sion is released under the MIT License. See [LICENSE](LICENSE) for details.
