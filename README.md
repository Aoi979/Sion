# Sion

Sion is a high-performance CUDA AI operator library, focusing on GPU implementations of core deep learning operators. It aims for extreme performance and numerical stability.

> ⚠️ Early development stage. Features are limited and high performance is not guaranteed.

> The name is inspired by the character Sion from the game *Eden**.

> 中文版 [README_CN.md](README_CN.md)
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

```CMake
find_package(Sion REQUIRED)

target_link_libraries(your_target
    PRIVATE
        Sion::sion
)
```
## License
Sion is released under the MIT License. See [LICENSE](LICENSE) for details.
