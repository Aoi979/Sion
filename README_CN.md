# Sion
> ⚠️ 当前处于早期开发阶段，功能较少，性能暂不保证。
 
Sion 是一个 高性能 CUDA AI 算子库，专注深度学习核心算子的 GPU 实现。追求极致性能与数值稳定性。

本项目分为两个层级：
- felix —— 核心 CUDA 算子实现层
- sion —— 基于 Libtorch 的封装层，提供 Python 绑定

💡 名称来源于游戏 Eden* 中的角色 Sion

🌐 [English README.md](README.md)
## 已支持
- **SGEMM** (**SIMT**)
## 部分支持
- **Flash Attention** (**Ampere**)  
  目前仅支持 FP16, 形状必须对齐, 不支持 **mask**等特性。

- **HGEMM** (**Ampere**)  
  仅支持问题规模满足 (M, N, K) = (128, 128, 64) 整数倍。

## 环境要求

- **C++20**
- **CUDA 13.1+**
- **Libtorch**（PyTorch C++ API）
- **CMake 4.0+**

## 编译

```bash
git clone https://github.com/Aoi979/Sion.git
cd sion
mkdir build && cd build
cmake -G Ninja ..
ninja
```
若需启用 Python 绑定：
```bash
cmake -G Ninja -DBUILD_PYTHON_BINDING=ON ..
```
## 安装
构建完成后，执行：
```bash
ninja install
```

## 使用方法
### C++
#### Felix
```CMake
find_package(Felix REQUIRED)

target_link_libraries(your_target
    PRIVATE
        Felix::felix
)
```
#### Sion
```CMake
find_package(Sion REQUIRED)

target_link_libraries(your_target
    PRIVATE
        Sion::sion
)
```
### Python
编译时开启 `-DBUILD_PYTHON_BINDING=ON`。

示例：

```pythonimport torch
import sion

A = torch.randn(128, 256, device="cuda")
B = torch.randn(256, 512, device="cuda")

C = sion.sgemm(A, B)
```
## 贡献

Sion 目前是一个小型的实验性项目。

如果你对 GPU Kernel 设计或高性能算子实现感兴趣，欢迎参与贡献。

实现方式可以是：

- 纯手写 CUDA Kernel

- 基于 CuTe 的实现

请避免直接封装现有高层算子库（如 cuBLAS、cuDNN 等）。

本项目的目标是从底层实现算子。
欢迎提出想法、讨论与改进建议。
## 许可证
Sion 采用 MIT 许可证发布。详情请参见 [LICENSE](LICENSE) 文件。
