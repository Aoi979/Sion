# Sion
Sion 是一个 高性能 CUDA AI 算子库，专注深度学习核心算子的 GPU 实现。追求极致性能与数值稳定性。
> 名称来源于游戏 Eden* 中的角色 Sion
> 
## 已支持
- **SGEMM** (**SIMT**)
## 部分支持
- **flash_attention** (**Ampere**) 
   
  目前仅支持 FP16, 形状必须对齐, 不支持 **mask**等特性。


## 环境要求

- **C++20**
- **CUDA 13.1+**
- **Libtorch**（PyTorch C++ API）

## 编译

```bash
git clone https://github.com/Aoi979/Sion.git
cd sion
mkdir build && cd build
cmake -G Ninja -DTORCH_ROOT=/path/to/libtorch ..
ninja
```
若需启用 Python 绑定：
```bash
cmake -G Ninja -DBUILD_PYTHON_BINDING=ON -DTORCH_ROOT=/path/to/libtorch ..
```
## 许可证
Sion 采用 MIT 许可证发布。详情请参见 [LICENSE](LICENSE) 文件。
