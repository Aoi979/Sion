# Sion
Sion 是一个 高性能 CUDA AI 算子库，专注深度学习核心算子的 GPU 实现。追求极致性能与数值稳定性。
> 名称来源于游戏 Eden* 中的角色 Sion
> 
## 已支持
- **SGEMM** (**SIMT**)
## 部分支持
- **flash_attention** (**Ampere**) 
   
  目前仅支持 FP16, 形状必须对齐, 不支持 **mask**等特性。


  