#include <cuda_ops/cuda_ops.hpp>

#include <torch/library.h>

namespace {

at::Tensor cuda_ops_gemm_cuda(const at::Tensor &A, const at::Tensor &B,
                          double alpha, double beta) {
  return cuda_ops::gemm(A, B, static_cast<float>(alpha), static_cast<float>(beta));
}

at::Tensor cuda_ops_sgemm_cuda(const at::Tensor &A, const at::Tensor &B,
                           double alpha, double beta) {
  return cuda_ops::sgemm(A, B, static_cast<float>(alpha), static_cast<float>(beta));
}

at::Tensor cuda_ops_hgemm_cuda(const at::Tensor &A, const at::Tensor &B,
                           double alpha, double beta) {
  return cuda_ops::hgemm(A, B, static_cast<float>(alpha), static_cast<float>(beta));
}

at::Tensor cuda_ops_hgemm_nt_cuda(const at::Tensor &A, const at::Tensor &B,
                              double alpha, double beta) {
  return cuda_ops::hgemm_nt(A, B, static_cast<float>(alpha),
                        static_cast<float>(beta));
}

at::Tensor cuda_ops_flash_attention_cuda(const at::Tensor &query,
                                     const at::Tensor &key,
                                     const at::Tensor &value) {
  return cuda_ops::flash_attention(query, key, value);
}

} // namespace

TORCH_LIBRARY(cuda_ops, m) {
  m.def("gemm(Tensor A, Tensor B, float alpha=1.0, float beta=0.0) -> Tensor");
  m.def("sgemm(Tensor A, Tensor B, float alpha=1.0, float beta=0.0) -> Tensor");
  m.def("hgemm(Tensor A, Tensor B, float alpha=1.0, float beta=0.0) -> Tensor");
  m.def("hgemm_nt(Tensor A, Tensor B, float alpha=1.0, float beta=0.0) -> "
        "Tensor");
  m.def("flash_attention(Tensor query, Tensor key, Tensor value) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_ops, CUDA, m) {
  m.impl("gemm", TORCH_FN(cuda_ops_gemm_cuda));
  m.impl("sgemm", TORCH_FN(cuda_ops_sgemm_cuda));
  m.impl("hgemm", TORCH_FN(cuda_ops_hgemm_cuda));
  m.impl("hgemm_nt", TORCH_FN(cuda_ops_hgemm_nt_cuda));
  m.impl("flash_attention", TORCH_FN(cuda_ops_flash_attention_cuda));
}
