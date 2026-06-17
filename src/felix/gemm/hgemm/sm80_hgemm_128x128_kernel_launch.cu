#include "kernels/sm80_hgemm_128x128.cuh"
#include <felix/felix.hpp>
#include <felix/registry.hpp>

namespace felix {
namespace {

FelixStatus validate_sm80_hgemm(uint32_t M, uint32_t N, uint32_t K,
                                float alpha, float beta, half const *A,
                                half const *B, half *C) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 ||
      K == 0 || M % shape_mnk::M != 0 || N % shape_mnk::N != 0 ||
      K % shape_mnk::K != 0) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM requires non-null A/B/C and M/N/K aligned to 128/128/64");
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return FelixStatus::make(
        FelixStatus::Type::API_ERROR, cudaErrorInvalidValue,
        "sm80 HGEMM implements C = A * B only; require alpha=1 and beta=0");
  }
  return {};
}

FelixStatus convert_sm80_hgemm_status(cudaError_t err) {
  if (err != cudaSuccess) {
    return FelixStatus::make(FelixStatus::Type::KERNEL_LAUNCH_FAILED, err);
  }
  return {};
}

} // namespace

FelixStatus sm80_hgemm_128x128x64_fp16acc_kernel_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm80_hgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  return convert_sm80_hgemm_status(sm80_hgemm::launch_hgemm_128x128x64_fp16acc(
      const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream));
}

FelixStatus sm80_hgemm_128x128x64_fp32acc_kernel_launch(
    uint32_t M, uint32_t N, uint32_t K, float alpha, half const *A,
    half const *B, float beta, half *C, cudaStream_t stream) {
  auto status = validate_sm80_hgemm(M, N, K, alpha, beta, A, B, C);
  if (!status.ok()) {
    return status;
  }
  return convert_sm80_hgemm_status(sm80_hgemm::launch_hgemm_128x128x64_fp32acc(
      const_cast<half *>(A), const_cast<half *>(B), C, static_cast<int>(M),
      static_cast<int>(N), static_cast<int>(K), stream));
}

} // namespace felix

REGISTER_KERNEL(
    sm80_hgemm_128x128x64_fp16acc,
    (felix::KernelEntry{
        felix::KernelType::HGEMM, "sm80_hgemm_128x128x64_fp16acc",
        (void *)felix::sm80_hgemm_128x128x64_fp16acc_kernel_launch, false}));

REGISTER_KERNEL(
    sm80_hgemm_128x128x64_fp32acc,
    (felix::KernelEntry{
        felix::KernelType::HGEMM, "sm80_hgemm_128x128x64_fp32acc",
        (void *)felix::sm80_hgemm_128x128x64_fp32acc_kernel_launch, true}));
