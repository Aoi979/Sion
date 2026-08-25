set(_cuda_ops_maca_hgemm_kernel
    "${CMAKE_CURRENT_SOURCE_DIR}/../kernels/maca/xcore1000_hgemm_f16_nt_m256n256k64_fp32acc.hpp")
set(_cuda_ops_maca_hgemm_builtins
    "${CMAKE_CURRENT_SOURCE_DIR}/../detail/maca/mxc_builtins.hpp")
set(_cuda_ops_maca_hgemm_launcher
    "${CMAKE_CURRENT_SOURCE_DIR}/../launchers/maca/xcore1000_hgemm_f16_nt_m256n256k64_fp32acc_launcher.cpp")

cuda_ops_add_maca_object(
    cuda_ops_core_maca_xcore1000_hgemm_f16_nt_m256n256k64_fp32acc
    SOURCE "${_cuda_ops_maca_hgemm_launcher}"
    OUTPUT xcore1000_hgemm_f16_nt_m256n256k64_fp32acc_launcher.o
    DEPENDS
        "${_cuda_ops_maca_hgemm_kernel}"
        "${_cuda_ops_maca_hgemm_builtins}")
