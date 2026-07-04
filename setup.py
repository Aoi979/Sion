import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent.resolve()

sources = [
    "python/binding.cpp",
    "src/felix/runtime/status.cpp",
    "src/felix/runtime/registry.cpp",
    "src/felix/runtime/api.cpp",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m64n64k8_basic.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m64n64k8_cute.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m64n64k8_cute_swizzle.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_cute.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_cute_warp_tiling_db.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k16_a1b0.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_one_cta_per_sm.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_warp_order.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_schedule.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_copy_schedule.cu",
    "src/felix/gemm/sgemm/launchers/sm80_sgemm_f32_nn_m128n128k8_stage5_cutlass_sm80_mma_order.cu",
    "src/felix/gemm/hgemm/launchers/sm80_hgemm_f16_nn_m128n128k64_cute_mma16816.cu",
    "src/felix/gemm/hgemm/launchers/sm80_hgemm_f16_nt_m128n128k64_cute_mma16816.cu",
    "src/felix/gemm/hgemm/launchers/sm80_hgemm_f16_nn_m128n128k64_fp16acc.cu",
    "src/felix/gemm/hgemm/launchers/sm80_hgemm_f16_nn_m128n128k64_fp32acc.cu",
    "src/felix/gemm/hgemm/launchers/sm80_hgemm_f16_nn_m128n256k64_fp16acc.cu",
    "src/felix/flash_attention/launchers/sm80_flash_attn_f16_hd64_bq128_bk128_mma16816_v2.cu",
    "src/felix/flash_attention/launchers/sm80_flash_attn_f16_hd128_bq128_bk64_mma16816_v2.cu",
    "src/felix/topk/launchers/cuda_topk_f32_radix_select.cu",
    "src/sion/operators/gemm.cpp",
    "src/sion/operators/sgemm.cpp",
    "src/sion/operators/hgemm.cpp",
    "src/sion/operators/flash_attention.cpp",
    "src/sion/torch/library.cpp",
]

if os.getenv("SION_BUILD_SM90_KERNELS", "0") == "1":
    sources += [
        "src/felix/gemm/hgemm/launchers/sm90_hgemm_f16_nn_m128n128k64_pingpong.cu",
        "src/felix/gemm/hgemm/launchers/sm90_hgemm_f16_nn_m128n256k64_cooperative.cu",
    ]

setup(
    name="sion",
    version="0.1.0",
    package_dir={"": "python"},
    packages=["sion"],
    package_data={"sion": ["py.typed", "__init__.pyi"]},
    ext_modules=[
        CUDAExtension(
            name="sion._C",
            sources=[str(ROOT / source) for source in sources],
            include_dirs=[
                str(ROOT / "include"),
                str(ROOT / "third_party/cutlass/include"),
                str(ROOT / "src/felix"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    "-lineinfo",
                    "--expt-relaxed-constexpr",
                ],
            },
            extra_link_args=[
                "-Wl,--no-as-needed",
                "-static-libstdc++",
                "-static-libgcc",
                "-lcuda",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
