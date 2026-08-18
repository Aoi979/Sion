find_package(CUDAToolkit REQUIRED)

add_executable(cuda_ops_bench runners/cuda_ops_bench.cu)

target_link_libraries(cuda_ops_bench PRIVATE cuda_ops CUDA::cudart)
target_include_directories(cuda_ops_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_custom_target(bench DEPENDS cuda_ops_bench)
add_custom_target(benchmarks DEPENDS cuda_ops_bench)
