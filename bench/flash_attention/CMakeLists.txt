find_package(CUDAToolkit REQUIRED)
add_executable(fa_bench fa_bench.cpp)
target_link_libraries(fa_bench PRIVATE sion CUDA::cudart)
