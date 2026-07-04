find_package(CUDAToolkit REQUIRED)

add_executable(sion_bench runners/sion_bench.cu)

target_link_libraries(sion_bench PRIVATE sion CUDA::cudart)
target_include_directories(sion_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

add_custom_target(bench DEPENDS sion_bench)
add_custom_target(benchmarks DEPENDS sion_bench)
