find_package(CUDAToolkit REQUIRED)
add_executable(sgemm_bench sgemm_bench.cpp)
target_link_libraries(sgemm_bench PRIVATE sion CUDA::cudart)
target_include_directories(sgemm_bench PRIVATE ${CMAKE_SOURCE_DIR}/include)
