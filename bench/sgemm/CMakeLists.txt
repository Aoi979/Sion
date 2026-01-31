add_executable(sgemm_bench sgemm_bench.cpp)
target_link_libraries(sgemm_bench PRIVATE ${TORCH_LIBRARIES})
