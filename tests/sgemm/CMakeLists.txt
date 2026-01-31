add_executable(sgemm_test sgemm_test.cpp)
target_link_libraries(sgemm_test PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(sgemm_test PRIVATE sion)
