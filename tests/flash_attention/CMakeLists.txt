add_executable(fa_test fa_test.cpp)
target_link_libraries(fa_test PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(fa_test PRIVATE sion)
