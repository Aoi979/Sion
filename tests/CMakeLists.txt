add_subdirectory(sgemm)
add_subdirectory(flash_attention)
add_test(NAME sgemm_test COMMAND sgemm_test)
add_test(NAME fa_test COMMAND fa_test)