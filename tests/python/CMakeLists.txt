function(add_cuda_ops_python_test name script)
  add_test(NAME ${name}
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${script})
  set_tests_properties(${name} PROPERTIES
      SKIP_RETURN_CODE 77
      ENVIRONMENT
        "PYTHONPATH=${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH};LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/src/cuda_ops:${CMAKE_BINARY_DIR}/src/cuda_ops_core:${TORCH_ROOT}/lib:$ENV{LD_LIBRARY_PATH}")
endfunction()

add_cuda_ops_python_test(python_sgemm_test test_sgemm.py)
add_cuda_ops_python_test(python_hgemm_test test_hgemm.py)
add_cuda_ops_python_test(python_flash_attention_test test_flash_attention.py)
