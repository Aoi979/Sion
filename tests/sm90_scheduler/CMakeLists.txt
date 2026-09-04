find_package(CUDAToolkit REQUIRED)

add_executable(sm90_tile_scheduler_test sm90_tile_scheduler_test.cu)

set_target_properties(sm90_tile_scheduler_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)

target_include_directories(sm90_tile_scheduler_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cuda_ops_core
)

target_link_libraries(sm90_tile_scheduler_test PRIVATE CUDA::cudart)

add_test(NAME sm90_tile_scheduler_test
         COMMAND sm90_tile_scheduler_test)
