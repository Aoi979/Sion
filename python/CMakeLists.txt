add_library(cuda_ops_python MODULE binding.cpp)

target_include_directories(cuda_ops_python PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(cuda_ops_python PRIVATE cuda_ops Python3::Module)
target_link_options(cuda_ops_python PRIVATE -static-libstdc++ -static-libgcc)

set_target_properties(cuda_ops_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cuda_ops"
    PREFIX ""
    OUTPUT_NAME "_C"
)
