add_library(pysion MODULE binding.cpp)

target_include_directories(pysion PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(pysion PRIVATE sion Python3::Module)
target_link_options(pysion PRIVATE -static-libstdc++ -static-libgcc)

set_target_properties(pysion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sion"
    PREFIX ""
    OUTPUT_NAME "_C"
)
