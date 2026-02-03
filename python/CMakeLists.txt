find_package(pybind11 REQUIRED)
set(TORCH_PYTHON_LIB "${TORCH_ROOT}/lib/libtorch_python.so")

message(STATUS "pybind11 include dir: ${pybind11_INCLUDE_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
add_library(pysion SHARED binding.cpp)

target_include_directories(pysion PRIVATE ${CMAKE_SOURCE_DIR}/include)

message(STATUS "BINDING: Torch libraries found: ${TORCH_LIBRARIES}")

target_link_libraries(pysion PRIVATE sion torch pybind11::module ${TORCH_PYTHON_LIB})

set_target_properties(pysion PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX ""
)
