find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
message(STATUS "pybind11 include dir: ${pybind11_INCLUDE_DIRS}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
# pybind11_add_module(pysion binding.cpp)
add_library(pysion SHARED binding.cpp)
message(${Python_INCLUDE_DIRS})
target_link_libraries(pysion PRIVATE sion pybind11::module)


