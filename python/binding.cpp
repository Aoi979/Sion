#include <pybind11/pybind11.h>

namespace py = pybind11;
int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(py_sion, m) {
    m.doc() = "Simple test module";
    m.def("add", &add, "A function that adds two numbers");
}
