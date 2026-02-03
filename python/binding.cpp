#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <sion/sion.hpp>
using namespace sion;

PYBIND11_MODULE(pysion, m) {
    m.doc() = "Sion, a High-Performance Deep Learning Operator Library";
    m.def("sgemm", &sgemm, "A function that performs matrix multiplication");
}
