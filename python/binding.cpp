#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <sion/sion.hpp>
using namespace sion;

PYBIND11_MODULE(sion, m) {
    m.doc() = "Sion, a High-Performance Deep Learning Operator Library";
    m.def("sgemm", &sgemm, "A function that performs Single-precision General Matrix-Matrix multiplication");
    m.def("flash_attention", &flash_attention, "A function that performs flash attention");
    m.def("gemm", &gemm, "A function that performs General Matrix-Matrix multiplication");
}
