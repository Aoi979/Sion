#include <torch/extension.h>
#include <felix/registry.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sion/sion.hpp>

namespace py = pybind11;
using namespace sion;

namespace {
std::vector<std::string> kernel_names(felix::KernelType type) {
    std::vector<std::string> names;
    for (const auto &entry : felix::global_registry().all()) {
        if (entry.type == type) {
            names.push_back(entry.name);
        }
    }
    return names;
}
} // namespace

PYBIND11_MODULE(sion, m) {
    m.doc() = "Sion, a High-Performance Deep Learning Operator Library";
    m.def("sgemm", &sgemm,
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          py::arg("kernel_name") = "cute_sgemm_64x64_nn",
          "Single-precision GEMM with selectable Felix kernel");
    m.def("hgemm", &hgemm,
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          py::arg("kernel_name") = "cute_hgemm_128x128_nn",
          "Half-precision GEMM with selectable Felix kernel");
    m.def("hgemm_nt", &hgemm_nt,
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f,
          py::arg("kernel_name") = "cute_hgemm_128x128_nt",
          "Half-precision NT GEMM with selectable Felix kernel");
    m.def("flash_attention", &flash_attention,
          "A function that performs flash attention");
    m.def("gemm", &gemm,
          py::arg("A"), py::arg("B"), py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f, py::arg("kernel_name") = "",
          "General GEMM with optional selectable Felix kernel");
    m.def("sgemm_kernels", [] { return kernel_names(felix::KernelType::SGEMM); });
    m.def("hgemm_kernels", [] { return kernel_names(felix::KernelType::HGEMM); });
}
