#include "../common.hpp"
#include "../../include/sion/sion.hpp"
torch::Tensor sgemm_ref(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    auto A_ = A.to(torch::kFloat32);
    auto B_ = B.to(torch::kFloat32);
    auto C = torch::mm(A_, B_);
    return C.to(A.dtype());
}

torch::Tensor sgemm_op(const torch::Tensor& A, const torch::Tensor& B){
    return sion::sgemm(A, B, 1.0, 0.0);
}

SION_TEST(test_sgemm_basic){
     int M = 2048, K = 2048, N = 2048;

    SION_CHECK(torch::cuda::is_available());

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor A = torch::rand({M, K}, opts);
    torch::Tensor B = torch::rand({K, N}, opts);

    auto ref = sgemm_ref(A, B);
    auto val = sgemm_op(A, B);
    auto stats = sion::test::compare_tensor(ref, val, 1e-6);
    sion::test::print_stats_md_file(stats, "sgemm_basic", A.numel(), 1e-6, "sgemm_report.md", true);
    SION_CHECK(sion::test::check_pass(stats, 1e-6));
}

int main(){
    auto& tests = TestRegistry::inst().tests;
    std::cout << "[Sion] running " << tests.size() << " tests\n";
    for(auto& [name, fn]: tests){
        std::cout << "  - " << name << std::endl;
        fn();
    }
    std::cout << "[Sion] all tests passed\n";
    return 0;
}