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

torch::Tensor dummy_sgemm(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int64_t M = A.size(0);
    int64_t K1 = A.size(1);
    int64_t K2 = B.size(0);
    int64_t N = B.size(1);
    TORCH_CHECK(K1 == K2, "Inner dimensions must match for matrix multiplication");
    return torch::zeros({M, N}, A.options());
}

torch::Tensor sgemm_op(const torch::Tensor& A, const torch::Tensor& B){
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    const bool mn_aligned = (M % 128 == 0) && (N % 128 == 0);
    const bool k_aligned  = (K % 16  == 0);
    if (mn_aligned && k_aligned) {
        return sion::sgemm(A, B, /*alpha=*/1.0f, /*beta=*/0.0f);
    } else {
        // fallback
        return dummy_sgemm(A, B);
    }
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
    sion::test::print_stats_md_file(stats, "sgemm_basic", ref.numel(), 1e-6, "sgemm_report.md", true);
}

SION_TEST(test_sgemm_unaligned0){
     int M = 2112, K = 2048, N = 2112;

    SION_CHECK(torch::cuda::is_available());

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor A = torch::rand({M, K}, opts);
    torch::Tensor B = torch::rand({K, N}, opts);

    auto ref = sgemm_ref(A, B);
    auto val = sgemm_op(A, B);
    auto stats = sion::test::compare_tensor(ref, val, 1e-6);
    sion::test::print_stats_md_file(stats, "sgemm_next", ref.numel(), 1e-6, "sgemm_report.md", false);
}

SION_TEST(test_sgemm_unaligned1){
     int M = 2049, K = 2048, N = 2049;

    SION_CHECK(torch::cuda::is_available());

    auto opts = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCUDA);

    torch::Tensor A = torch::rand({M, K}, opts);
    torch::Tensor B = torch::rand({K, N}, opts);

    auto ref = sgemm_ref(A, B);
    auto val = sgemm_op(A, B);
    auto stats = sion::test::compare_tensor(ref, val, 1e-6);
    sion::test::print_stats_md_file(stats, "sgemm_final", ref.numel(), 1e-6, "sgemm_report.md", false);
}

int main(){
    auto& tests = TestRegistry::inst().tests;
    std::cout << "[Sion] running " << tests.size() << " tests\n";
    for(auto& [name, fn]: tests){
        std::cout << "  - " << name << std::endl;
        fn();
    }
    std::cout << "[Sion] all tests completed, please check the report\n";
    return 0;
}