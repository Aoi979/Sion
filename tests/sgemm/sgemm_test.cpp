#include "../../include/sion/sion.hpp"
#include "../common.hpp"
torch::Tensor sgemm_ref(const torch::Tensor &A, const torch::Tensor &B,
                        float alpha, float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  auto A_ = A.to(torch::kFloat32);
  auto B_ = B.to(torch::kFloat32);

  int64_t m = A_.size(0);
  int64_t n = B_.size(1);

  auto C_ = torch::zeros({m, n}, A_.options());

  auto result = torch::addmm(C_, // self
                             A_, // mat1
                             B_, // mat2
                             beta, alpha);

  return result.to(A.dtype());
}

torch::Tensor sgemm_op(const torch::Tensor &A, const torch::Tensor &B, float alpha, float beta) {
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(1);
  return sion::sgemm(A, B, alpha, beta);
}

SION_TEST(test_sgemm_basic) {
  int M = 2048, K = 2048, N = 2048;

  SION_CHECK(torch::cuda::is_available());

  auto opts =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

  torch::Tensor A = torch::rand({M, K}, opts);
  torch::Tensor B = torch::rand({K, N}, opts);

  auto ref = sgemm_ref(A, B, 2, 2);
  auto val = sgemm_op(A, B, 2, 2);
  auto stats = sion::test::compare_tensor(ref, val, 1e-6);
  sion::test::add_record("sgemm_basic", ref.numel(), stats, 1e-6);
}

// SION_TEST(test_sgemm_unaligned0){
//      int M = 2112, K = 2048, N = 2112;

//     SION_CHECK(torch::cuda::is_available());

//     auto opts = torch::TensorOptions()
//         .dtype(torch::kFloat32)
//         .device(torch::kCUDA);

//     torch::Tensor A = torch::rand({M, K}, opts);
//     torch::Tensor B = torch::rand({K, N}, opts);

//     auto ref = sgemm_ref(A, B);
//     auto val = sgemm_op(A, B);
//     auto stats = sion::test::compare_tensor(ref, val, 1e-6);
//     sion::test::add_record("sgemm_unaligned0", ref.numel(), stats, 1e-6);
// }

// SION_TEST(test_sgemm_unaligned1){
//      int M = 2049, K = 2048, N = 2049;

//     SION_CHECK(torch::cuda::is_available());

//     auto opts = torch::TensorOptions()
//         .dtype(torch::kFloat32)
//         .device(torch::kCUDA);

//     torch::Tensor A = torch::rand({M, K}, opts);
//     torch::Tensor B = torch::rand({K, N}, opts);

//     auto ref = sgemm_ref(A, B);
//     auto val = sgemm_op(A, B);
//     auto stats = sion::test::compare_tensor(ref, val, 1e-6);
//     sion::test::add_record("sgemm_unaligned1", ref.numel(), stats, 1e-6);
// }

int main() {
  auto &tests = TestRegistry::inst().tests;
  std::cout << "[Sion] running " << tests.size() << " tests\n";
  for (auto &[name, fn] : tests) {
    std::cout << "  - " << name << std::endl;
    fn();
  }
  std::cout << "[Sion] all tests completed, please check the report\n";
  sion::test::write_report("sgemm_report.md");
  return 0;
}