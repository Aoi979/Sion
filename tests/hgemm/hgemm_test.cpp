#include "../../include/sion/sion.hpp"
#include "../common.hpp"
torch::Tensor hgemm_ref(const torch::Tensor &A, const torch::Tensor &B,
                        float alpha, float beta) {
  TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
  auto A_ = A.to(torch::kFloat16);
  auto B_ = B.to(torch::kFloat16);

  int64_t m = A_.size(0);
  int64_t n = B_.size(1);

  auto C_ = torch::zeros({m, n}, A_.options());

  auto result = torch::addmm(C_, // self
                             A_, // mat1
                             B_, // mat2
                             beta, alpha);

  return result.to(A.dtype());
}

torch::Tensor hgemm_op(const torch::Tensor &A, const torch::Tensor &B, float alpha, float beta) {
  const int64_t M = A.size(0);
  const int64_t K = A.size(1);
  const int64_t N = B.size(1);
  return sion::hgemm(A, B, alpha, beta);
}

SION_TEST(test_sgemm_basic0) {
  int M = 2048, K = 2048, N = 2048;

  SION_CHECK(torch::cuda::is_available());

  auto opts =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);

  torch::Tensor A = torch::rand({M, K}, opts);
  torch::Tensor B = torch::rand({K, N}, opts);

  auto ref = hgemm_ref(A, B, 1, 0);
  auto val = hgemm_op(A, B, 1, 0);
  auto stats = sion::test::compare_tensor(ref, val);
  sion::test::add_record("hgemm_basic0", ref.numel(), stats, 100);
}

int main() {
  auto &tests = TestRegistry::inst().tests;
  std::cout << "[Sion] running " << tests.size() << " tests\n";
  for (auto &[name, fn] : tests) {
    std::cout << "  - " << name << std::endl;
    fn();
  }
  std::cout << "[Sion] all tests completed, please check the report\n";
  sion::test::write_report("hgemm_tc_report.md");
  return 0;
}