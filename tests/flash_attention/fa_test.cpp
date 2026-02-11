#include "../common.hpp"
#include "../../include/sion/sion.hpp"

torch::Tensor SDPA_ref(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be CUDA tensor");
    auto O = at::scaled_dot_product_attention(
        Q, K, V,
        /*attn_mask=*/c10::nullopt,
        /*dropout_p=*/0.0,
        /*is_causal=*/false,
        /*scale=*/c10::nullopt,
        /*enable_gqa=*/false
    );
    return O;
}

torch::Tensor SDPA_op(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    return sion::flash_attention(Q, K, V);
}

SION_TEST(test_SDPA_basic){
    torch::manual_seed(0);
    torch::Device device(torch::kCUDA);

    constexpr int B = 16;
    constexpr int H = 16;
    constexpr int D = 128;
    constexpr int STAGE = 2;
    int N = 1280;

    auto opt = torch::TensorOptions().device(device).dtype(torch::kFloat16);

    float scale = 1.0f;
    auto Q = scale * torch::randn({B, H, N, D}, opt);
    auto K = scale * torch::randn({B, H, N, D}, opt);
    auto V = scale * torch::randn({B, H, N, D}, opt);
    auto ref = SDPA_ref(Q, K, V);
    auto val = SDPA_op(Q, K, V);
    auto stats = sion::test::compare_tensor(ref, val);
    sion::test::add_record("SDPA_basic", ref.numel(), stats, 1e-2);
}

int main(){
    auto& tests = TestRegistry::inst().tests;
    std::cout << "[Sion] running " << tests.size() << " tests\n";
    for(auto& [name, fn]: tests){
        std::cout << "  - " << name << std::endl;
        fn();
    }
    std::cout << "[Sion] all tests completed, please check the report\n";
    sion::test::write_report("SDPA_report.md");
    return 0;
}