#include <torch/torch.h>

namespace sion {
template<int HEAD_DIM, int STAGE>
void launch_flash_attn_mma_stages(const torch::Tensor &Q, const torch::Tensor &K,
                                  const torch::Tensor &V, torch::Tensor &O);

template<int HEAD_DIM, int STAGE>
void launch_flash_attn_mma_stages_1D(const torch::Tensor &Q,const torch::Tensor &K,
                                  const torch::Tensor &V, torch::Tensor &O);
                                  
    torch::Tensor flash_attention(const torch::Tensor &query, const torch::Tensor &key, const torch::Tensor &value) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto N = query.size(2);
        auto D = query.size(3);
        auto opt = query.options();
        auto O = torch::empty({B, H, N, D}, opt);
        switch (D) {
            case 64:
                launch_flash_attn_mma_stages_1D<64, 2>(query, key, value, O);
                break;
            case 128:
                launch_flash_attn_mma_stages_1D<128, 2>(query, key, value, O);
                break;
            default:
                TORCH_CHECK(false, "flash_attention: unsupported head dimension ", D);
        
        }
        return O;
    };

}