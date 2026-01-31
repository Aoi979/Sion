#include <torch/torch.h>

namespace sion {

    torch::Tensor flash_attention(torch::Tensor &query, torch::Tensor &key, torch::Tensor &value) {
        auto B = query.size(0);
        auto H = query.size(1);
        auto N = query.size(2);
        auto D = query.size(3);
        auto opt = query.options();

        auto O = torch::empty({B, H, N, D}, opt);
        return O;
    };

}