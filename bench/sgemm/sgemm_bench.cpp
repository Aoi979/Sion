#include "../common.hpp"
#include <torch/torch.h>

int main(){
    auto& benches = sion::bench::BenchmarkRegistry::inst().benches;
    std::cout << "[Sion] running " << benches.size() << " benchmarks\n";
    for(auto& [name, fn]: benches){
        std::cout << "  - " << name << std::endl;
        fn();
    }
    std::cout << "[Sion] benchmarks finished, results in sion_bench.md\n";
    return 0;
}
