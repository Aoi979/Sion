#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <string>

namespace sion::bench {

struct BenchmarkStats {
    double min_ms = 0.0;
    double max_ms = 0.0;
    double avg_ms = 0.0;
    double throughput = 0.0; 
};


struct BenchEntry {
    std::string name;
    std::function<void()> fn;
};

struct BenchmarkRegistry {
    std::vector<BenchEntry> benches;
    static BenchmarkRegistry& inst(){ static BenchmarkRegistry r; return r; }
};


#define SION_BENCH(name) \
    void name(); \
    static struct { \
        struct Dummy{ Dummy(){ sion::bench::BenchmarkRegistry::inst().benches.push_back({#name,name}); } }; \
        Dummy d; \
    } _sion_bench_##name; \
    void name()


inline BenchmarkStats run_benchmark(std::function<void()> fn, int warmup=2, int repeat=10){
    for(int i=0;i<warmup;i++) fn();

    std::vector<double> times;
    for(int i=0;i<repeat;i++){
        auto start = std::chrono::high_resolution_clock::now();
        fn();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(end-start).count();
        times.push_back(ms);
    }

    BenchmarkStats stats;
    stats.min_ms = *std::min_element(times.begin(), times.end());
    stats.max_ms = *std::max_element(times.begin(), times.end());
    stats.avg_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();

    return stats;
}


inline void print_stats_md_file(const BenchmarkStats& stats,
                                const std::string& bench_name,
                                const std::string& device,
                                const std::string& dtype,
                                const std::string& shape,
                                const std::string& filename="sion_bench.md",
                                bool header=false)
{
    std::ofstream out(filename, header ? std::ios::trunc : std::ios::app);
    if(header){
        out << "| Bench Name | Device | DType | Shape | min_ms | max_ms | avg_ms |\n";
        out << "|------------|--------|-------|-------|--------|--------|--------|\n";
    }

    out << "| " << bench_name
        << " | " << device
        << " | " << dtype
        << " | " << shape
        << " | " << stats.min_ms
        << " | " << stats.max_ms
        << " | " << stats.avg_ms
        << " |\n";
}

} // namespace sion::bench
