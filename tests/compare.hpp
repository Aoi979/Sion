#pragma once
#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

namespace sion::test {

struct ErrorStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    double max_rel = 0.0;
    double mean_rel = 0.0;
    int64_t nan_count = 0;
    int64_t inf_count = 0;
    int64_t subnormal_count = 0;
    double rms_error = 0.0;
    double quantile_50 = 0.0;
    double quantile_90 = 0.0;
    double quantile_99 = 0.0;

    // 张量概览
    std::string tensor_preview_ref;
    std::string tensor_preview_val;
};

inline std::string tensor_preview(const torch::Tensor& t, int64_t edgeitems=3) {
    std::ostringstream ss;
    auto dims = t.sizes();
    auto dtype = t.dtype();
    auto device = t.device();

    ss << "tensor(";
    if (t.dim() == 0) {
        ss << t.item<double>();
    } else if (t.dim() == 1) {
        int64_t n = dims[0];
        ss << "[";
        for(int64_t i=0;i<std::min(edgeitems,n);i++) ss << t[i].item<double>() << ", ";
        if(n > 2*edgeitems) ss << "... , ";
        for(int64_t i=std::max(edgeitems,int64_t(0));i<n;i++) if(i>=n-edgeitems) ss << t[i].item<double>() << (i<n-1?", ":"");
        ss << "]";
    } else if (t.dim() == 2) {
        int64_t rows = dims[0], cols = dims[1];
        ss << "[";
        for(int64_t i=0;i<std::min(edgeitems,rows);i++){
            ss << "[";
            for(int64_t j=0;j<std::min(edgeitems,cols);j++) ss << t[i][j].item<double>() << ", ";
            if(cols > 2*edgeitems) ss << "... , ";
            for(int64_t j=std::max(edgeitems,int64_t(0));j<cols;j++) if(j>=cols-edgeitems) ss << t[i][j].item<double>() << (j<cols-1?", ":"");
            ss << "]";
            if(i<rows-1) ss << ",\n ";
        }
        if(rows > 2*edgeitems) ss << ",\n ...\n ";
        for(int64_t i=std::max(edgeitems,int64_t(0));i<rows;i++) if(i>=rows-edgeitems){
            ss << "[";
            for(int64_t j=0;j<std::min(edgeitems,cols);j++) ss << t[i][j].item<double>() << ", ";
            if(cols > 2*edgeitems) ss << "... , ";
            for(int64_t j=std::max(edgeitems,int64_t(0));j<cols;j++) if(j>=cols-edgeitems) ss << t[i][j].item<double>() << (j<cols-1?", ":"");
            ss << "]";
            if(i<rows-1) ss << ",\n ";
        }
        ss << "]";
    } else {
        ss << "... (dim=" << t.dim() << ")";
    }
    ss << ", dtype=" << dtype << ", device=" << device << ")";
    return ss.str();
}

inline bool is_subnormal(double v) { return v != 0.0 && std::abs(v) < std::numeric_limits<float>::min(); }


inline ErrorStats compare_tensor(const torch::Tensor& ref, const torch::Tensor& val, double eps=1e-6) {
    TORCH_CHECK(ref.sizes() == val.sizes(), "Tensor size mismatch");

    auto ref_flat = ref.flatten().to(torch::kCPU).to(torch::kFloat64);
    auto val_flat = val.flatten().to(torch::kCPU).to(torch::kFloat64);

    ErrorStats stats;

    stats.tensor_preview_ref = tensor_preview(ref);
    stats.tensor_preview_val = tensor_preview(val);

    std::vector<double> abs_errors;
    auto n = ref_flat.numel();
    for(int64_t i=0;i<n;i++){
        double r = ref_flat[i].item<double>();
        double v = val_flat[i].item<double>();
        double abs_err = std::abs(r - v);
        double rel_err = (std::abs(r) > eps) ? abs_err / std::abs(r) : abs_err;

        stats.max_abs = std::max(stats.max_abs, abs_err);
        stats.max_rel = std::max(stats.max_rel, rel_err);
        stats.mean_abs += abs_err;
        stats.mean_rel += rel_err;
        stats.rms_error += abs_err * abs_err;
        abs_errors.push_back(abs_err);

        if(std::isnan(v)) stats.nan_count++;
        if(std::isinf(v)) stats.inf_count++;
        if(is_subnormal(v)) stats.subnormal_count++;
    }

    stats.mean_abs /= n;
    stats.mean_rel /= n;
    stats.rms_error = std::sqrt(stats.rms_error / n);

    std::sort(abs_errors.begin(), abs_errors.end());
    auto q = [&](double p){
        size_t idx = static_cast<size_t>(p * n);
        if(idx >= n) idx = n-1;
        return abs_errors[idx];
    };
    stats.quantile_50 = q(0.5);
    stats.quantile_90 = q(0.9);
    stats.quantile_99 = q(0.99);

    return stats;
}


inline bool check_pass(const ErrorStats& stats, double tol=1e-3){
    return stats.max_abs <= tol && stats.nan_count == 0 && stats.inf_count == 0;
}

inline void print_stats(const ErrorStats& stats, const std::string& name, int64_t n){
    std::cout << "[TEST] " << name << " n=" << n
              << " max_abs=" << stats.max_abs
              << " mean_abs=" << stats.mean_abs
              << " max_rel=" << stats.max_rel
              << " mean_rel=" << stats.mean_rel
              << " rms=" << stats.rms_error
              << " nan=" << stats.nan_count
              << " inf=" << stats.inf_count
              << " subnormal=" << stats.subnormal_count
              << " q50=" << stats.quantile_50
              << " q90=" << stats.quantile_90
              << " q99=" << stats.quantile_99
              << std::endl;

    std::cout << "Ref Tensor Preview:\n" << stats.tensor_preview_ref << std::endl;
    std::cout << "Val Tensor Preview:\n" << stats.tensor_preview_val << std::endl;
}

inline void print_stats_md_file(const ErrorStats& stats, const std::string& test_name, int64_t n,
                                double tol=1e-3, const std::string& filename="sion_report.md", bool header=false)
{
    std::ofstream out(filename, header ? std::ios::trunc : std::ios::app);
    if(!out) return;

    // 表格头
    if(header){
        out << "| Test Name | n | max_abs | mean_abs | max_rel | mean_rel | rms | NaN | Inf | Subnormal | q50 | q90 | q99 | Pass |\n";
        out << "|:---------|:--:|--------:|---------:|--------:|---------:|----:|:--:|:--:|----------:|:--:|:--:|:--:|:----:|\n";
    }

    // 统计信息表格
    out << "| " << test_name
        << " | " << n
        << " | " << stats.max_abs
        << " | " << stats.mean_abs
        << " | " << stats.max_rel
        << " | " << stats.mean_rel
        << " | " << stats.rms_error
        << " | " << stats.nan_count
        << " | " << stats.inf_count
        << " | " << stats.subnormal_count
        << " | " << stats.quantile_50
        << " | " << stats.quantile_90
        << " | " << stats.quantile_99
        << " | " << (check_pass(stats, tol) ? "PASS" : "FAIL")
        << " |\n\n";

    out << "> **Ref Tensor Preview**\n>\n";
    out << "> ```text\n";
    std::istringstream ref_ss(stats.tensor_preview_ref);
    std::string line;
    while(std::getline(ref_ss, line)) out << "> " << line << "\n";
    out << "> ```\n>\n";

    out << "> **Val Tensor Preview**\n>\n";
    out << "> ```text\n";
    std::istringstream val_ss(stats.tensor_preview_val);
    while(std::getline(val_ss, line)) out << "> " << line << "\n";
    out << "> ```\n\n";
}

} // namespace sion::test
