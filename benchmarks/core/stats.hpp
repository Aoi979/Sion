#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cuda_ops::bench {

struct SeriesStats {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double stddev = 0.0;
};

inline double percentile_sorted(const std::vector<double> &values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  if (values.size() == 1) {
    return values.front();
  }
  const double pos = p * static_cast<double>(values.size() - 1);
  const auto lo = static_cast<std::size_t>(std::floor(pos));
  const auto hi = static_cast<std::size_t>(std::ceil(pos));
  const double frac = pos - static_cast<double>(lo);
  return values[lo] * (1.0 - frac) + values[hi] * frac;
}

inline SeriesStats summarize(std::vector<double> values) {
  SeriesStats stats;
  if (values.empty()) {
    return stats;
  }

  std::sort(values.begin(), values.end());
  stats.min = values.front();
  stats.max = values.back();
  stats.mean =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  stats.median = percentile_sorted(values, 0.5);
  stats.p90 = percentile_sorted(values, 0.9);

  double variance = 0.0;
  for (double v : values) {
    const double delta = v - stats.mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(values.size());
  stats.stddev = std::sqrt(variance);
  return stats;
}

} // namespace cuda_ops::bench
