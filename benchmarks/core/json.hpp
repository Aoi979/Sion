#pragma once

#include "cuda_utils.hpp"
#include "stats.hpp"
#include "timer.hpp"

#include <ostream>
#include <sstream>
#include <string>

namespace sion::bench {

inline std::string json_escape(const std::string &s) {
  std::ostringstream out;
  for (char c : s) {
    switch (c) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    default:
      out << c;
      break;
    }
  }
  return out.str();
}

inline void write_stats(std::ostream &out, const SeriesStats &stats) {
  out << "{"
      << "\"min\":" << stats.min << ","
      << "\"max\":" << stats.max << ","
      << "\"mean\":" << stats.mean << ","
      << "\"median\":" << stats.median << ","
      << "\"p90\":" << stats.p90 << ","
      << "\"stddev\":" << stats.stddev << "}";
}

inline void write_result_json(std::ostream &out, const std::string &op,
                              const std::string &layer,
                              const std::string &kernel,
                              const std::string &shape,
                              const DeviceInfo &device,
                              const TimingConfig &cfg,
                              const TimingResult &result,
                              double work_units) {
  out << "{\n";
  out << "  \"op\":\"" << json_escape(op) << "\",\n";
  out << "  \"layer\":\"" << json_escape(layer) << "\",\n";
  out << "  \"kernel\":\"" << json_escape(kernel) << "\",\n";
  out << "  \"shape\":\"" << json_escape(shape) << "\",\n";
  out << "  \"device\":{";
  out << "\"ordinal\":" << device.ordinal << ",";
  out << "\"name\":\"" << json_escape(device.name) << "\",";
  out << "\"cc\":" << device.cc << ",";
  out << "\"sm_count\":" << device.sm_count << ",";
  out << "\"max_dynamic_smem\":" << device.max_dynamic_smem << ",";
  out << "\"max_threads_per_block\":" << device.max_threads_per_block;
  out << "},\n";
  out << "  \"config\":{";
  out << "\"warmup\":" << cfg.warmup << ",";
  out << "\"repeat\":" << cfg.repeat << ",";
  out << "\"iters\":" << result.iters << ",";
  out << "\"min_sample_ms\":" << cfg.min_sample_ms;
  out << "},\n";
  out << "  \"timing\":{";
  out << "\"gpu_ms\":";
  write_stats(out, result.gpu_ms);
  out << ",\"host_issue_us\":";
  write_stats(out, result.host_issue_us);
  out << ",\"e2e_us\":";
  write_stats(out, result.e2e_us);
  out << "},\n";
  out << "  \"work_units\":" << work_units << ",\n";
  out << "  \"throughput_tunits_per_s\":"
      << (result.gpu_ms.median > 0.0
              ? work_units / (result.gpu_ms.median * 1.0e9)
              : 0.0)
      << "\n";
  out << "}\n";
}

} // namespace sion::bench
