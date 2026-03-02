#!/usr/bin/env bash
# sgemm_bench_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BIN_DIR="${ROOT_DIR}/build/bench/sgemm"
BIN_DIR="${BIN_DIR:-${DEFAULT_BIN_DIR}}"

# 可执行文件路径（可通过环境变量覆盖）
SION_BIN="${SION_BIN:-${BIN_DIR}/sgemm_bench}"
CUBLAS_BIN="${CUBLAS_BIN:-${BIN_DIR}/cublas_bench}"

# 输出目录与汇总文件
OUT_DIR="${OUT_DIR:-${BIN_DIR}/bench_results}"
OUT_MD="${OUT_MD:-${OUT_DIR}/sgemm_compare.md}"
REF_IMPL="${REF_IMPL:-cublas}"
mkdir -p "${OUT_DIR}"

# 基准配置（可通过环境变量覆盖）
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
ITERS="${ITERS:-10}"

# 可整除的 shape 列表 (M/N 按 128, K 按 16 对齐)
SHAPES=(
  "1024x1024x1024"
  "1024x1024x2048"
  "1024x1024x2560"
  "1024x2048x1024"
  "1024x2048x2048"
  "1024x2048x2560"
  "2048x1024x1024"
  "2048x1024x2048"
  "2048x1024x2560"
  "2048x2048x1024"
  "2048x2048x2048"
  "2048x2048x2560"
  "2304x2304x1024"
  "2304x2304x2048"
  "2304x2304x2304"
  "2304x2304x2560"
  "2432x2432x1024"
  "2432x2432x2048"
  "2432x2432x2432"
  "2432x2432x2560"
  "2560x2560x1024"
  "2560x2560x2048"
  "2560x2560x2560"
  "2560x2560x2688"
  "2688x2688x1024"
  "2688x2688x2048"
  "2688x2688x2560"
  "2688x2688x2816"
  "3072x3072x3072"
  "3072x3072x3584"
  "3584x3584x3584"
  "3840x3840x3840"
  "4096x4096x1024"
  "4096x4096x2048"
  "4096x4096x2560"
  "4096x4096x3072"
  "4096x4096x4096"
  "4096x4096x4608"
  "4736x4736x1024"
  "4736x4736x2048"
  "4736x4736x2560"
  "4736x4736x3072"
  "4736x4736x4736"
  "5120x5120x1024"
  "5120x5120x2048"
  "5120x5120x2560"
  "5120x5120x3072"
  "5120x5120x5120"
)

require_bin() {
  local bin="$1"
  if [[ ! -x "${bin}" ]]; then
    echo "[ERROR] executable not found: ${bin}"
    echo "[HINT] build first: cmake --build ${ROOT_DIR}/build --target bench -j"
    exit 1
  fi
}

validate_shape() {
  local shape="$1"
  local m n k
  IFS='x' read -r m n k <<<"${shape}"
  if [[ -z "${m}" || -z "${n}" || -z "${k}" ]]; then
    return 1
  fi
  if (( m % 128 != 0 || n % 128 != 0 || k % 16 != 0 )); then
    return 1
  fi
  return 0
}

extract_col() {
  local file="$1"
  local col="$2"
  awk -F'|' -v c="${col}" 'NR==3 {gsub(/^[ \t]+|[ \t]+$/, "", $c); print $c}' "${file}"
}

calc_ratio() {
  local num="$1"
  local den="$2"
  awk -v n="${num}" -v d="${den}" 'BEGIN {if (d == 0) {print "nan"} else {printf "%.4f", n / d}}'
}

require_bin "${SION_BIN}"
require_bin "${CUBLAS_BIN}"

if [[ "${REF_IMPL}" != "cublas" ]]; then
  echo "[WARN] REF_IMPL=${REF_IMPL} is currently unsupported, fallback to cublas."
  REF_IMPL="cublas"
fi

echo "[INFO] Starting SGEMM compare benchmark..."
echo "[INFO] SION_BIN=${SION_BIN}"
echo "[INFO] CUBLAS_BIN=${CUBLAS_BIN}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] OUT_MD=${OUT_MD}"
echo "[INFO] REF_IMPL=${REF_IMPL}"

{
  echo "# SGEMM Benchmark Compare (ref: cuBLAS)"
  echo
  echo "- Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "- Config: warmup=${WARMUP}, repeat=${REPEAT}, iters=${ITERS}"
  echo "- Sion binary: \`${SION_BIN}\`"
  echo "- cuBLAS binary: \`${CUBLAS_BIN}\`"
  echo
  echo "| Shape | Sion avg_ms | cuBLAS avg_ms (ref) | Sion speedup vs cuBLAS | Sion TFLOPS | cuBLAS TFLOPS | Sion TFLOPS ratio | Winner |"
  echo "|-------|-------------|---------------------|------------------------|-------------|---------------|-------------------|--------|"
} > "${OUT_MD}"

# 遍历每个 shape
for SHAPE in "${SHAPES[@]}"; do
  if ! validate_shape "${SHAPE}"; then
    echo "[ERROR] invalid shape for aligned kernel: ${SHAPE}"
    exit 1
  fi

  SION_TMP="$(mktemp "${OUT_DIR}/.sion_${SHAPE}_XXXXXX.md")"
  CUBLAS_TMP="$(mktemp "${OUT_DIR}/.cublas_${SHAPE}_XXXXXX.md")"

  echo "[INFO] Benchmarking Sion kernel for shape ${SHAPE}..."
  "${SION_BIN}" --shape "${SHAPE}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --iters "${ITERS}" \
    --out "${SION_TMP}"

  echo "[INFO] Benchmarking cuBLAS kernel for shape ${SHAPE}..."
  "${CUBLAS_BIN}" --shape "${SHAPE}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --iters "${ITERS}" \
    --out "${CUBLAS_TMP}"

  SION_AVG_MS="$(extract_col "${SION_TMP}" 8)"
  SION_TFLOPS="$(extract_col "${SION_TMP}" 9)"
  CUBLAS_AVG_MS="$(extract_col "${CUBLAS_TMP}" 8)"
  CUBLAS_TFLOPS="$(extract_col "${CUBLAS_TMP}" 9)"

  SION_SPEEDUP_VS_CUBLAS="$(calc_ratio "${CUBLAS_AVG_MS}" "${SION_AVG_MS}")"
  SION_TFLOPS_RATIO="$(calc_ratio "${SION_TFLOPS}" "${CUBLAS_TFLOPS}")"

  WINNER="tie"
  if awk -v s="${SION_AVG_MS}" -v c="${CUBLAS_AVG_MS}" 'BEGIN{exit !(s < c)}'; then
    WINNER="Sion"
  elif awk -v s="${SION_AVG_MS}" -v c="${CUBLAS_AVG_MS}" 'BEGIN{exit !(s > c)}'; then
    WINNER="cuBLAS"
  fi

  echo "| ${SHAPE} | ${SION_AVG_MS} | ${CUBLAS_AVG_MS} | ${SION_SPEEDUP_VS_CUBLAS}x | ${SION_TFLOPS} | ${CUBLAS_TFLOPS} | ${SION_TFLOPS_RATIO}x | ${WINNER} |" >> "${OUT_MD}"

  rm -f "${SION_TMP}" "${CUBLAS_TMP}"
done

echo
echo "[INFO] Benchmark finished for all shapes."
echo "[INFO] Summary saved to ${OUT_MD}"
