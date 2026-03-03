#!/usr/bin/env bash
# sgemm_kernel_compare_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BIN_DIR="${ROOT_DIR}/build/bench/sgemm"
BIN_DIR="${BIN_DIR:-${DEFAULT_BIN_DIR}}"

SION_BIN="${SION_BIN:-${BIN_DIR}/sgemm_bench}"
OUT_DIR="${OUT_DIR:-${BIN_DIR}/bench_results}"
OUT_MD="${OUT_MD:-${OUT_DIR}/sgemm_kernel_compare.md}"

KERNEL_A="${KERNEL_A:-cute_sgemm_64x64_nn}"
KERNEL_B="${KERNEL_B:-cute_sgemm_64x64_nn_swizzle}"

WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
ITERS="${ITERS:-10}"

# For these two kernels, use M/N % 64 == 0 and K % 8 == 0
SHAPES=(
  "960x960x960"
  "960x1536x960"
  "1024x1024x1024"
  "1536x1536x1536"
  "2048x1024x2048"
  "2048x2048x1024"
  "2048x2048x2048"
  "4096x4096x1024"
  "4096x4096x4096"
)

mkdir -p "${OUT_DIR}"

if [[ ! -x "${SION_BIN}" ]]; then
  echo "[ERROR] executable not found: ${SION_BIN}"
  echo "[HINT] build first: cmake --build ${ROOT_DIR}/build --target sgemm_bench -j"
  exit 1
fi

validate_shape() {
  local shape="$1"
  local m n k
  IFS='x' read -r m n k <<<"${shape}"
  if [[ -z "${m}" || -z "${n}" || -z "${k}" ]]; then
    return 1
  fi
  if (( m % 64 != 0 || n % 64 != 0 || k % 8 != 0 )); then
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

{
  echo "# SGEMM Internal Kernel Compare"
  echo
  echo "- Generated: $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "- Kernel A: ${KERNEL_A}"
  echo "- Kernel B: ${KERNEL_B}"
  echo "- Config: warmup=${WARMUP}, repeat=${REPEAT}, iters=${ITERS}"
  echo "- Binary: \`${SION_BIN}\`"
  echo
  echo "| Shape | Kernel A | Kernel B | A avg_ms | B avg_ms | A speedup vs B | A TFLOPS | B TFLOPS | A TFLOPS ratio | Winner |"
  echo "|-------|----------|----------|----------|----------|----------------|----------|----------|----------------|--------|"
} > "${OUT_MD}"

echo "[INFO] Starting SGEMM internal kernel compare..."
echo "[INFO] SION_BIN=${SION_BIN}"
echo "[INFO] KERNEL_A=${KERNEL_A}"
echo "[INFO] KERNEL_B=${KERNEL_B}"
echo "[INFO] OUT_MD=${OUT_MD}"

for shape in "${SHAPES[@]}"; do
  if ! validate_shape "${shape}"; then
    echo "[ERROR] invalid shape: ${shape}"
    echo "[HINT] require M/N % 64 == 0 and K % 8 == 0"
    exit 1
  fi

  A_TMP="$(mktemp "${OUT_DIR}/.kernel_a_${shape}_XXXXXX.md")"
  B_TMP="$(mktemp "${OUT_DIR}/.kernel_b_${shape}_XXXXXX.md")"

  echo "[INFO] shape=${shape} kernel=${KERNEL_A}"
  "${SION_BIN}" --shape "${shape}" \
    --kernel "${KERNEL_A}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --iters "${ITERS}" \
    --out "${A_TMP}"

  echo "[INFO] shape=${shape} kernel=${KERNEL_B}"
  "${SION_BIN}" --shape "${shape}" \
    --kernel "${KERNEL_B}" \
    --warmup "${WARMUP}" --repeat "${REPEAT}" --iters "${ITERS}" \
    --out "${B_TMP}"

  A_AVG_MS="$(extract_col "${A_TMP}" 8)"
  B_AVG_MS="$(extract_col "${B_TMP}" 8)"
  A_TFLOPS="$(extract_col "${A_TMP}" 9)"
  B_TFLOPS="$(extract_col "${B_TMP}" 9)"

  A_SPEEDUP_VS_B="$(calc_ratio "${B_AVG_MS}" "${A_AVG_MS}")"
  A_TFLOPS_RATIO="$(calc_ratio "${A_TFLOPS}" "${B_TFLOPS}")"

  WINNER="tie"
  if awk -v a="${A_AVG_MS}" -v b="${B_AVG_MS}" 'BEGIN{exit !(a < b)}'; then
    WINNER="Kernel A"
  elif awk -v a="${A_AVG_MS}" -v b="${B_AVG_MS}" 'BEGIN{exit !(a > b)}'; then
    WINNER="Kernel B"
  fi

  echo "| ${shape} | ${KERNEL_A} | ${KERNEL_B} | ${A_AVG_MS} | ${B_AVG_MS} | ${A_SPEEDUP_VS_B}x | ${A_TFLOPS} | ${B_TFLOPS} | ${A_TFLOPS_RATIO}x | ${WINNER} |" >> "${OUT_MD}"

  rm -f "${A_TMP}" "${B_TMP}"
done

echo "[INFO] Done. Summary saved to ${OUT_MD}"
