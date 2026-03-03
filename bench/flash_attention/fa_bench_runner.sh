#!/usr/bin/env bash
# fa_bench_runner.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BIN_DIR="${ROOT_DIR}/build/bench/flash_attention"
BIN_DIR="${BIN_DIR:-${DEFAULT_BIN_DIR}}"

FA_BIN="${FA_BIN:-${BIN_DIR}/fa_bench}"
OUT_DIR="${OUT_DIR:-${BIN_DIR}/bench_results}"
OUT_MD="${OUT_MD:-${OUT_DIR}/fa_compare.md}"
REF_IMPL="${REF_IMPL:-libtorch_sdpa}"
KERNEL_NAME="${KERNEL_NAME:-ampere_flash_attn_mma16168_64_1D_warp_tiling}"

WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-20}"
ITERS="${ITERS:-10}"
SHAPES=(
  # Short sequence, D=64
  "2x8x512x64"     # 小 batch，标准头数
  "4x16x512x64"    # 中等 batch，多头
  "8x32x512x64"    # 大 batch，很多头

  # Mid sequence, D=64
  "2x8x1024x64"    # 小 batch，长序列
  "4x16x1024x64"   # 中等 batch
  "16x32x1024x64"  # 很大 batch + 多头

  # Long sequence, D=64
  "4x8x2048x64"    # 中 batch，长序列
  "8x16x2048x64"   # 中大 batch
  "32x32x2048x64"  # 超大 batch

  # Extra long, D=64
  "8x8x3072x64"    # 长序列
  "16x16x3072x64"  # 大 batch + 多头

  # Short sequence, D=128
  "4x8x512x128"
  "8x16x512x128"

  # Mid sequence, D=128
  "4x8x1024x128"
  "16x16x1024x128"  # 大 batch + 多头

  # Long sequence, D=128
  "8x8x2048x128"
  "16x16x2048x128"

  # Extra long, D=128
  "8x16x3072x128"
  "32x32x3072x128"  # 最大 batch + 多头
)
mkdir -p "${OUT_DIR}"

if [[ ! -x "${FA_BIN}" ]]; then
  echo "[ERROR] executable not found: ${FA_BIN}"
  echo "[HINT] build first: cmake --build ${ROOT_DIR}/build --target fa_bench -j"
  exit 1
fi

validate_shape() {
  local shape="$1"
  local b h n d
  IFS='x' read -r b h n d <<<"${shape}"
  if [[ -z "${b}" || -z "${h}" || -z "${n}" || -z "${d}" ]]; then
    return 1
  fi
  if (( b <= 0 || h <= 0 || n <= 0 || d <= 0 )); then
    return 1
  fi
  if (( n % 64 != 0 )); then
    return 1
  fi
  if (( d != 64 && d != 128 )); then
    return 1
  fi
  return 0
}

ARGS=(
  --ref "${REF_IMPL}"
  --kernel "${KERNEL_NAME}"
  --warmup "${WARMUP}"
  --repeat "${REPEAT}"
  --iters "${ITERS}"
  --out "${OUT_MD}"
)

for shape in "${SHAPES[@]}"; do
  if ! validate_shape "${shape}"; then
    echo "[ERROR] invalid shape: ${shape}"
    echo "[HINT] require B/H/N/D > 0, N % 64 == 0, D in {64,128}"
    exit 1
  fi
  ARGS+=(--shape "${shape}")
done

echo "[INFO] Running flash attention benchmark compare..."
echo "[INFO] FA_BIN=${FA_BIN}"
echo "[INFO] REF_IMPL=${REF_IMPL}"
echo "[INFO] KERNEL_NAME=${KERNEL_NAME}"
echo "[INFO] OUT_MD=${OUT_MD}"
echo "[INFO] NUM_SHAPES=${#SHAPES[@]}"

"${FA_BIN}" "${ARGS[@]}"

echo "[INFO] Done. Summary: ${OUT_MD}"
