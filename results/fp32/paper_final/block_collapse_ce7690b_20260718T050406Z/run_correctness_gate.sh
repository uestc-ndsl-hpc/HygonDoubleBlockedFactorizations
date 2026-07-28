#!/usr/bin/env bash

set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 ROOT_DIR RESULT_DIR" >&2
    exit 2
fi

ROOT_DIR=$1
RESULT_DIR=$2
SUMMARY_LOG=${RESULT_DIR}/manifest.log
POTRF_EXEC=${ROOT_DIR}/build/benchmark_potrf
GETRF_EXEC=${ROOT_DIR}/build/benchmark_getrf
N=16384
NB=8192

export HIP_VISIBLE_DEVICES=1
export BENCHMARK_WARMUP_ITERS=0
export BENCHMARK_PROFILE_ITERS=1

source "${ROOT_DIR}/scripts/benchmark_env.sh"
setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

mkdir -p "${RESULT_DIR}"
{
    echo "Block-collapse correctness gate"
    echo "N                   : ${N}"
    echo "POTRF B/b           : ${NB}/${NB}"
    echo "GETRF B/b           : ${NB}/${NB}"
    echo "Metric              : full-matrix normwise backward error"
    echo "Residual product    : factorization precision; FP64 norm accumulation"
} | tee "${SUMMARY_LOG}"

log_toolchain "${SUMMARY_LOG}" "${ROOT_DIR}"
check_benchmark_binary \
    "${POTRF_EXEC}" "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
check_benchmark_binary \
    "${GETRF_EXEC}" "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

export BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR=1e-5
echo ">>> POTRF b=nb correctness gate" | tee -a "${SUMMARY_LOG}"
BENCHMARK_CSV_PATH="${RESULT_DIR}/potrf_b_eq_nb_N${N}_timings.csv" \
    "${LAUNCH_PREFIX[@]}" "${POTRF_EXEC}" \
    "${N}" 0 0 2 0 0 0 2048 \
    --methods mgemm \
    --double-nb "${NB}" \
    --b "${NB}" \
    --syrk-k-tile 8192 \
    --gemm-m-tile 8192 \
    --gemm-m-threshold 32768 \
    --gemm-k-threshold 5632 \
    --full-backward-error \
    --full-backward-error-max-n "${N}" \
    2>&1 | tee "${RESULT_DIR}/potrf_b_eq_nb_N${N}.log"

export BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR=5e-4
echo ">>> GETRF b=nb correctness gate" | tee -a "${SUMMARY_LOG}"
BENCHMARK_CSV_PATH="${RESULT_DIR}/getrf_b_eq_nb_N${N}_timings.csv" \
    "${LAUNCH_PREFIX[@]}" "${GETRF_EXEC}" \
    "${N}" 0 1 2048 \
    --pivot \
    --methods mgemm \
    --double-nb "${NB}" \
    --b "${NB}" \
    --outer-trsm iterative \
    --trsm-block 2048 \
    --pivot-tile 2048 \
    --inner-gemm-m-tile 8192 \
    --full-backward-error \
    --full-backward-error-max-n "${N}" \
    2>&1 | tee "${RESULT_DIR}/getrf_b_eq_nb_N${N}.log"

echo "Block-collapse correctness gate completed" | tee -a "${SUMMARY_LOG}"
