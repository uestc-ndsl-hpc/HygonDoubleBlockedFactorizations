#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

ROOT_DIR=${SCRIPT_DIR}
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_getrf
RESULT_DIR=${GETRF_RESULT_DIR:-${ROOT_DIR}/results/fp32/getrf_npvt}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${GETRF_NS:-4096 8192 16384 24576 32768 40960}"
read -r -a BASELINE_NBS <<< "${GETRF_BASELINE_NBS:-1024 2048}"
read -r -a DOUBLE_NBS <<< "${GETRF_DOUBLE_NBS:-8192}"
read -r -a MICRO_BLOCKS <<< "${GETRF_MICRO_BLOCKS:-1024 2048}"

PRECISION=${GETRF_PRECISION:-0}
VERIFY=${GETRF_VERIFY:-0}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "GETRF-NPVT benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "Precision mode     : ${PRECISION} (0=FP32, 1=FP64)"
    echo "Backward verify    : ${VERIFY}"
    echo "Blocked-RL NBs     : ${BASELINE_NBS[*]}"
    echo "DoubleBlock NB/B   : ${DOUBLE_NBS[*]} / ${MICRO_BLOCKS[*]}"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"; do
    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> Running GETRF-NPVT benchmark: N=${N}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/getrf_N${N}.log
    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        "${PRECISION}" \
        "${VERIFY}" \
        "${BASELINE_NBS[@]}" \
        --double-nb "${DOUBLE_NBS[@]}" \
        --b "${MICRO_BLOCKS[@]}" \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "GETRF-NPVT benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
