#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment

ROOT_DIR=${SCRIPT_DIR}
EXEC=${ROOT_DIR}/build/benchmark_gemm
RESULT_DIR=${GEMM_RESULT_DIR:-${ROOT_DIR}/results/fp32/gemm}
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "GEMM shape benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Total matrix size  : 32768"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

echo "" | tee -a "${SUMMARY_LOG}"
echo "Starting right-looking GEMM shape sweep..." | tee -a "${SUMMARY_LOG}"

"${LAUNCH_PREFIX[@]}" "${EXEC}" 2>&1 | tee -a "${SUMMARY_LOG}"

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "GEMM shape benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
