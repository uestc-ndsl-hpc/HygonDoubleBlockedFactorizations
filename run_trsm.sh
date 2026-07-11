#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment

ROOT_DIR=${SCRIPT_DIR}
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_trsm
RESULT_DIR=${TRSM_RESULT_DIR:-${ROOT_DIR}/results/fp32/trsm}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${TRSM_NS:-8192 16384 24576 32768 40960 49152 57344 65536}"
read -r -a NBS <<< "${TRSM_NBS:-256 512 1024 2048}"
read -r -a DOUBLE_NBS <<< "${TRSM_DOUBLE_NBS:-8192}"
read -r -a MICRO_BLOCKS <<< "${TRSM_MICRO_BLOCKS:-1024}"

USE_EQUAL_M=${TRSM_USE_EQUAL_M:-1}
FIXED_M=${TRSM_FIXED_M:-8192}
ERROR_MODE=${TRSM_ERROR_MODE:-2}
TEST_RECURSIVE=${TRSM_TEST_RECURSIVE:-1}
TEST_ROCBLAS=${TRSM_TEST_ROCBLAS:-1}
TEST_ITERATIVE=${TRSM_TEST_ITERATIVE:-1}
TEST_HALF=${TRSM_TEST_HALF:-0}
TEST_DOUBLE_BLOCKED=${TRSM_TEST_DOUBLE_BLOCKED:-1}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "TRSM FP32 benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "M mode             : equal-N=${USE_EQUAL_M}, fixed=${FIXED_M}"
    echo "Algorithm NBs      : ${NBS[*]}"
    echo "DoubleBlock NB/B   : ${DOUBLE_NBS[*]} / ${MICRO_BLOCKS[*]}"
    echo "Error verify mode  : ${ERROR_MODE}"
    echo "Test rocBLAS       : ${TEST_ROCBLAS}"
    echo "Test Recursive     : ${TEST_RECURSIVE}"
    echo "Test Iterative     : ${TEST_ITERATIVE}"
    echo "Test DoubleBlock   : ${TEST_DOUBLE_BLOCKED}"
    echo "Test Mixed         : ${TEST_HALF}"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"; do
    if [ "${USE_EQUAL_M}" -eq 1 ]; then
        M=${N}
    else
        M=${FIXED_M}
    fi

    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> Running TRSM Benchmark: N=${N}, M=${M}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/trsm_N${N}_M${M}.log
    echo "Output detail      : ${OUT_FILE}" | tee -a "${SUMMARY_LOG}"

    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        "${M}" \
        "${ERROR_MODE}" \
        "${TEST_RECURSIVE}" \
        "${TEST_ROCBLAS}" \
        "${TEST_ITERATIVE}" \
        "${TEST_HALF}" \
        "${TEST_DOUBLE_BLOCKED}" \
        "${NBS[@]}" \
        --double-nb "${DOUBLE_NBS[@]}" \
        --b "${MICRO_BLOCKS[@]}" \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "TRSM benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
