#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment

ROOT_DIR=${SCRIPT_DIR}
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_syrk
RESULT_DIR=${SYRK_RESULT_DIR:-${ROOT_DIR}/results/fp32/syrk}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${SYRK_NS:-8192 16384 24576 32768 40960 49152 57344 65536}"
read -r -a NBS <<< "${SYRK_NBS:-256 512 1024 2048}"
read -r -a K_TILES <<< "${SYRK_K_TILES:-8192}"

USE_EQUAL_K=${SYRK_USE_EQUAL_K:-1}
FIXED_K=${SYRK_FIXED_K:-8192}
ERROR_MODE=${SYRK_ERROR_MODE:-2}
TEST_RECURSIVE=${SYRK_TEST_RECURSIVE:-1}
TEST_ROCBLAS=${SYRK_TEST_ROCBLAS:-1}
TEST_BATCHED=${SYRK_TEST_BATCHED:-1}
TEST_MIXED=${SYRK_TEST_MIXED:-0}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "SYRK FP32 benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "K mode             : equal-N=${USE_EQUAL_K}, fixed=${FIXED_K}"
    echo "Algorithm NBs      : ${NBS[*]}"
    echo "K-blocked tiles    : ${K_TILES[*]}"
    echo "Error verify mode  : ${ERROR_MODE}"
    echo "Test rocBLAS       : ${TEST_ROCBLAS}"
    echo "Test Recursive     : ${TEST_RECURSIVE}"
    echo "Test Batched       : ${TEST_BATCHED}"
    echo "Test Mixed         : ${TEST_MIXED}"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"; do
    if [ "${USE_EQUAL_K}" -eq 1 ]; then
        K=${N}
    else
        K=${FIXED_K}
    fi

    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> Running SYRK Benchmark: N=${N}, K=${K}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/syrk_N${N}_K${K}.log
    echo "Output detail      : ${OUT_FILE}" | tee -a "${SUMMARY_LOG}"

    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        "${K}" \
        "${ERROR_MODE}" \
        "${TEST_RECURSIVE}" \
        "${TEST_ROCBLAS}" \
        "${TEST_BATCHED}" \
        "${TEST_MIXED}" \
        "${NBS[@]}" \
        --k-tile "${K_TILES[@]}" \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "SYRK benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
