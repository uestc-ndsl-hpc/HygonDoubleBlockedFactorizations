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
EXEC=${BUILD_DIR}/benchmark_potrf
RESULT_DIR=${POTRF_RESULT_DIR:-${ROOT_DIR}/results/fp32/potrf}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${POTRF_NS:-8192 16384 24576 32768 40960}"
read -r -a BASELINE_NBS <<< "${POTRF_BASELINE_NBS:-256 512 1024 2048}"
read -r -a DOUBLE_NBS <<< "${POTRF_DOUBLE_NBS:-8192}"
read -r -a MICRO_BLOCKS <<< "${POTRF_MICRO_BLOCKS:-512}"

ERROR_MODE=${POTRF_ERROR_MODE:-2}
PRECISION=${POTRF_PRECISION:-0}
TEST_ROCSOLVER=${POTRF_TEST_ROCSOLVER:-1}
TEST_RECURSIVE=${POTRF_TEST_RECURSIVE:-1}
TEST_BLOCKED=${POTRF_TEST_BLOCKED:-1}
TEST_DOUBLE_BLOCKED=${POTRF_TEST_DOUBLE_BLOCKED:-1}
TEST_DOUBLE_LEFT_LOOKING=${POTRF_TEST_DOUBLE_LEFT_LOOKING:-0}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "POTRF benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "Precision mode     : ${PRECISION} (0=FP32, 1=FP64)"
    echo "Error verify mode  : ${ERROR_MODE}"
    echo "Baseline NBs       : ${BASELINE_NBS[*]}"
    echo "DoubleBlock NB/B   : ${DOUBLE_NBS[*]} / ${MICRO_BLOCKS[*]}"
    echo "DoubleBlock methods: base, DBTRSM, KBSYRK, KBSYRK+MGEMM"
    echo "Test rocSOLVER     : ${TEST_ROCSOLVER}"
    echo "Test Blocked-RL    : ${TEST_BLOCKED}"
    echo "Test DoubleBlock   : ${TEST_DOUBLE_BLOCKED}"
    echo "Test DoubleBlock LL: ${TEST_DOUBLE_LEFT_LOOKING}"
    echo "Test Recursive     : ${TEST_RECURSIVE}"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"; do
    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> Running POTRF Benchmark: N=${N}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/potrf_N${N}.log
    echo "Output detail      : ${OUT_FILE}" | tee -a "${SUMMARY_LOG}"

    EXTRA_ARGS=()
    if [[ "${TEST_DOUBLE_LEFT_LOOKING}" == "1" ]]; then
        EXTRA_ARGS+=(--double-left-looking)
    fi

    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        "${TEST_ROCSOLVER}" \
        "${TEST_RECURSIVE}" \
        "${ERROR_MODE}" \
        "${PRECISION}" \
        "${TEST_BLOCKED}" \
        "${TEST_DOUBLE_BLOCKED}" \
        "${BASELINE_NBS[@]}" \
        --double-nb "${DOUBLE_NBS[@]}" \
        --b "${MICRO_BLOCKS[@]}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "POTRF benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
