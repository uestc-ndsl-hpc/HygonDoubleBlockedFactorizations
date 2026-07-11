#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

ROOT_DIR=${SCRIPT_DIR}
EXEC=${ROOT_DIR}/build/benchmark_potrf
RESULT_DIR=${POTRF_HIER_RESULT_DIR:-${ROOT_DIR}/results/fp32/potrf_hierarchical}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${POTRF_NS:-16384 32768 40960 57344 65536 73728 81920}"
read -r -a BLOCKED_NBS <<< "${POTRF_BASELINE_NBS:-2048}"
read -r -a DOUBLE_NBS <<< "${POTRF_DOUBLE_NBS:-8192}"
read -r -a PANEL_BLOCKS <<< "${POTRF_PANEL_BLOCKS:-2048 4096}"
read -r -a POTRF_LEAVES <<< "${POTRF_LEAF_BLOCKS:-512}"
read -r -a SYRK_K_TILES <<< "${POTRF_HIER_SYRK_K_TILES:-8192}"

ERROR_MODE=${POTRF_ERROR_MODE:-2}
PRECISION=${POTRF_PRECISION:-0}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "POTRF hierarchical DoubleBlock experiment"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "Blocked-RL NBs     : ${BLOCKED_NBS[*]}"
    echo "Outer NBs          : ${DOUBLE_NBS[*]}"
    echo "Panel blocks       : ${PANEL_BLOCKS[*]}"
    echo "POTRF leaves       : ${POTRF_LEAVES[*]}"
    echo "SYRK K tiles       : ${SYRK_K_TILES[*]}"
    echo "GEMM M tile        : 8192 when M>32768"
    echo "Error verify mode  : ${ERROR_MODE}"
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
    echo ">>> Running hierarchical comparison: N=${N}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/potrf_hierarchical_N${N}.log
    echo "Output detail      : ${OUT_FILE}" | tee -a "${SUMMARY_LOG}"

    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        0 \
        0 \
        "${ERROR_MODE}" \
        "${PRECISION}" \
        1 \
        1 \
        "${BLOCKED_NBS[@]}" \
        --double-nb "${DOUBLE_NBS[@]}" \
        --b "${POTRF_LEAVES[@]}" \
        --panel-b "${PANEL_BLOCKS[@]}" \
        --hier-syrk-k-tile "${SYRK_K_TILES[@]}" \
        --tail-nb 0 \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "POTRF hierarchical experiment finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
