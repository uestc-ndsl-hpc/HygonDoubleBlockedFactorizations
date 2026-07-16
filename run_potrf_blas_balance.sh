#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

ROOT_DIR=${SCRIPT_DIR}
BLOCKED_EXEC=${ROOT_DIR}/build/profile_blocked_potrf
DOUBLE_EXEC=${ROOT_DIR}/build/profile_double_blocked_potrf
RESULT_DIR=${POTRF_BALANCE_RESULT_DIR:-${ROOT_DIR}/results/fp32/potrf_blas_balance}
SUMMARY_LOG=${RESULT_DIR}/summary.log
BLOCKED_LOG=${RESULT_DIR}/blocked_profile.log
DOUBLE_LOG=${RESULT_DIR}/double_block_profile.log

N=${POTRF_N:-40960}
BLOCKED_NB=${POTRF_BLOCKED_NB:-2048}
DOUBLE_NB=${POTRF_DOUBLE_NB:-8192}
MICRO_BLOCK=${POTRF_MICRO_BLOCK:-512}
SYRK_K_TILE=${POTRF_SYRK_K_TILE:-8192}
GEMM_M_TILE=${POTRF_GEMM_M_TILE:-8192}
GEMM_M_THRESHOLD=${POTRF_GEMM_M_THRESHOLD:-32768}
GEMM_K_THRESHOLD=${POTRF_GEMM_K_THRESHOLD:-5632}
WARMUP=${POTRF_PROFILE_WARMUP:-1}
ITERS=${POTRF_PROFILE_ITERS:-3}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}" "${BLOCKED_LOG}" "${DOUBLE_LOG}"

{
    echo "=================================================="
    echo "POTRF BLAS balance experiment"
    echo "N                  : ${N}"
    echo "Blocked-RL NB      : ${BLOCKED_NB}"
    echo "DoubleBlock NB/B   : ${DOUBLE_NB}/${MICRO_BLOCK}"
    echo "SYRK K tile        : ${SYRK_K_TILE}"
    echo "GEMM M tile        : ${GEMM_M_TILE}"
    echo "GEMM thresholds    : M>${GEMM_M_THRESHOLD}, K>=${GEMM_K_THRESHOLD}"
    echo "Warmup/iterations  : ${WARMUP}/${ITERS}"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}" "${ROOT_DIR}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${BLOCKED_EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
check_benchmark_binary \
    "${DOUBLE_EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

echo "" | tee -a "${SUMMARY_LOG}"
echo ">>> Profiling Blocked-RL" | tee -a "${SUMMARY_LOG}"
"${LAUNCH_PREFIX[@]}" "${BLOCKED_EXEC}" \
    --n "${N}" \
    --nb "${BLOCKED_NB}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    2>&1 | tee "${BLOCKED_LOG}"

echo "" | tee -a "${SUMMARY_LOG}"
echo ">>> Profiling optimized DoubleBlock" | tee -a "${SUMMARY_LOG}"
"${LAUNCH_PREFIX[@]}" "${DOUBLE_EXEC}" \
    --n "${N}" \
    --nb "${DOUBLE_NB}" \
    --b "${MICRO_BLOCK}" \
    --syrk-k-tile "${SYRK_K_TILE}" \
    --inner-gemm-m-tile "${GEMM_M_TILE}" \
    --inner-gemm-m-threshold "${GEMM_M_THRESHOLD}" \
    --inner-gemm-k-threshold "${GEMM_K_THRESHOLD}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    2>&1 | tee "${DOUBLE_LOG}"

extract_stage_field() {
    local log_file=$1
    local stage=$2
    local field=$3

    awk -F '|' -v target="${stage}" -v field="${field}" '
        function trim(value) {
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
            return value
        }
        {
            name = trim($1)
            if (name == target) {
                print trim($field)
                exit
            }
        }
    ' "${log_file}"
}

require_metric() {
    local name=$1
    local value=$2

    if [ -z "${value}" ]; then
        echo "Error: failed to extract '${name}' from profiler output" >&2
        exit 1
    fi

    return 0
}

calculate() {
    local expression=$1
    awk "BEGIN { printf \"%.4f\", ${expression} }"
}

BLOCK_POTRF_MS=$(extract_stage_field "${BLOCKED_LOG}" "POTRF" 2)
BLOCK_TRSM_MS=$(extract_stage_field "${BLOCKED_LOG}" "TRSM" 2)
BLOCK_SYRK_MS=$(extract_stage_field "${BLOCKED_LOG}" "SYRK" 2)
BLOCK_SYRK_GFLOPS=$(extract_stage_field "${BLOCKED_LOG}" "SYRK" 6)

DOUBLE_POTRF_MS=$(extract_stage_field "${DOUBLE_LOG}" "POTRF" 2)
DOUBLE_TRSM_MS=$(extract_stage_field "${DOUBLE_LOG}" "TRSM" 2)
INNER_GEMM_MS=$(extract_stage_field "${DOUBLE_LOG}" "Inner GEMM" 2)
INNER_GEMM_GFLOPS=$(extract_stage_field "${DOUBLE_LOG}" "Inner GEMM" 6)
OUTER_SYRK_MS=$(extract_stage_field "${DOUBLE_LOG}" "Outer SYRK" 2)
OUTER_SYRK_GFLOPS=$(extract_stage_field "${DOUBLE_LOG}" "Outer SYRK" 6)
OUTER_GEMM_MS=$(extract_stage_field "${DOUBLE_LOG}" "Outer GEMM" 2)
OUTER_GEMM_GFLOPS=$(extract_stage_field "${DOUBLE_LOG}" "Outer GEMM" 6)

for metric in \
    BLOCK_POTRF_MS BLOCK_TRSM_MS BLOCK_SYRK_MS BLOCK_SYRK_GFLOPS \
    DOUBLE_POTRF_MS DOUBLE_TRSM_MS INNER_GEMM_MS INNER_GEMM_GFLOPS \
    OUTER_SYRK_MS OUTER_SYRK_GFLOPS OUTER_GEMM_MS OUTER_GEMM_GFLOPS; do
    require_metric "${metric}" "${!metric}"
done

BLOCK_PANEL_MS=$(calculate "${BLOCK_POTRF_MS} + ${BLOCK_TRSM_MS}")
DOUBLE_PANEL_MS=$(calculate "${DOUBLE_POTRF_MS} + ${DOUBLE_TRSM_MS}")
BLOCK_UPDATE_MS=${BLOCK_SYRK_MS}
DOUBLE_UPDATE_MS=$(calculate "${INNER_GEMM_MS} + ${OUTER_SYRK_MS} + ${OUTER_GEMM_MS}")
BLOCK_TOTAL_MS=$(calculate "${BLOCK_PANEL_MS} + ${BLOCK_UPDATE_MS}")
DOUBLE_TOTAL_MS=$(calculate "${DOUBLE_PANEL_MS} + ${DOUBLE_UPDATE_MS}")

COMBINED_GEMM_GFLOPS=$(calculate \
    "(${INNER_GEMM_GFLOPS} * ${INNER_GEMM_MS} + ${OUTER_GEMM_GFLOPS} * ${OUTER_GEMM_MS}) / (${INNER_GEMM_MS} + ${OUTER_GEMM_MS})")
GEMM_TO_BLOCK_SYRK=$(calculate "${COMBINED_GEMM_GFLOPS} / ${BLOCK_SYRK_GFLOPS}")
OUTER_SYRK_TO_BLOCK_SYRK=$(calculate "${OUTER_SYRK_GFLOPS} / ${BLOCK_SYRK_GFLOPS}")
UPDATE_SPEEDUP=$(calculate "${BLOCK_UPDATE_MS} / ${DOUBLE_UPDATE_MS}")
TOTAL_SPEEDUP=$(calculate "${BLOCK_TOTAL_MS} / ${DOUBLE_TOTAL_MS}")
PANEL_RATIO=$(calculate "${DOUBLE_PANEL_MS} / ${BLOCK_PANEL_MS}")

{
    echo ""
    echo "================================================================================================"
    echo "Extracted stage comparison (totals cover ${ITERS} profiled iterations)"
    echo "================================================================================================"
    printf "%-28s | %12s | %12s\n" "Stage" "Total(ms)" "GFLOPS"
    echo "------------------------------------------------------------------------------------------------"
    printf "%-28s | %12s | %12s\n" "Blocked POTRF+TRSM" "${BLOCK_PANEL_MS}" "N/A"
    printf "%-28s | %12s | %12s\n" "Blocked SYRK" "${BLOCK_SYRK_MS}" "${BLOCK_SYRK_GFLOPS}"
    printf "%-28s | %12s | %12s\n" "Double POTRF+TRSM" "${DOUBLE_PANEL_MS}" "N/A"
    printf "%-28s | %12s | %12s\n" "Double Inner GEMM" "${INNER_GEMM_MS}" "${INNER_GEMM_GFLOPS}"
    printf "%-28s | %12s | %12s\n" "Double Outer SYRK" "${OUTER_SYRK_MS}" "${OUTER_SYRK_GFLOPS}"
    printf "%-28s | %12s | %12s\n" "Double Outer GEMM" "${OUTER_GEMM_MS}" "${OUTER_GEMM_GFLOPS}"
    echo "================================================================================================"
    echo "Combined GEMM / Blocked SYRK throughput : ${GEMM_TO_BLOCK_SYRK}x"
    echo "Outer SYRK / Blocked SYRK throughput    : ${OUTER_SYRK_TO_BLOCK_SYRK}x"
    echo "Blocked update / Double update time     : ${UPDATE_SPEEDUP}x"
    echo "Double panel / Blocked panel time       : ${PANEL_RATIO}x"
    echo "Profiled total Blocked / Double         : ${TOTAL_SPEEDUP}x"
    echo "================================================================================================"
} | tee -a "${SUMMARY_LOG}"

awk -v ratio="${GEMM_TO_BLOCK_SYRK}" '
    BEGIN {
        if (ratio < 1.30) {
            print "Conclusion: GEMM has only a limited throughput advantage over Blocked-RL SYRK."
            print "The hypothesis is supported: replacing work with GEMM alone cannot create a large speedup."
        } else if (ratio < 1.70) {
            print "Conclusion: GEMM has a moderate advantage; panel overhead and update scheduling decide the result."
        } else {
            print "Conclusion: GEMM has a strong advantage. If total POTRF speedup is still small, inspect panel overhead."
        }
    }
' | tee -a "${SUMMARY_LOG}"

echo "Raw Blocked-RL profile : ${BLOCKED_LOG}" | tee -a "${SUMMARY_LOG}"
echo "Raw DoubleBlock profile: ${DOUBLE_LOG}" | tee -a "${SUMMARY_LOG}"
echo "Summary                : ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
