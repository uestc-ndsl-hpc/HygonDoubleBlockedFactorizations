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
PROFILE_EXEC=${BUILD_DIR}/profile_double_blocked_getrf
PIVOTING=${GETRF_PIVOTING:-1}
if [[ "${PIVOTING}" == "1" ]]; then
    MODE_NAME=GETRF
    DEFAULT_RESULT_SUBDIR=getrf_pivoted
    DEFAULT_METHODS=rocsolver,blocked,double,batched,mgemm
else
    MODE_NAME=GETRF-NPVT
    DEFAULT_RESULT_SUBDIR=getrf_npvt
    DEFAULT_METHODS=rocsolver,blocked,double
fi

RESULT_DIR=${GETRF_RESULT_DIR:-${ROOT_DIR}/results/fp32/${DEFAULT_RESULT_SUBDIR}}
SUMMARY_LOG=${RESULT_DIR}/summary.log

read -r -a NS <<< "${GETRF_NS:-4096 8192 12288 16384 20480 24576 28672 32768 36864 40960}"
read -r -a BASELINE_NBS <<< "${GETRF_BASELINE_NBS:-1024 2048}"
DOUBLE_NBS=()
MICRO_BLOCKS=()
if [[ -n "${GETRF_DOUBLE_NBS:-}" ]]; then
    read -r -a DOUBLE_NBS <<< "${GETRF_DOUBLE_NBS}"
fi
if [[ -n "${GETRF_MICRO_BLOCKS:-}" ]]; then
    read -r -a MICRO_BLOCKS <<< "${GETRF_MICRO_BLOCKS}"
fi

PRECISION=${GETRF_PRECISION:-0}
VERIFY=${GETRF_VERIFY:-0}
TRSM_LEAF=${GETRF_TRSM_LEAF:-512}
PIVOT_COLUMN_TILE=${GETRF_PIVOT_COLUMN_TILE:-2048}
INNER_GEMM_M_TILE=${GETRF_INNER_GEMM_M_TILE:-8192}
METHODS=${GETRF_METHODS:-${DEFAULT_METHODS}}
FULL_RESIDUAL=${GETRF_FULL_RESIDUAL:-0}
FULL_RESIDUAL_MAX_N=${GETRF_FULL_RESIDUAL_MAX_N:-8192}
RUN_STAGE_PROFILE=${GETRF_RUN_STAGE_PROFILE:-0}
STAGE_PROFILE_WARMUP=${GETRF_STAGE_PROFILE_WARMUP:-1}
STAGE_PROFILE_ITERS=${GETRF_STAGE_PROFILE_ITERS:-3}
STAGE_PROFILE_PIVOT_MODE=${GETRF_STAGE_PROFILE_PIVOT_MODE:-batched}

if [[ "${RUN_STAGE_PROFILE}" == "1" && "${PIVOTING}" != "1" ]]; then
    echo "GETRF stage profiling currently supports pivoted GETRF only." >&2
    exit 2
fi
if [[ "${RUN_STAGE_PROFILE}" == "1" && "${PRECISION}" != "0" ]]; then
    echo "GETRF stage profiling currently supports FP32 only." >&2
    exit 2
fi

PIVOT_ARGS=()
if [[ "${PIVOTING}" == "1" ]]; then
    PIVOT_ARGS+=(--pivot)
fi

FULL_RESIDUAL_ARGS=()
if [[ "${FULL_RESIDUAL}" == "1" ]]; then
    FULL_RESIDUAL_ARGS+=(
        --full-residual
        --full-residual-max-n "${FULL_RESIDUAL_MAX_N}"
    )
fi

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "${MODE_NAME} benchmark"
    echo "Executable         : ${EXEC}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix sizes       : ${NS[*]}"
    echo "Precision mode     : ${PRECISION} (0=FP32, 1=FP64)"
    echo "Backward verify    : ${VERIFY}"
    echo "Pivoting            : ${PIVOTING}"
    echo "Methods             : ${METHODS}"
    echo "Blocked-RL NBs     : ${BASELINE_NBS[*]}"
    if (( ${#DOUBLE_NBS[@]} == 0 && ${#MICRO_BLOCKS[@]} == 0 )); then
        echo "DoubleBlock NB/B   : N/2 / min(N/4,4096)"
    else
        echo "DoubleBlock NB/B   : ${DOUBLE_NBS[*]:-N/2} / ${MICRO_BLOCKS[*]:-N/4}"
    fi
    echo "Recursive TRSM leaf: ${TRSM_LEAF}"
    echo "Pivot column tile  : ${PIVOT_COLUMN_TILE}"
    echo "Inner GEMM M tile  : ${INNER_GEMM_M_TILE}"
    echo "Full residual      : ${FULL_RESIDUAL} (max N=${FULL_RESIDUAL_MAX_N})"
    echo "Run stage profile   : ${RUN_STAGE_PROFILE}"
    if [[ "${RUN_STAGE_PROFILE}" == "1" ]]; then
        echo "Stage warmup/iters : ${STAGE_PROFILE_WARMUP}/${STAGE_PROFILE_ITERS}"
        echo "Stage pivot mode   : ${STAGE_PROFILE_PIVOT_MODE}"
    fi
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}" "${ROOT_DIR}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
if [[ "${RUN_STAGE_PROFILE}" == "1" ]]; then
    check_benchmark_binary \
        "${PROFILE_EXEC}" \
        "${SUMMARY_LOG}" \
        "rocsolver|rocblas|amdhip64|hsa-runtime64"
fi
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"; do
    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> Running ${MODE_NAME} benchmark: N=${N}" | tee -a "${SUMMARY_LOG}"

    OUT_FILE=${RESULT_DIR}/getrf_N${N}.log
    CSV_FILE=${RESULT_DIR}/getrf_N${N}_timings.csv

    RUN_DOUBLE_NBS=("${DOUBLE_NBS[@]}")
    RUN_MICRO_BLOCKS=("${MICRO_BLOCKS[@]}")
    if (( ${#RUN_DOUBLE_NBS[@]} == 0 )); then
        RUN_DOUBLE_NBS=($((N / 2)))
    fi
    if (( ${#RUN_MICRO_BLOCKS[@]} == 0 )); then
        DEFAULT_B=$((N / 4))
        if (( DEFAULT_B > 4096 )); then
            DEFAULT_B=4096
        fi
        RUN_MICRO_BLOCKS=("${DEFAULT_B}")
    fi

    BENCHMARK_CSV_PATH="${CSV_FILE}" \
    "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${N}" \
        "${PRECISION}" \
        "${VERIFY}" \
        "${BASELINE_NBS[@]}" \
        "${PIVOT_ARGS[@]}" \
        --methods "${METHODS}" \
        --double-nb "${RUN_DOUBLE_NBS[@]}" \
        --b "${RUN_MICRO_BLOCKS[@]}" \
        --trsm-leaf "${TRSM_LEAF}" \
        --pivot-tile "${PIVOT_COLUMN_TILE}" \
        --inner-gemm-m-tile "${INNER_GEMM_M_TILE}" \
        "${FULL_RESIDUAL_ARGS[@]}" \
        2>&1 | tee "${OUT_FILE}"

    cat "${OUT_FILE}" >> "${SUMMARY_LOG}"

    if [[ "${RUN_STAGE_PROFILE}" == "1" ]]; then
        PROFILE_OUTER_NB=${GETRF_PROFILE_OUTER_NB:-${RUN_DOUBLE_NBS[0]}}
        PROFILE_INNER_NB=${GETRF_PROFILE_INNER_NB:-${RUN_MICRO_BLOCKS[0]}}
        PROFILE_OUT_FILE=${RESULT_DIR}/getrf_profile_N${N}.log

        echo ">>> Profiling DoubleBlock GETRF: N=${N}, NB/B=${PROFILE_OUTER_NB}/${PROFILE_INNER_NB}" \
            | tee -a "${SUMMARY_LOG}"
        "${LAUNCH_PREFIX[@]}" "${PROFILE_EXEC}" \
            --n "${N}" \
            --outer-nb "${PROFILE_OUTER_NB}" \
            --inner-nb "${PROFILE_INNER_NB}" \
            --trsm-leaf "${TRSM_LEAF}" \
            --pivot-tile "${PIVOT_COLUMN_TILE}" \
            --inner-gemm-m-tile "${INNER_GEMM_M_TILE}" \
            --pivot-mode "${STAGE_PROFILE_PIVOT_MODE}" \
            --warmup "${STAGE_PROFILE_WARMUP}" \
            --iters "${STAGE_PROFILE_ITERS}" \
            2>&1 | tee "${PROFILE_OUT_FILE}"

        cat "${PROFILE_OUT_FILE}" >> "${SUMMARY_LOG}"
    fi
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "${MODE_NAME} benchmark finished" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
