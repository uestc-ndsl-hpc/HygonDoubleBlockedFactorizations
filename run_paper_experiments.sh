#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

export HIP_VISIBLE_DEVICES=${PAPER_DEVICE:-${HIP_VISIBLE_DEVICES:-7}}
export BENCHMARK_WARMUP_ITERS=${PAPER_WARMUP_ITERS:-2}
export BENCHMARK_PROFILE_ITERS=${PAPER_PROFILE_ITERS:-10}

setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

ROOT_DIR=${SCRIPT_DIR}
BUILD_DIR=${ROOT_DIR}/build
POTRF_EXEC=${BUILD_DIR}/benchmark_potrf
GETRF_EXEC=${BUILD_DIR}/benchmark_getrf
POTRF_PROFILE_EXEC=${BUILD_DIR}/profile_double_blocked_potrf
GETRF_PROFILE_EXEC=${BUILD_DIR}/profile_double_blocked_getrf

PAPER_SECTIONS=${PAPER_SECTIONS:-correctness,scaling,profiles}
PAPER_RUN_ID=${PAPER_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_DIR=${PAPER_RESULT_DIR:-${ROOT_DIR}/results/fp32/paper_final/${PAPER_RUN_ID}}
SUMMARY_LOG=${RESULT_DIR}/manifest.log

read -r -a SCALING_NS <<< "${PAPER_NS:-8192 16384 24576 32768 40960 49152 57344 65536}"
read -r -a CORRECTNESS_NS <<< "${PAPER_CORRECTNESS_NS:-4096 5000}"

POTRF_NB=${PAPER_POTRF_NB:-8192}
POTRF_B=${PAPER_POTRF_B:-1024}
POTRF_SYRK_K_TILE=${PAPER_POTRF_SYRK_K_TILE:-8192}
POTRF_GEMM_M_TILE=${PAPER_POTRF_GEMM_M_TILE:-8192}
POTRF_GEMM_M_THRESHOLD=${PAPER_POTRF_GEMM_M_THRESHOLD:-32768}
POTRF_GEMM_K_THRESHOLD=${PAPER_POTRF_GEMM_K_THRESHOLD:-5632}
POTRF_METHODS=${PAPER_POTRF_METHODS:-rocsolver,double,kbsyrk,mgemm}
POTRF_LARGE_METHODS=${PAPER_POTRF_LARGE_METHODS:-double,kbsyrk,mgemm}

GETRF_OUTER_TRSM=${PAPER_GETRF_OUTER_TRSM:-iterative}
GETRF_TRSM_BLOCK=${PAPER_GETRF_TRSM_BLOCK:-2048}
GETRF_PIVOT_TILE=${PAPER_GETRF_PIVOT_TILE:-2048}
GETRF_GEMM_M_TILE=${PAPER_GETRF_GEMM_M_TILE:-8192}
GETRF_METHODS=${PAPER_GETRF_METHODS:-rocsolver,double,batched,mgemm}
GETRF_LARGE_METHODS=${PAPER_GETRF_LARGE_METHODS:-double,batched,mgemm}
ROCSOLVER_MAX_N=${PAPER_ROCSOLVER_MAX_N:-40960}

PROFILE_N=${PAPER_PROFILE_N:-40960}
STAGE_WARMUP=${PAPER_STAGE_WARMUP:-1}
STAGE_ITERS=${PAPER_STAGE_ITERS:-3}
CORRECTNESS_TILE=${PAPER_CORRECTNESS_TILE:-1024}
POTRF_CORRECTNESS_MAX_BACKWARD_ERROR=${PAPER_POTRF_CORRECTNESS_MAX_BACKWARD_ERROR:-2e-6}
POTRF_SCALING_MAX_BACKWARD_ERROR=${PAPER_POTRF_SCALING_MAX_BACKWARD_ERROR:-5e-6}
GETRF_CORRECTNESS_MAX_BACKWARD_ERROR=${PAPER_GETRF_CORRECTNESS_MAX_BACKWARD_ERROR:-5e-5}
GETRF_SCALING_MAX_BACKWARD_ERROR=${PAPER_GETRF_SCALING_MAX_BACKWARD_ERROR:-5e-4}

if [[ ",${POTRF_LARGE_METHODS}," == *",rocsolver,"* ]] \
    || [[ ",${GETRF_LARGE_METHODS}," == *",rocsolver,"* ]]; then
    echo "Large-N method lists must not include rocSOLVER." >&2
    exit 2
fi

section_enabled() {
    local section=$1
    [[ ",${PAPER_SECTIONS}," == *",${section},"* ]]
}

run_benchmark() {
    local csv_path=$1
    local log_path=$2
    shift 2

    echo ">>> $*" | tee -a "${SUMMARY_LOG}"
    BENCHMARK_CSV_PATH="${csv_path}" \
        "${LAUNCH_PREFIX[@]}" "$@" 2>&1 | tee "${log_path}"
}

run_profile() {
    local log_path=$1
    shift

    echo ">>> $*" | tee -a "${SUMMARY_LOG}"
    "${LAUNCH_PREFIX[@]}" "$@" 2>&1 | tee "${log_path}"
}

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

{
    echo "=================================================="
    echo "Double-blocked factorization paper experiments"
    echo "Sections            : ${PAPER_SECTIONS}"
    echo "Result directory    : ${RESULT_DIR}"
    echo "Scaling sizes       : ${SCALING_NS[*]}"
    echo "Correctness sizes   : ${CORRECTNESS_NS[*]}"
    echo "Correctness tile    : ${CORRECTNESS_TILE}"
    echo "POTRF input seed    : 20260716"
    echo "GETRF input seed    : 1234567890"
    echo "POTRF B/b           : ${POTRF_NB}/${POTRF_B}"
    echo "POTRF TK/TM         : ${POTRF_SYRK_K_TILE}/${POTRF_GEMM_M_TILE}"
    echo "POTRF thresholds    : M>${POTRF_GEMM_M_THRESHOLD}, K>=${POTRF_GEMM_K_THRESHOLD}"
    echo "POTRF methods       : ${POTRF_METHODS}"
    echo "POTRF large methods : ${POTRF_LARGE_METHODS}"
    echo "GETRF B policy      : N/2"
    echo "GETRF b policy      : min(N/4,4096)"
    echo "GETRF outer TRSM    : ${GETRF_OUTER_TRSM}/${GETRF_TRSM_BLOCK}"
    echo "GETRF bp/TM         : ${GETRF_PIVOT_TILE}/${GETRF_GEMM_M_TILE}"
    echo "GETRF methods       : ${GETRF_METHODS}"
    echo "GETRF large methods : ${GETRF_LARGE_METHODS}"
    echo "rocSOLVER max N     : ${ROCSOLVER_MAX_N}"
    echo "Stage profile N     : ${PROFILE_N}"
    echo "Scaling warmup/iters: ${PAPER_WARMUP_ITERS:-2}/${PAPER_PROFILE_ITERS:-10}"
    echo "Correct warmup/iters: ${PAPER_CORRECTNESS_WARMUP:-0}/${PAPER_CORRECTNESS_ITERS:-1}"
    echo "Stage warmup/iters  : ${STAGE_WARMUP}/${STAGE_ITERS}"
    echo "LU backward error   : ||P*A-L*U||_F / ||A||_F"
    echo "POTRF backward error: ||A-L*L^T||_F / ||A||_F"
    echo "POTRF full limit    : ${POTRF_CORRECTNESS_MAX_BACKWARD_ERROR}"
    echo "POTRF sampled limit : ${POTRF_SCALING_MAX_BACKWARD_ERROR}"
    echo "GETRF full limit    : ${GETRF_CORRECTNESS_MAX_BACKWARD_ERROR}"
    echo "GETRF sampled limit : ${GETRF_SCALING_MAX_BACKWARD_ERROR}"
    echo "Scaling estimator   : 4096 deterministic full-matrix samples"
    echo "Correctness metric  : full FP64-accumulated Frobenius norms"
} | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}" "${ROOT_DIR}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

if [[ "${PAPER_CONFIRM_IDLE:-0}" != "1" ]]; then
    echo "Refusing to start GPU experiments without PAPER_CONFIRM_IDLE=1." \
        | tee -a "${SUMMARY_LOG}" >&2
    echo "Inspect the device state above, then rerun with the same PAPER_RUN_ID." \
        | tee -a "${SUMMARY_LOG}" >&2
    exit 3
fi

if [[ "${PAPER_ALLOW_DIRTY:-0}" != "1" ]] \
    && [[ -n "$(git -C "${ROOT_DIR}" status --porcelain)" ]]; then
    echo "Refusing to run final experiments from a dirty worktree." \
        | tee -a "${SUMMARY_LOG}" >&2
    exit 4
fi

check_benchmark_binary \
    "${POTRF_EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
check_benchmark_binary \
    "${GETRF_EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|amdhip64|hsa-runtime64"
if section_enabled profiles; then
    check_benchmark_binary \
        "${POTRF_PROFILE_EXEC}" \
        "${SUMMARY_LOG}" \
        "rocsolver|rocblas|amdhip64|hsa-runtime64"
    check_benchmark_binary \
        "${GETRF_PROFILE_EXEC}" \
        "${SUMMARY_LOG}" \
        "rocsolver|rocblas|amdhip64|hsa-runtime64"
fi
configure_numa_launch_prefix "${SUMMARY_LOG}"

if section_enabled correctness; then
    CORRECTNESS_DIR=${RESULT_DIR}/correctness
    mkdir -p "${CORRECTNESS_DIR}"

    export BENCHMARK_WARMUP_ITERS=${PAPER_CORRECTNESS_WARMUP:-0}
    export BENCHMARK_PROFILE_ITERS=${PAPER_CORRECTNESS_ITERS:-1}

    for N in "${CORRECTNESS_NS[@]}"; do
        export BENCHMARK_MAX_BACKWARD_ERROR=${POTRF_CORRECTNESS_MAX_BACKWARD_ERROR}
        run_benchmark \
            "${CORRECTNESS_DIR}/potrf_N${N}_timings.csv" \
            "${CORRECTNESS_DIR}/potrf_N${N}.log" \
            "${POTRF_EXEC}" \
            "${N}" 0 0 2 0 0 0 2048 \
            --methods "${POTRF_METHODS}" \
            --double-nb 2048 \
            --b 512 \
            --syrk-k-tile "${CORRECTNESS_TILE}" \
            --gemm-m-tile "${CORRECTNESS_TILE}" \
            --gemm-m-threshold 0 \
            --gemm-k-threshold 0 \
            --full-residual \
            --full-residual-max-n 8192

        export BENCHMARK_MAX_BACKWARD_ERROR=${GETRF_CORRECTNESS_MAX_BACKWARD_ERROR}
        run_benchmark \
            "${CORRECTNESS_DIR}/getrf_N${N}_timings.csv" \
            "${CORRECTNESS_DIR}/getrf_N${N}.log" \
            "${GETRF_EXEC}" \
            "${N}" 0 1 2048 \
            --pivot \
            --methods "${GETRF_METHODS}" \
            --double-nb 2048 \
            --b 512 \
            --outer-trsm "${GETRF_OUTER_TRSM}" \
            --trsm-block "${GETRF_TRSM_BLOCK}" \
            --pivot-tile "${GETRF_PIVOT_TILE}" \
            --inner-gemm-m-tile "${CORRECTNESS_TILE}" \
            --full-residual \
            --full-residual-max-n 8192
    done

    unset BENCHMARK_MAX_BACKWARD_ERROR
fi

if section_enabled scaling; then
    SCALING_DIR=${RESULT_DIR}/scaling
    mkdir -p "${SCALING_DIR}"

    export BENCHMARK_WARMUP_ITERS=${PAPER_WARMUP_ITERS:-2}
    export BENCHMARK_PROFILE_ITERS=${PAPER_PROFILE_ITERS:-10}

    for N in "${SCALING_NS[@]}"; do
        CURRENT_POTRF_METHODS=${POTRF_METHODS}
        CURRENT_GETRF_METHODS=${GETRF_METHODS}
        if (( N > ROCSOLVER_MAX_N )); then
            CURRENT_POTRF_METHODS=${POTRF_LARGE_METHODS}
            CURRENT_GETRF_METHODS=${GETRF_LARGE_METHODS}
        fi

        CURRENT_POTRF_NB=${POTRF_NB}
        if (( CURRENT_POTRF_NB > N )); then
            CURRENT_POTRF_NB=${N}
        fi

        export BENCHMARK_MAX_BACKWARD_ERROR=${POTRF_SCALING_MAX_BACKWARD_ERROR}
        run_benchmark \
            "${SCALING_DIR}/potrf_N${N}_timings.csv" \
            "${SCALING_DIR}/potrf_N${N}.log" \
            "${POTRF_EXEC}" \
            "${N}" 0 0 2 0 0 0 2048 \
            --methods "${CURRENT_POTRF_METHODS}" \
            --double-nb "${CURRENT_POTRF_NB}" \
            --b "${POTRF_B}" \
            --syrk-k-tile "${POTRF_SYRK_K_TILE}" \
            --gemm-m-tile "${POTRF_GEMM_M_TILE}" \
            --gemm-m-threshold "${POTRF_GEMM_M_THRESHOLD}" \
            --gemm-k-threshold "${POTRF_GEMM_K_THRESHOLD}"

        GETRF_NB=$((N / 2))
        GETRF_B=$((N / 4))
        if (( GETRF_B > 4096 )); then
            GETRF_B=4096
        fi

        export BENCHMARK_MAX_BACKWARD_ERROR=${GETRF_SCALING_MAX_BACKWARD_ERROR}
        run_benchmark \
            "${SCALING_DIR}/getrf_N${N}_timings.csv" \
            "${SCALING_DIR}/getrf_N${N}.log" \
            "${GETRF_EXEC}" \
            "${N}" 0 1 2048 \
            --pivot \
            --methods "${CURRENT_GETRF_METHODS}" \
            --double-nb "${GETRF_NB}" \
            --b "${GETRF_B}" \
            --outer-trsm "${GETRF_OUTER_TRSM}" \
            --trsm-block "${GETRF_TRSM_BLOCK}" \
            --pivot-tile "${GETRF_PIVOT_TILE}" \
            --inner-gemm-m-tile "${GETRF_GEMM_M_TILE}"
    done

    unset BENCHMARK_MAX_BACKWARD_ERROR
fi

if section_enabled profiles; then
    PROFILE_DIR=${RESULT_DIR}/profiles
    mkdir -p "${PROFILE_DIR}"

    PROFILE_GETRF_NB=$((PROFILE_N / 2))
    PROFILE_GETRF_B=$((PROFILE_N / 4))
    if (( PROFILE_GETRF_B > 4096 )); then
        PROFILE_GETRF_B=4096
    fi

    run_profile \
        "${PROFILE_DIR}/potrf_basic_N${PROFILE_N}.log" \
        "${POTRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --nb "${POTRF_NB}" \
        --b "${POTRF_B}" \
        --syrk-k-tile 0 \
        --inner-gemm-m-tile 0 \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/potrf_kbsyrk_N${PROFILE_N}.log" \
        "${POTRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --nb "${POTRF_NB}" \
        --b "${POTRF_B}" \
        --syrk-k-tile "${POTRF_SYRK_K_TILE}" \
        --inner-gemm-m-tile 0 \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/potrf_final_N${PROFILE_N}.log" \
        "${POTRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --nb "${POTRF_NB}" \
        --b "${POTRF_B}" \
        --syrk-k-tile "${POTRF_SYRK_K_TILE}" \
        --inner-gemm-m-tile "${POTRF_GEMM_M_TILE}" \
        --inner-gemm-m-threshold "${POTRF_GEMM_M_THRESHOLD}" \
        --inner-gemm-k-threshold "${POTRF_GEMM_K_THRESHOLD}" \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/getrf_immediate_N${PROFILE_N}.log" \
        "${GETRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --outer-nb "${PROFILE_GETRF_NB}" \
        --inner-nb "${PROFILE_GETRF_B}" \
        --outer-trsm "${GETRF_OUTER_TRSM}" \
        --trsm-block "${GETRF_TRSM_BLOCK}" \
        --pivot-tile "${GETRF_PIVOT_TILE}" \
        --inner-gemm-m-tile 0 \
        --pivot-mode immediate \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/getrf_batched_untiled_N${PROFILE_N}.log" \
        "${GETRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --outer-nb "${PROFILE_GETRF_NB}" \
        --inner-nb "${PROFILE_GETRF_B}" \
        --outer-trsm "${GETRF_OUTER_TRSM}" \
        --trsm-block "${GETRF_TRSM_BLOCK}" \
        --pivot-tile "${GETRF_PIVOT_TILE}" \
        --inner-gemm-m-tile 0 \
        --pivot-mode batched \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/getrf_direct_outer_trsm_N${PROFILE_N}.log" \
        "${GETRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --outer-nb "${PROFILE_GETRF_NB}" \
        --inner-nb "${PROFILE_GETRF_B}" \
        --outer-trsm direct \
        --trsm-block "${GETRF_TRSM_BLOCK}" \
        --pivot-tile "${GETRF_PIVOT_TILE}" \
        --inner-gemm-m-tile "${GETRF_GEMM_M_TILE}" \
        --pivot-mode batched \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"

    run_profile \
        "${PROFILE_DIR}/getrf_final_N${PROFILE_N}.log" \
        "${GETRF_PROFILE_EXEC}" \
        --n "${PROFILE_N}" \
        --outer-nb "${PROFILE_GETRF_NB}" \
        --inner-nb "${PROFILE_GETRF_B}" \
        --outer-trsm "${GETRF_OUTER_TRSM}" \
        --trsm-block "${GETRF_TRSM_BLOCK}" \
        --pivot-tile "${GETRF_PIVOT_TILE}" \
        --inner-gemm-m-tile "${GETRF_GEMM_M_TILE}" \
        --pivot-mode batched \
        --warmup "${STAGE_WARMUP}" \
        --iters "${STAGE_ITERS}"
fi

echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "Paper experiment run completed" | tee -a "${SUMMARY_LOG}"
echo "Manifest: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
