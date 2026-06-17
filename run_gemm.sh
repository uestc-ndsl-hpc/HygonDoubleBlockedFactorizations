#!/usr/bin/env bash
set -u
set -o pipefail

ROCM_PATH=${ROCM_PATH:-/opt/dtk-26.04}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

export ROCM_PATH
export HIP_VISIBLE_DEVICES
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export ROCBLAS_LAYER=0
export TENSILE_DB=0

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build"}
RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/results/fp32/gemm"}
SUMMARY_LOG=${SUMMARY_LOG:-"${RESULT_DIR}/summary.log"}

find_executable() {
    if [ -n "${EXEC:-}" ]; then
        echo "${EXEC}"
        return
    fi

    local candidates=(
        "${BUILD_DIR}/src/gemm/benchmark_gemm"
        "${BUILD_DIR}/benchmark_gemm"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [ -x "${candidate}" ]; then
            echo "${candidate}"
            return
        fi
    done

    echo "${candidates[0]}"
}

EXEC=$(find_executable)

mkdir -p "${RESULT_DIR}"
: > "${SUMMARY_LOG}"

log() {
    echo "$*" | tee -a "${SUMMARY_LOG}"
}

log "=================================================="
log "GEMM benchmark"
log "DTK environment    : ${ROCM_PATH}"
log "Executable         : ${EXEC}"
log "Result directory   : ${RESULT_DIR}"
log "Target device      : HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
log "GEMM_TOTAL_SIZE    : ${GEMM_TOTAL_SIZE:-32768}"
log "GEMM_VERIFY        : ${GEMM_VERIFY:-0}"
log "HSS A mode         : ${HYGON_HSS_A_MODE:-preconvert}"
log "HSS input mode     : ${HYGON_HSS_INPUT_MODE:-true_hss}"
log "HSS kernel         : ${HYGON_HSS_KERNEL:-auto}"
log "=================================================="

if [ ! -x "${EXEC}" ]; then
    log "Error: executable not found or not executable: ${EXEC}"
    log "Build benchmark_gemm before running this script."
    exit 1
fi

log ""
log "Checking binary dependencies..."
missing_deps=$(ldd "${EXEC}" 2>/dev/null | grep "not found" || true)
if [ -n "${missing_deps}" ]; then
    log "${missing_deps}"
    log "Error: missing shared libraries."
    exit 1
fi
log "Dependency check passed."

log ""
log "Starting GEMM benchmark..."

if ! "${EXEC}" "$@" 2>&1 | tee -a "${SUMMARY_LOG}"; then
    log "Error: benchmark failed."
    exit 1
fi

log ""
log "=================================================="
log "GEMM benchmark finished"
log "Summary log: ${SUMMARY_LOG}"
log "=================================================="
