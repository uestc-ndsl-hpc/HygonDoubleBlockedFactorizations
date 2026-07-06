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

module load compiler/openmp/50 2>/dev/null || true

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build"}

RUN_MODE=${RUN_MODE:-sweep}
PRECISION=${PRECISION:-0}

case "${PRECISION}" in
    0) PRECISION_NAME="fp32" ;;
    1) PRECISION_NAME="fp64" ;;
    2) PRECISION_NAME="mixed" ;;
    *)
        echo "Error: unsupported PRECISION=${PRECISION} (0=FP32, 1=FP64, 2=mixed)"
        exit 1
        ;;
esac

case "${RUN_MODE}" in
    smoke)
        DEFAULT_NS="4096"
        DEFAULT_NBS="256 512"
        DEFAULT_ERROR_MODE=3
        ;;
    sweep)
        DEFAULT_NS="8192 16384 24576 32768 40960"
        DEFAULT_NBS="256 512 1024 2048"
        DEFAULT_ERROR_MODE=2
        ;;
    perf)
        DEFAULT_NS="8192 16384 24576 32768 40960 49152 57344 65536"
        DEFAULT_NBS="256 512 1024 2048"
        DEFAULT_ERROR_MODE=0
        ;;
    *)
        echo "Error: unsupported RUN_MODE=${RUN_MODE} (smoke, sweep, perf)"
        exit 1
        ;;
esac

RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/results/${PRECISION_NAME}/potrf_${RUN_MODE}"}
SUMMARY_LOG=${SUMMARY_LOG:-"${RESULT_DIR}/summary.log"}

find_executable() {
    if [ -n "${EXEC:-}" ]; then
        echo "${EXEC}"
        return
    fi

    local candidates=(
        "${BUILD_DIR}/src/potrf/benchmark_potrf"
        "${BUILD_DIR}/benchmark_potrf"
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

NS=${NS:-"${DEFAULT_NS}"}
NBS=${NBS:-"${DEFAULT_NBS}"}

ERROR_MODE=${ERROR_MODE:-"${DEFAULT_ERROR_MODE}"}
TEST_ROCSOLVER=${TEST_ROCSOLVER:-1}
TEST_BLOCKED=${TEST_BLOCKED:-1}
TEST_RECURSIVE=${TEST_RECURSIVE:-1}

mkdir -p "${RESULT_DIR}"
: > "${SUMMARY_LOG}"

log() {
    echo "$*" | tee -a "${SUMMARY_LOG}"
}

log "=================================================="
log "POTRF benchmark"
log "DTK environment    : ${ROCM_PATH}"
log "Executable         : ${EXEC}"
log "Result directory   : ${RESULT_DIR}"
log "Run mode           : ${RUN_MODE} (smoke, sweep, perf)"
log "Precision mode     : ${PRECISION} (${PRECISION_NAME}; 0=FP32, 1=FP64, 2=mixed)"
log "Error mode         : ${ERROR_MODE} (0=none, 1=forward, 2=backward, 3=both)"
log "Test rocSOLVER     : ${TEST_ROCSOLVER}"
log "Test blocked RL    : ${TEST_BLOCKED}"
log "Test recursive     : ${TEST_RECURSIVE}"
log "N sweep            : ${NS}"
log "NB sweep           : ${NBS}"
log "Executable args    : <N> ${TEST_ROCSOLVER} ${TEST_RECURSIVE} ${ERROR_MODE} ${PRECISION} ${TEST_BLOCKED} ${NBS}"
log "=================================================="

if [ ! -x "${EXEC}" ]; then
    log "Error: executable not found or not executable: ${EXEC}"
    log "Build benchmark_potrf before running this script."
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

NUMA_NODE=$(cat /sys/class/drm/card0/device/numa_node 2>/dev/null || true)
if [ -n "${NUMA_NODE}" ] && [ "${NUMA_NODE}" != "-1" ] && command -v numactl >/dev/null 2>&1; then
    LAUNCH_PREFIX=(numactl "--cpunodebind=${NUMA_NODE}" "--membind=${NUMA_NODE}")
    log "NUMA binding enabled: node ${NUMA_NODE}"
else
    LAUNCH_PREFIX=()
    log "NUMA binding disabled."
fi

run_one_case() {
    local n=$1
    local out_file="${RESULT_DIR}/potrf_N${n}.log"

    log ""
    log ">>> Running POTRF benchmark: N=${n}"
    log "Output file: ${out_file}"

    if ! "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${n}" \
        "${TEST_ROCSOLVER}" \
        "${TEST_RECURSIVE}" \
        "${ERROR_MODE}" \
        "${PRECISION}" \
        "${TEST_BLOCKED}" \
        ${NBS} \
        2>&1 | tee "${out_file}"; then
        log "Error: benchmark failed for N=${n}."
        exit 1
    fi

    cat "${out_file}" >> "${SUMMARY_LOG}"
}

for n in ${NS}; do
    run_one_case "${n}"
done

log ""
log "=================================================="
log "POTRF benchmark finished"
log "Summary log: ${SUMMARY_LOG}"
log "=================================================="
