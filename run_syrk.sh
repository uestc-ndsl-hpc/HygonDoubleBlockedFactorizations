#!/usr/bin/env bash
set -u
set -o pipefail

ROCM_PATH=${ROCM_PATH:-/public/software/compiler/rocm/dtk-24.04}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

export ROCM_PATH
export HIP_VISIBLE_DEVICES
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ROCM_PATH}/bin:${PATH}"
export ROCBLAS_LAYER=0
export TENSILE_DB=0

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build"}
EXEC=${EXEC:-"${BUILD_DIR}/benchmark_syrk"}
RESULT_DIR=${RESULT_DIR:-"${ROOT_DIR}/results/fp32/syrk"}
SUMMARY_LOG=${SUMMARY_LOG:-"${RESULT_DIR}/summary.log"}

NS=${NS:-"8192 16384 24576 32768 40960 49152 57344 65536"}
USE_EQUAL_K=${USE_EQUAL_K:-1}
FIXED_K=${FIXED_K:-8192}
NBS=${NBS:-"256 512 1024 2048"}

ERROR_MODE=${ERROR_MODE:-2}
TEST_RECURSIVE=${TEST_RECURSIVE:-1}
TEST_ROCBLAS=${TEST_ROCBLAS:-1}
TEST_BATCHED=${TEST_BATCHED:-1}
TEST_MIXED=${TEST_MIXED:-0}

mkdir -p "${RESULT_DIR}"
: > "${SUMMARY_LOG}"

log() {
    echo "$*" | tee -a "${SUMMARY_LOG}"
}

log "=================================================="
log "SYRK benchmark"
log "DTK environment    : ${ROCM_PATH}"
log "Executable         : ${EXEC}"
log "Result directory   : ${RESULT_DIR}"
log "Error mode         : ${ERROR_MODE} (0=none, 1=forward, 2=backward, 3=both)"
log "Test rocBLAS       : ${TEST_ROCBLAS}"
log "Test recursive     : ${TEST_RECURSIVE}"
log "Test batched       : ${TEST_BATCHED}"
log "Test mixed         : ${TEST_MIXED}"
log "Use equal K        : ${USE_EQUAL_K}"
log "Fixed K            : ${FIXED_K}"
log "N sweep            : ${NS}"
log "NB sweep           : ${NBS}"
log "=================================================="

if [ ! -x "${EXEC}" ]; then
    log "Error: executable not found or not executable: ${EXEC}"
    log "Build benchmark_syrk before running this script."
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

resolve_k() {
    local n=$1
    if [ "${USE_EQUAL_K}" -eq 1 ]; then
        echo "${n}"
    else
        echo "${FIXED_K}"
    fi
}

run_one_case() {
    local n=$1
    local k
    k=$(resolve_k "${n}")
    local out_file="${RESULT_DIR}/syrk_N${n}_K${k}.log"

    log ""
    log ">>> Running SYRK benchmark: N=${n}, K=${k}"
    log "Output file: ${out_file}"

    if ! "${LAUNCH_PREFIX[@]}" "${EXEC}" \
        "${n}" \
        "${k}" \
        "${ERROR_MODE}" \
        "${TEST_RECURSIVE}" \
        "${TEST_ROCBLAS}" \
        "${TEST_BATCHED}" \
        "${TEST_MIXED}" \
        ${NBS} \
        2>&1 | tee "${out_file}"; then
        log "Error: benchmark failed for N=${n}, K=${k}."
        exit 1
    fi

    cat "${out_file}" >> "${SUMMARY_LOG}"
}

for n in ${NS}; do
    run_one_case "${n}"
done

log ""
log "=================================================="
log "SYRK benchmark finished"
log "Summary log: ${SUMMARY_LOG}"
log "=================================================="
