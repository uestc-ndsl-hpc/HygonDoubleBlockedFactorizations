#!/bin/bash

setup_dtk_environment() {
    export ROCM_PATH=${ROCM_PATH:-/opt/dtk}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
    export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}
    export PATH=$ROCM_PATH/bin:$PATH

    export ROCBLAS_LAYER=${ROCBLAS_LAYER:-0}
    export TENSILE_DB=${TENSILE_DB:-0}
}

log_toolchain() {
    local summary_log=$1

    echo "DTK Environment    : ${ROCM_PATH}" | tee -a "${summary_log}"
    echo "hipcc path         : $(command -v hipcc 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "hipcc version      : $(hipcc --version 2>/dev/null | head -n 1 || echo unavailable)" | tee -a "${summary_log}"
}

check_benchmark_binary() {
    local exec_path=$1
    local summary_log=$2
    local lib_pattern=$3

    echo "" | tee -a "${summary_log}"
    echo "Checking binary dependencies..." | tee -a "${summary_log}"

    if [ -f "${exec_path}" ]; then
        ldd "${exec_path}" | grep "not found" | tee -a "${summary_log}"
        echo "Resolved ROCm libraries:" | tee -a "${summary_log}"
        ldd "${exec_path}" | grep -E "${lib_pattern}" | tee -a "${summary_log}"
    else
        echo "Error: Executable ${exec_path} not found! Please build first." | tee -a "${summary_log}"
        exit 1
    fi
}

configure_numa_launch_prefix() {
    local summary_log=$1

    NUMA_NODE=$(cat /sys/class/drm/card0/device/numa_node 2>/dev/null)
    if [ "${NUMA_NODE}" != "-1" ] && [ ! -z "${NUMA_NODE}" ]; then
        LAUNCH_PREFIX="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
        echo "NUMA Binding detected and enabled: ${LAUNCH_PREFIX}" | tee -a "${summary_log}"
    else
        LAUNCH_PREFIX=""
        echo "Warning: No direct NUMA affinity mapping found. Running without numactl." | tee -a "${summary_log}"
    fi
}
