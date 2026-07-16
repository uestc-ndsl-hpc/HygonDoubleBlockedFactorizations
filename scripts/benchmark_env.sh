#!/bin/bash

setup_dtk_environment() {
    export ROCM_PATH=${ROCM_PATH:-/opt/dtk}
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
    export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${LD_LIBRARY_PATH:-}
    export PATH=${ROCM_PATH}/bin:${PATH:-}

    export ROCBLAS_LAYER=${ROCBLAS_LAYER:-0}
    export TENSILE_DB=${TENSILE_DB:-0}
    export BENCHMARK_WARMUP_ITERS=${BENCHMARK_WARMUP_ITERS:-1}
    export BENCHMARK_PROFILE_ITERS=${BENCHMARK_PROFILE_ITERS:-3}

    return 0
}

log_toolchain() {
    local summary_log=$1
    local root_dir=${2:-$(pwd)}

    echo "Timestamp          : $(date --iso-8601=seconds 2>/dev/null || date)" | tee -a "${summary_log}"
    echo "Hostname           : $(hostname 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "Kernel             : $(uname -srmo 2>/dev/null || uname -a 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "Git commit         : $(git -C "${root_dir}" rev-parse HEAD 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "Git describe       : $(git -C "${root_dir}" describe --always --dirty 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "DTK Environment    : ${ROCM_PATH}" | tee -a "${summary_log}"
    echo "HIP device         : ${HIP_VISIBLE_DEVICES}" | tee -a "${summary_log}"
    echo "Warmup iterations  : ${BENCHMARK_WARMUP_ITERS}" | tee -a "${summary_log}"
    echo "Profile iterations : ${BENCHMARK_PROFILE_ITERS}" | tee -a "${summary_log}"
    echo "hipcc path         : $(command -v hipcc 2>/dev/null || echo unavailable)" | tee -a "${summary_log}"
    echo "hipcc version      : $(hipcc --version 2>/dev/null | head -n 1 || echo unavailable)" | tee -a "${summary_log}"
    echo "cmake version      : $(cmake --version 2>/dev/null | head -n 1 || echo unavailable)" | tee -a "${summary_log}"

    if command -v rocm-smi >/dev/null 2>&1; then
        echo "ROCm device state:" | tee -a "${summary_log}"
        rocm-smi --showproductname --showuse --showmemuse --showpids \
            2>&1 | tee -a "${summary_log}" || true
    fi

    return 0
}

check_benchmark_binary() {
    local exec_path=$1
    local summary_log=$2
    local lib_pattern=$3

    echo "" | tee -a "${summary_log}"
    echo "Checking binary dependencies..." | tee -a "${summary_log}"

    if [ -f "${exec_path}" ]; then
        ldd "${exec_path}" | grep "not found" | tee -a "${summary_log}" || true
        echo "Resolved ROCm libraries:" | tee -a "${summary_log}"
        ldd "${exec_path}" | grep -E "${lib_pattern}" | tee -a "${summary_log}" || true
    else
        echo "Error: Executable ${exec_path} not found! Please build first." | tee -a "${summary_log}"
        exit 1
    fi

    return 0
}

configure_numa_launch_prefix() {
    local summary_log=$1
    local visible_device=${HIP_VISIBLE_DEVICES%%,*}
    local drm_card=${BENCHMARK_DRM_CARD:-${visible_device}}
    local numa_path=/sys/class/drm/card${drm_card}/device/numa_node
    local numa_node=""

    LAUNCH_PREFIX=()

    if [ "${BENCHMARK_NUMA_BIND:-1}" = "0" ]; then
        echo "NUMA binding       : disabled" | tee -a "${summary_log}"
        return 0
    fi

    if [ -r "${numa_path}" ]; then
        numa_node=$(cat "${numa_path}")
    fi

    if [[ "${numa_node}" =~ ^[0-9]+$ ]] && command -v numactl >/dev/null 2>&1; then
        LAUNCH_PREFIX=(
            numactl
            "--cpunodebind=${numa_node}"
            "--membind=${numa_node}"
        )
        echo "NUMA binding       : card${drm_card} -> node ${numa_node}" | tee -a "${summary_log}"
    else
        echo "NUMA binding       : unavailable for card${drm_card}" | tee -a "${summary_log}"
    fi

    return 0
}
