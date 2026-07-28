#!/usr/bin/env bash

set -euo pipefail

REPO=/home/uestc_zss/HygonCholeskyDecomposition
ROOT=${REPO}/results/fp32/paper_final/getrf_large_profiles_df1eb08
BIN=${REPO}/build/profile_double_blocked_getrf
EXPECTED_HEAD=df1eb08b18e33110619d8286ed7a0cecb84ae272

mkdir -p "${ROOT}/profiles"

if [[ -e "${ROOT}/completed" || -e "${ROOT}/running.pid" ]]; then
    echo "Refusing duplicate launch: completion or running marker already exists." >&2
    exit 3
fi

echo "$$" > "${ROOT}/running.pid"
date --iso-8601=seconds > "${ROOT}/started"

/usr/local/hyhal/bin/hy-smi > "${ROOT}/device_precheck.log" 2>&1
/opt/dtk/bin/rocm-smi --showpids >> "${ROOT}/device_precheck.log" 2>&1

if ! grep -q "No KFD PIDs currently running" "${ROOT}/device_precheck.log"; then
    echo "KFD activity appeared before launch; no profile was started." >&2
    date --iso-8601=seconds > "${ROOT}/failed_precheck"
    exit 4
fi

head_commit=$(git -C "${REPO}" rev-parse HEAD)
if [[ "${head_commit}" != "${EXPECTED_HEAD}" ]]; then
    echo "Unexpected repository HEAD: ${head_commit}" >&2
    date --iso-8601=seconds > "${ROOT}/failed_head"
    exit 5
fi

if [[ -n "$(git -C "${REPO}" status --porcelain=v1)" ]]; then
    echo "Remote repository is not clean; no profile was started." >&2
    git -C "${REPO}" status --short > "${ROOT}/failed_git_status.log"
    date --iso-8601=seconds > "${ROOT}/failed_git"
    exit 6
fi

if [[ ! -x "${BIN}" ]]; then
    echo "Missing executable: ${BIN}" >&2
    date --iso-8601=seconds > "${ROOT}/failed_binary"
    exit 7
fi

export HIP_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=1
export ROCM_PATH=/opt/dtk
export BENCHMARK_WARMUP_ITERS=1
export BENCHMARK_PROFILE_ITERS=3

source "${REPO}/scripts/benchmark_env.sh"
setup_dtk_environment
configure_numa_launch_prefix "${ROOT}/manifest.log"

{
    echo "GETRF large-shape stage profiles"
    echo "Timestamp          : $(date --iso-8601=seconds)"
    echo "Hostname           : $(hostname)"
    echo "Git commit         : ${head_commit}"
    echo "Git status         : clean"
    echo "Binary             : ${BIN}"
    echo "Binary SHA256      : $(sha256sum "${BIN}" | awk '{print $1}')"
    echo "Physical HCU       : 1"
    echo "Sizes              : 57344 65536"
    echo "Outer block policy : N/2"
    echo "Inner block        : 4096"
    echo "Outer TRSM         : iterative/2048"
    echo "Pivot mode/tile    : batched/2048"
    echo "Inner GEMM M tile  : 8192"
    echo "Warmup/iterations  : 1/3"
} >> "${ROOT}/manifest.log"

for n in 57344 65536; do
    outer_nb=$((n / 2))
    log="${ROOT}/profiles/getrf_final_N${n}.log"
    command_file="${ROOT}/profiles/getrf_final_N${n}.command"
    exit_file="${ROOT}/profiles/getrf_final_N${n}.exit"

    printf '%q ' "${BIN}" \
        --n "${n}" \
        --outer-nb "${outer_nb}" \
        --inner-nb 4096 \
        --outer-trsm iterative \
        --trsm-block 2048 \
        --pivot-tile 2048 \
        --inner-gemm-m-tile 8192 \
        --pivot-mode batched \
        --warmup 1 \
        --iters 3 > "${command_file}"
    printf '\n' >> "${command_file}"

    echo "Starting N=${n} at $(date --iso-8601=seconds)" >> "${ROOT}/manifest.log"
    set +e
    "${LAUNCH_PREFIX[@]}" "${BIN}" \
        --n "${n}" \
        --outer-nb "${outer_nb}" \
        --inner-nb 4096 \
        --outer-trsm iterative \
        --trsm-block 2048 \
        --pivot-tile 2048 \
        --inner-gemm-m-tile 8192 \
        --pivot-mode batched \
        --warmup 1 \
        --iters 3 > "${log}" 2>&1
    rc=$?
    set -e

    echo "${rc}" > "${exit_file}"
    if [[ ${rc} -ne 0 || ! -s "${log}" ]]; then
        echo "N=${n} failed with exit=${rc}" >> "${ROOT}/manifest.log"
        date --iso-8601=seconds > "${ROOT}/failed"
        if [[ ${rc} -ne 0 ]]; then
            exit "${rc}"
        fi
        exit 8
    fi
    echo "Completed N=${n} at $(date --iso-8601=seconds)" >> "${ROOT}/manifest.log"
done

sha256sum "${ROOT}"/profiles/*.log > "${ROOT}/profile_log_sha256.txt"
echo "0" > "${ROOT}/exit_code"
date --iso-8601=seconds > "${ROOT}/completed"
echo "All large-shape GETRF profiles completed." >> "${ROOT}/manifest.log"
