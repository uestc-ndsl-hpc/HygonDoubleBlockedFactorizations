#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_final/rocsolver_n49152_probe_6098e38_20260727T145925Z
DEVICE=1
N=49152

if [[ -e "${OUT}/started" ]]; then
    echo "Refusing to reuse completed or active probe directory: ${OUT}" >&2
    exit 2
fi

mkdir -p "${OUT}"
touch "${OUT}/started"

{
    echo "rocSOLVER N=49152 diagnostic probe"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=${ROOT}"
    echo "out=${OUT}"
    echo "physical_device=${DEVICE}"
    echo "n=${N}"
    echo "warmups=0"
    echo "measurements=1"
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "hipcc_version:"
    /opt/dtk/bin/hipcc --version 2>&1 || true
    echo
    echo "git_status:"
    git -C "${ROOT}" status --short
    echo
    echo "linked_libraries:"
    ldd "${ROOT}/build/benchmark_potrf"
    ldd "${ROOT}/build/benchmark_getrf"
    echo
    echo "device_before:"
    /opt/dtk/bin/rocm-smi --showpids 2>&1 || true
    /opt/dtk/bin/rocm-smi --showmeminfo vram --showuse 2>&1 || true
} > "${OUT}/environment.log" 2>&1

run_case() {
    local name=$1
    shift
    local log="${OUT}/${name}.log"
    local monitor="${OUT}/${name}.memory.log"
    local command_file="${OUT}/${name}.command"
    local exit_file="${OUT}/${name}.exit"

    printf '%q ' env HIP_VISIBLE_DEVICES="${DEVICE}" \
        BENCHMARK_WARMUP_ITERS=0 BENCHMARK_PROFILE_ITERS=1 "$@" \
        > "${command_file}"
    printf '\n' >> "${command_file}"

    {
        echo "case=${name}"
        echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "command=$(cat "${command_file}")"
    } > "${log}"

    set +e
    /usr/bin/time -v env \
        HIP_VISIBLE_DEVICES="${DEVICE}" \
        BENCHMARK_WARMUP_ITERS=0 \
        BENCHMARK_PROFILE_ITERS=1 \
        "$@" >> "${log}" 2>&1 &
    local pid=$!

    {
        echo "monitor_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "pid=${pid}"
        while kill -0 "${pid}" 2>/dev/null; do
            echo "sample_utc=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)"
            /opt/dtk/bin/rocm-smi --showpids 2>&1 || true
            /opt/dtk/bin/rocm-smi --showmeminfo vram --showuse 2>&1 || true
            sleep 1
        done
        echo "monitor_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${monitor}" 2>&1

    wait "${pid}"
    local rc=$?
    set -e

    {
        echo "${rc}"
    } > "${exit_file}"
    {
        echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "exit_code=${rc}"
    } >> "${log}"

    /opt/dtk/bin/rocm-smi --showpids >> "${log}" 2>&1 || true
    /opt/dtk/bin/rocm-smi --showmeminfo vram --showuse >> "${log}" 2>&1 || true
    sleep 5
}

run_case potrf_rocsolver_N49152 \
    "${ROOT}/build/benchmark_potrf" \
    "${N}" 0 0 2 0 0 0 2048 \
    --methods rocsolver

run_case getrf_rocsolver_N49152 \
    "${ROOT}/build/benchmark_getrf" \
    "${N}" 0 1 2048 \
    --pivot \
    --methods rocsolver \
    --pivot-tile 2048

{
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "potrf_exit=$(cat "${OUT}/potrf_rocsolver_N49152.exit")"
    echo "getrf_exit=$(cat "${OUT}/getrf_rocsolver_N49152.exit")"
    /opt/dtk/bin/rocm-smi --showpids 2>&1 || true
    /opt/dtk/bin/rocm-smi --showmeminfo vram --showuse 2>&1 || true
} > "${OUT}/summary.log" 2>&1

sha256sum "${OUT}"/* > "${OUT}/sha256sums.txt"
touch "${OUT}/completed"
