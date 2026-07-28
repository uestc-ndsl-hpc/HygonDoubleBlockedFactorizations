#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_coupled_threeway_paired_6098e38
POTRF=${ROOT}/build/benchmark_potrf
GETRF=${ROOT}/build/benchmark_getrf
EXPECTED_HEAD=6098e38e1f8f5e56beaeb1446df4bc207455cb12
NS=(8192 16384 24576 32768 40960 49152 57344 65536)

mkdir -p "${OUT}"
cd "${ROOT}"

printf '%s\n' "$$" > "${OUT}/driver.pid"
readlink "/proc/$$/cwd" > "${OUT}/driver.cwd.txt"
ps -p "$$" -o pid,ppid,user,stat,etime,lstart,args > "${OUT}/driver.process.txt"
date -Is > "${OUT}/driver.started"

finish() {
    local rc=$?
    printf '%s\n' "${rc}" > "${OUT}/driver.exit"
    if (( rc == 0 )); then
        date -Is > "${OUT}/driver.completed"
    else
        date -Is > "${OUT}/driver.failed"
    fi
}
trap finish EXIT

source "${ROOT}/scripts/benchmark_env.sh"
setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

export CUDA_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES=1

git -C "${ROOT}" rev-parse HEAD > "${OUT}/commit.txt"
git -C "${ROOT}" status --short > "${OUT}/git_status.txt"
sha256sum "${POTRF}" "${GETRF}" > "${OUT}/binaries.sha256"
ldd "${POTRF}" > "${OUT}/benchmark_potrf.ldd"
ldd "${GETRF}" > "${OUT}/benchmark_getrf.ldd"
/opt/dtk/bin/hipcc --version > "${OUT}/compiler.txt"
configure_numa_launch_prefix "${OUT}/manifest.log"

if [[ "$(cat "${OUT}/commit.txt")" != "${EXPECTED_HEAD}" ]]; then
    echo "Unexpected Git HEAD" >&2
    exit 2
fi
if [[ -s "${OUT}/git_status.txt" ]]; then
    echo "Remote worktree is dirty" >&2
    exit 3
fi

check_idle() {
    local label=$1
    local pids
    pids=$(fuser /dev/kfd 2>/dev/null || true)
    printf '%s\n' "${pids}" > "${OUT}/${label}.kfd-pids.txt"
    if [[ -n "${pids//[[:space:]]/}" ]]; then
        echo "KFD is busy before ${label}: ${pids}" >&2
        return 4
    fi
}

run_case() {
    local label=$1
    local csv=$2
    local log=$3
    shift 3
    local prefix="${OUT}/${label}"
    local -a cmd=("$@")

    check_idle "before-${label//\//_}"
    mkdir -p "$(dirname "${csv}")" "$(dirname "${log}")"
    printf '%q ' "${cmd[@]}" > "${prefix}.command"
    printf '\n' >> "${prefix}.command"
    date -Is > "${prefix}.started"

    if "${cmd[@]}" > "${log}" 2>&1; then
        printf '0\n' > "${prefix}.exit"
        date -Is > "${prefix}.completed"
    else
        local rc=$?
        printf '%s\n' "${rc}" > "${prefix}.exit"
        date -Is > "${prefix}.failed"
        return "${rc}"
    fi
}

mkdir -p "${OUT}/accuracy" "${OUT}/cholesky/scaling" "${OUT}/lu/scaling"

run_case \
    accuracy/cholesky_threeway \
    "${OUT}/accuracy/cholesky_threeway.csv" \
    "${OUT}/accuracy/cholesky_threeway.log" \
    env \
    CUDA_VISIBLE_DEVICES=1 \
    HIP_VISIBLE_DEVICES=1 \
    BENCHMARK_WARMUP_ITERS=0 \
    BENCHMARK_PROFILE_ITERS=1 \
    BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR=1e-5 \
    "BENCHMARK_CSV_PATH=${OUT}/accuracy/cholesky_threeway.csv" \
    "${LAUNCH_PREFIX[@]}" \
    "${POTRF}" \
    16384 0 0 2 0 0 0 2048 \
    --methods mgemm \
    --double-nb 8192 1024 \
    --b 1024 8192 \
    --syrk-k-tile 8192 \
    --gemm-m-tile 8192 \
    --gemm-m-threshold 32768 \
    --gemm-k-threshold 5632 \
    --full-backward-error \
    --full-backward-error-max-n 16384

for n in "${NS[@]}"; do
    csv="${OUT}/cholesky/scaling/potrf_N${n}_timings.csv"
    log="${OUT}/cholesky/scaling/potrf_N${n}.log"
    run_case \
        "cholesky/scaling/potrf_N${n}" \
        "${csv}" \
        "${log}" \
        env \
        CUDA_VISIBLE_DEVICES=1 \
        HIP_VISIBLE_DEVICES=1 \
        BENCHMARK_WARMUP_ITERS=2 \
        BENCHMARK_PROFILE_ITERS=10 \
        "BENCHMARK_CSV_PATH=${csv}" \
        "${LAUNCH_PREFIX[@]}" \
        "${POTRF}" \
        "${n}" 0 0 0 0 0 0 2048 \
        --methods mgemm \
        --double-nb 8192 1024 \
        --b 1024 8192 \
        --syrk-k-tile 8192 \
        --gemm-m-tile 8192 \
        --gemm-m-threshold 32768 \
        --gemm-k-threshold 5632
done

run_case \
    accuracy/lu_coupled_small \
    "${OUT}/accuracy/lu_coupled_small.csv" \
    "${OUT}/accuracy/lu_coupled_small.log" \
    env \
    CUDA_VISIBLE_DEVICES=1 \
    HIP_VISIBLE_DEVICES=1 \
    BENCHMARK_WARMUP_ITERS=0 \
    BENCHMARK_PROFILE_ITERS=1 \
    BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR=5e-4 \
    "BENCHMARK_CSV_PATH=${OUT}/accuracy/lu_coupled_small.csv" \
    "${LAUNCH_PREFIX[@]}" \
    "${GETRF}" \
    16384 0 1 2048 \
    --pivot \
    --methods mgemm \
    --double-nb 4096 \
    --b 4096 \
    --outer-trsm iterative \
    --trsm-block 2048 \
    --pivot-tile 2048 \
    --inner-gemm-m-tile 8192 \
    --full-backward-error \
    --full-backward-error-max-n 16384

for n in "${NS[@]}"; do
    block=4096
    if (( n == 8192 )); then
        block=2048
    fi
    csv="${OUT}/lu/scaling/getrf_N${n}_timings.csv"
    log="${OUT}/lu/scaling/getrf_N${n}.log"
    run_case \
        "lu/scaling/getrf_N${n}" \
        "${csv}" \
        "${log}" \
        env \
        CUDA_VISIBLE_DEVICES=1 \
        HIP_VISIBLE_DEVICES=1 \
        BENCHMARK_WARMUP_ITERS=2 \
        BENCHMARK_PROFILE_ITERS=10 \
        "BENCHMARK_CSV_PATH=${csv}" \
        "${LAUNCH_PREFIX[@]}" \
        "${GETRF}" \
        "${n}" 0 0 2048 \
        --pivot \
        --methods mgemm \
        --double-nb "${block}" \
        --b "${block}" \
        --outer-trsm iterative \
        --trsm-block 2048 \
        --pivot-tile 2048 \
        --inner-gemm-m-tile 8192
done

for n in "${NS[@]}"; do
    cholesky_csv="${OUT}/cholesky/scaling/potrf_N${n}_timings.csv"
    lu_csv="${OUT}/lu/scaling/getrf_N${n}_timings.csv"
    [[ $(($(wc -l < "${cholesky_csv}") - 1)) -eq 30 ]]
    [[ $(($(wc -l < "${lu_csv}") - 1)) -eq 10 ]]
done
[[ $(($(wc -l < "${OUT}/accuracy/cholesky_threeway.csv") - 1)) -eq 3 ]]
[[ $(($(wc -l < "${OUT}/accuracy/lu_coupled_small.csv") - 1)) -eq 1 ]]

/usr/local/hyhal/bin/hy-smi > "${OUT}/postcheck.hy-smi.txt"
/opt/dtk/bin/rocm-smi --showpids > "${OUT}/postcheck.rocm-smi-pids.txt"
check_idle postcheck
git -C "${ROOT}" rev-parse HEAD > "${OUT}/postcheck.commit.txt"
git -C "${ROOT}" status --short > "${OUT}/postcheck.git-status.txt"
if [[ -s "${OUT}/postcheck.git-status.txt" ]]; then
    echo "Remote worktree became dirty" >&2
    exit 5
fi

