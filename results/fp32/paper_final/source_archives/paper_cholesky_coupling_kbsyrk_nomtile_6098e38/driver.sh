#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_cholesky_coupling_kbsyrk_nomtile_6098e38
POTRF=${ROOT}/build/benchmark_potrf
EXPECTED_HEAD=6098e38e1f8f5e56beaeb1446df4bc207455cb12
NS=(8192 16384 24576 32768 40960 49152 57344 65536)
EXPECTED_PARAMETERS=(8192/1024/8192 8192/8192/8192 1024/1024/8192)

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
sha256sum "${POTRF}" > "${OUT}/binary.sha256"
ldd "${POTRF}" > "${OUT}/benchmark_potrf.ldd"
/opt/dtk/bin/hipcc --version > "${OUT}/compiler.txt"
configure_numa_launch_prefix "${OUT}/manifest.log"
printf '%s\n' \
    'method=DoubleBlock+KBSYRK' \
    'syrk_k_tile=8192' \
    'gemm_m_tile=0' \
    'configurations=8192/1024,8192/8192,1024/1024' \
    'warmup=2' \
    'measurements=10' > "${OUT}/policy.txt"

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
    local log=$2
    shift 2
    local prefix="${OUT}/${label}"
    local -a cmd=("$@")

    check_idle "before-${label//\//_}"
    mkdir -p "$(dirname "${log}")"
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

mkdir -p "${OUT}/accuracy" "${OUT}/scaling"

accuracy_csv="${OUT}/accuracy/cholesky_threeway.csv"
run_case \
    accuracy/cholesky_threeway \
    "${OUT}/accuracy/cholesky_threeway.log" \
    env \
    CUDA_VISIBLE_DEVICES=1 \
    HIP_VISIBLE_DEVICES=1 \
    BENCHMARK_WARMUP_ITERS=0 \
    BENCHMARK_PROFILE_ITERS=1 \
    BENCHMARK_MAX_NORMWISE_BACKWARD_ERROR=1e-5 \
    "BENCHMARK_CSV_PATH=${accuracy_csv}" \
    "${LAUNCH_PREFIX[@]}" \
    "${POTRF}" \
    16384 0 0 2 0 0 0 2048 \
    --methods kbsyrk \
    --double-nb 8192 1024 \
    --b 1024 8192 \
    --syrk-k-tile 8192 \
    --gemm-m-tile 0 \
    --gemm-m-threshold 0 \
    --gemm-k-threshold 0 \
    --full-backward-error \
    --full-backward-error-max-n 16384

for n in "${NS[@]}"; do
    csv="${OUT}/scaling/potrf_N${n}_timings.csv"
    log="${OUT}/scaling/potrf_N${n}.log"
    run_case \
        "scaling/potrf_N${n}" \
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
        --methods kbsyrk \
        --double-nb 8192 1024 \
        --b 1024 8192 \
        --syrk-k-tile 8192 \
        --gemm-m-tile 0 \
        --gemm-m-threshold 0 \
        --gemm-k-threshold 0
done

[[ $(($(wc -l < "${accuracy_csv}") - 1)) -eq 3 ]]
! grep -q 'MGEMM' "${accuracy_csv}"
for parameters in "${EXPECTED_PARAMETERS[@]}"; do
    [[ $(grep -c "\"${parameters}\"" "${accuracy_csv}") -eq 1 ]]
done
awk -F, '
    NR > 1 {
        gsub(/"/, "", $2)
        gsub(/"/, "", $14)
        if ($2 != "DoubleBlock+KBSYRK" ||
            $14 != "normwise_backward_error" ||
            $15 !~ /^[0-9.eE+-]+$/ ||
            $15 + 0 > 1e-5) {
            exit 1
        }
    }
' "${accuracy_csv}"

for n in "${NS[@]}"; do
    csv="${OUT}/scaling/potrf_N${n}_timings.csv"
    [[ $(($(wc -l < "${csv}") - 1)) -eq 30 ]]
    [[ $(grep -c '"DoubleBlock+KBSYRK"' "${csv}") -eq 30 ]]
    ! grep -q 'MGEMM' "${csv}"
    for parameters in "${EXPECTED_PARAMETERS[@]}"; do
        [[ $(grep -c "\"${parameters}\"" "${csv}") -eq 10 ]]
    done
done

if find "${OUT}" -type f -name '*.failed' -print -quit | grep -q .; then
    echo "Failure marker found" >&2
    exit 6
fi

/usr/local/hyhal/bin/hy-smi > "${OUT}/postcheck.hy-smi.txt"
/opt/dtk/bin/rocm-smi --showpids > "${OUT}/postcheck.rocm-smi-pids.txt"
check_idle postcheck
git -C "${ROOT}" rev-parse HEAD > "${OUT}/postcheck.commit.txt"
git -C "${ROOT}" status --short > "${OUT}/postcheck.git-status.txt"
if [[ -s "${OUT}/postcheck.git-status.txt" ]]; then
    echo "Remote worktree became dirty" >&2
    exit 5
fi

