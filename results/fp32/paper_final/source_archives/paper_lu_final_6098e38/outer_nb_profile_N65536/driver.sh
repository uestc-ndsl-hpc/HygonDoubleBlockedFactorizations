#!/usr/bin/env bash

set -uo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_lu_final_6098e38/outer_nb_profile_N65536
EXEC=${ROOT}/build/profile_double_blocked_getrf

mkdir -p "${OUT}"
date -Is > "${OUT}/driver.started"

run_profile() {
    local label=$1
    local outer_nb=$2
    local prefix="${OUT}/${label}"
    local -a cmd=(
        env CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1
        numactl --cpunodebind=0 --membind=0
        "${EXEC}"
        --n 65536
        --outer-nb "${outer_nb}"
        --inner-nb 4096
        --outer-trsm iterative
        --trsm-block 2048
        --pivot-tile 2048
        --inner-gemm-m-tile 8192
        --pivot-mode batched
        --warmup 1
        --iters 3
    )

    printf '%q ' "${cmd[@]}" > "${prefix}.command"
    printf '\n' >> "${prefix}.command"
    date -Is > "${prefix}.started"
    if "${cmd[@]}" > "${prefix}.log" 2>&1; then
        printf '0\n' > "${prefix}.exit"
        date -Is > "${prefix}.completed"
    else
        rc=$?
        printf '%s\n' "${rc}" > "${prefix}.exit"
        date -Is > "${prefix}.failed"
        return "${rc}"
    fi
}

run_profile control_B32768 32768 || exit $?

pids=$(fuser /dev/kfd 2>/dev/null || true)
printf '%s\n' "${pids}" > "${OUT}/between-profiles.kfd-pids.txt"
if [[ -n "${pids//[[:space:]]/}" ]]; then
    date -Is > "${OUT}/between-profiles.blocked"
    exit 3
fi

run_profile capped_B20480 20480 || exit $?

for label in control_B32768 capped_B20480; do
    if [[ ! -s "${OUT}/${label}.log" ]]; then
        date -Is > "${OUT}/driver.artifacts-missing"
        exit 4
    fi
done

printf '0\n' > "${OUT}/driver.exit"
date -Is > "${OUT}/driver.completed"
