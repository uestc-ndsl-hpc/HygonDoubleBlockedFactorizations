#!/usr/bin/env bash

set -uo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_lu_final_6098e38/accuracy_suite
SMALL=${OUT}/small
LARGE=${OUT}/large_N49152

mkdir -p "${SMALL}" "${LARGE}"
date -Is > "${OUT}/driver.started"

printf '%s\n' \
    'env CUDA_VISIBLE_DEVICES=1 PAPER_DEVICE=1 PAPER_CONFIRM_IDLE=1 PAPER_FACTORIZATIONS=getrf PAPER_SECTIONS=accuracy PAPER_ACCURACY_NS="1024 2048 4096 8192 16384 24576" PAPER_FULL_BACKWARD_ERROR_MAX_N=24576 PAPER_ACCURACY_WARMUP=0 PAPER_ACCURACY_ITERS=1 PAPER_RESULT_DIR=/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_lu_final_6098e38/accuracy_suite/small /home/uestc_zss/HygonCholeskyDecomposition/run_paper_experiments.sh' \
    > "${SMALL}/driver.command"

if env \
    CUDA_VISIBLE_DEVICES=1 \
    PAPER_DEVICE=1 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_FACTORIZATIONS=getrf \
    PAPER_SECTIONS=accuracy \
    PAPER_ACCURACY_NS="1024 2048 4096 8192 16384 24576" \
    PAPER_FULL_BACKWARD_ERROR_MAX_N=24576 \
    PAPER_ACCURACY_WARMUP=0 \
    PAPER_ACCURACY_ITERS=1 \
    PAPER_RESULT_DIR="${SMALL}" \
    "${ROOT}/run_paper_experiments.sh" \
    > "${SMALL}/driver.log" 2>&1; then
    printf '0\n' > "${SMALL}/driver.exit"
    date -Is > "${SMALL}/driver.completed"
else
    rc=$?
    printf '%s\n' "${rc}" > "${SMALL}/driver.exit"
    date -Is > "${SMALL}/driver.failed"
    exit "${rc}"
fi

pids=$(fuser /dev/kfd 2>/dev/null || true)
printf '%s\n' "${pids}" > "${OUT}/between-phases.kfd-pids.txt"
if [[ -n "${pids//[[:space:]]/}" ]]; then
    date -Is > "${OUT}/between-phases.blocked"
    exit 3
fi

printf '%s\n' \
    'env CUDA_VISIBLE_DEVICES=1 PAPER_DEVICE=1 PAPER_CONFIRM_IDLE=1 PAPER_FACTORIZATIONS=getrf PAPER_SECTIONS=accuracy PAPER_ACCURACY_NS=49152 PAPER_FULL_BACKWARD_ERROR_MAX_N=49152 PAPER_GETRF_METHODS=mgemm PAPER_GETRF_LARGE_METHODS=mgemm PAPER_ACCURACY_WARMUP=0 PAPER_ACCURACY_ITERS=1 PAPER_RESULT_DIR=/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_lu_final_6098e38/accuracy_suite/large_N49152 /home/uestc_zss/HygonCholeskyDecomposition/run_paper_experiments.sh' \
    > "${LARGE}/driver.command"

if env \
    CUDA_VISIBLE_DEVICES=1 \
    PAPER_DEVICE=1 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_FACTORIZATIONS=getrf \
    PAPER_SECTIONS=accuracy \
    PAPER_ACCURACY_NS=49152 \
    PAPER_FULL_BACKWARD_ERROR_MAX_N=49152 \
    PAPER_GETRF_METHODS=mgemm \
    PAPER_GETRF_LARGE_METHODS=mgemm \
    PAPER_ACCURACY_WARMUP=0 \
    PAPER_ACCURACY_ITERS=1 \
    PAPER_RESULT_DIR="${LARGE}" \
    "${ROOT}/run_paper_experiments.sh" \
    > "${LARGE}/driver.log" 2>&1; then
    printf '0\n' > "${LARGE}/driver.exit"
    date -Is > "${LARGE}/driver.completed"
else
    rc=$?
    printf '%s\n' "${rc}" > "${LARGE}/driver.exit"
    date -Is > "${LARGE}/driver.failed"
    exit "${rc}"
fi

for n in 1024 2048 4096 8192 16384 24576; do
    if [[ ! -s "${SMALL}/accuracy/getrf_N${n}.log" \
          || ! -s "${SMALL}/accuracy/getrf_N${n}_timings.csv" ]]; then
        date -Is > "${OUT}/driver.artifacts-missing"
        exit 4
    fi
done
if [[ ! -s "${LARGE}/accuracy/getrf_N49152.log" \
      || ! -s "${LARGE}/accuracy/getrf_N49152_timings.csv" ]]; then
    date -Is > "${OUT}/driver.artifacts-missing"
    exit 4
fi

printf '0\n' > "${OUT}/driver.exit"
date -Is > "${OUT}/driver.completed"
