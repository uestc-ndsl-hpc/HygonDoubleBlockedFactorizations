#!/usr/bin/env bash

set -uo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_lu_final_6098e38/block_collapse_suite

mkdir -p "${OUT}"
date -Is > "${OUT}/driver.started"
printf '%s\n' \
    'env CUDA_VISIBLE_DEVICES=1 PAPER_DEVICE=1 PAPER_CONFIRM_IDLE=1 PAPER_FACTORIZATIONS=getrf PAPER_SECTIONS=block-collapse PAPER_BLOCK_COLLAPSE_NS="8192 16384 24576 32768 40960 49152 57344 65536" PAPER_BLOCK_COLLAPSE_WARMUP=2 PAPER_BLOCK_COLLAPSE_ITERS=10 PAPER_RESULT_DIR=/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_lu_final_6098e38/block_collapse_suite /home/uestc_zss/HygonCholeskyDecomposition/run_paper_experiments.sh' \
    > "${OUT}/driver.command"

if env \
    CUDA_VISIBLE_DEVICES=1 \
    PAPER_DEVICE=1 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_FACTORIZATIONS=getrf \
    PAPER_SECTIONS=block-collapse \
    PAPER_BLOCK_COLLAPSE_NS="8192 16384 24576 32768 40960 49152 57344 65536" \
    PAPER_BLOCK_COLLAPSE_WARMUP=2 \
    PAPER_BLOCK_COLLAPSE_ITERS=10 \
    PAPER_RESULT_DIR="${OUT}" \
    "${ROOT}/run_paper_experiments.sh" \
    > "${OUT}/driver.log" 2>&1; then
    printf '0\n' > "${OUT}/driver.exit"
else
    rc=$?
    printf '%s\n' "${rc}" > "${OUT}/driver.exit"
    date -Is > "${OUT}/driver.failed"
    exit "${rc}"
fi

for n in 8192 16384 24576 32768 40960 49152 57344 65536; do
    for mode in final_b b_eq_nb; do
        if [[ ! -s "${OUT}/block_collapse/getrf_${mode}_N${n}.log" \
              || ! -s "${OUT}/block_collapse/getrf_${mode}_N${n}_timings.csv" ]]; then
            date -Is > "${OUT}/driver.artifacts-missing"
            exit 4
        fi
    done
done

date -Is > "${OUT}/driver.completed"
