#!/usr/bin/env bash

set +e

env \
    PAPER_SECTIONS=block-collapse \
    PAPER_DEVICE=1 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_RESULT_DIR=/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_final/block_collapse_ce7690b_20260718T050406Z/performance_retry1 \
    PAPER_BLOCK_COLLAPSE_NS="8192 16384 24576 32768 40960" \
    PAPER_BLOCK_COLLAPSE_WARMUP=2 \
    PAPER_BLOCK_COLLAPSE_ITERS=10 \
    bash /home/uestc_zss/HygonCholeskyDecomposition/run_paper_experiments.sh
rc=$?
printf '%s\n' "${rc}" > "/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_final/block_collapse_ce7690b_20260718T050406Z/performance_retry1/driver.exit"
if [[ ${rc} -eq 0 ]]; then
    touch "/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_final/block_collapse_ce7690b_20260718T050406Z/performance_retry1/completed"
else
    touch "/home/uestc_zss/HygonCholeskyDecomposition/results/fp32/paper_final/block_collapse_ce7690b_20260718T050406Z/performance_retry1/failed"
fi
exit "${rc}"
