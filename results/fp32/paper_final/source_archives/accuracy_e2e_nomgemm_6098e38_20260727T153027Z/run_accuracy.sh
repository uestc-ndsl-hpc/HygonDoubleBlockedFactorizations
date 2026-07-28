#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
RUN_ID=accuracy_e2e_nomgemm_6098e38_20260727T153027Z
RESULT_ROOT=${ROOT}/results/fp32/paper_final/${RUN_ID}
ARCHIVE=${ROOT}/results/server_archives/${RUN_ID}.tar.gz

mkdir -p "${RESULT_ROOT}"

finish() {
    local rc=$?
    printf '%s\n' "${rc}" > "${RESULT_ROOT}/driver.exit"
    if (( rc == 0 )); then
        touch "${RESULT_ROOT}/completed"
    fi
}
trap finish EXIT

{
    echo "Run ID: ${RUN_ID}"
    echo "Purpose: full normwise residuals on the end-to-end comparison grid"
    echo "POTRF policy: rocSOLVER and DoubleBlock+KBSYRK, gemm_m_tile=0"
    echo "GETRF addition: n=32768 only, to complete the five-size table"
    echo "Git head: $(git -C "${ROOT}" rev-parse HEAD)"
    echo "Start: $(date --iso-8601=seconds)"
} > "${RESULT_ROOT}/run_manifest.txt"

active_pids=$(fuser /dev/kfd 2>/dev/null || true)
if [[ -n "${active_pids//[[:space:]]/}" ]]; then
    echo "Refusing to start: /dev/kfd is in use by ${active_pids}" \
        | tee -a "${RESULT_ROOT}/run_manifest.txt" >&2
    exit 20
fi

if [[ -n "$(git -C "${ROOT}" status --porcelain)" ]]; then
    echo "Refusing to start: remote worktree is dirty." \
        | tee -a "${RESULT_ROOT}/run_manifest.txt" >&2
    exit 21
fi

env \
    PAPER_SECTIONS=accuracy \
    PAPER_FACTORIZATIONS=potrf \
    PAPER_DEVICE=7 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_ACCURACY_NS="8192 16384 24576 32768 40960" \
    PAPER_ACCURACY_WARMUP=0 \
    PAPER_ACCURACY_ITERS=1 \
    PAPER_FULL_BACKWARD_ERROR_MAX_N=40960 \
    PAPER_POTRF_METHODS=rocsolver,kbsyrk \
    PAPER_POTRF_LARGE_METHODS=kbsyrk \
    PAPER_POTRF_GEMM_M_TILE=0 \
    PAPER_RESULT_DIR="${RESULT_ROOT}/potrf_grid" \
    bash "${ROOT}/run_paper_experiments.sh" \
    > "${RESULT_ROOT}/potrf_driver.log" 2>&1

env \
    PAPER_SECTIONS=accuracy \
    PAPER_FACTORIZATIONS=getrf \
    PAPER_DEVICE=7 \
    PAPER_CONFIRM_IDLE=1 \
    PAPER_ACCURACY_NS=32768 \
    PAPER_ACCURACY_WARMUP=0 \
    PAPER_ACCURACY_ITERS=1 \
    PAPER_FULL_BACKWARD_ERROR_MAX_N=40960 \
    PAPER_GETRF_METHODS=rocsolver,mgemm \
    PAPER_GETRF_LARGE_METHODS=mgemm \
    PAPER_RESULT_DIR="${RESULT_ROOT}/getrf_N32768" \
    bash "${ROOT}/run_paper_experiments.sh" \
    > "${RESULT_ROOT}/getrf_driver.log" 2>&1

for n in 8192 16384 24576 32768 40960; do
    csv="${RESULT_ROOT}/potrf_grid/accuracy/potrf_N${n}_timings.csv"
    [[ -s "${csv}" ]]
    [[ "$(grep -c 'normwise_backward_error' "${csv}")" -eq 2 ]]
    grep -q '"rocSOLVER"' "${csv}"
    grep -q '"DoubleBlock+KBSYRK"' "${csv}"
    if grep -q 'MGEMM' "${csv}"; then
        echo "Unexpected MGEMM row in ${csv}" >&2
        exit 22
    fi
done

getrf_csv="${RESULT_ROOT}/getrf_N32768/accuracy/getrf_N32768_timings.csv"
[[ -s "${getrf_csv}" ]]
[[ "$(grep -c 'normwise_backward_error' "${getrf_csv}")" -eq 2 ]]
grep -q '"rocSOLVER GETRF"' "${getrf_csv}"
grep -q '"DoubleBlock-LU+BatchedPivot+MGEMM"' "${getrf_csv}"

{
    echo "End: $(date --iso-8601=seconds)"
    echo "Audit: PASS"
} >> "${RESULT_ROOT}/run_manifest.txt"

(
    cd "${RESULT_ROOT}"
    find . -type f ! -name checksums.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum > checksums.sha256
)

tar -C "$(dirname "${RESULT_ROOT}")" \
    -czf "${ARCHIVE}" "$(basename "${RESULT_ROOT}")"
sha256sum "${ARCHIVE}" > "${ARCHIVE}.sha256"
