#!/usr/bin/env bash

set -euo pipefail

REPO=/home/uestc_zss/HygonCholeskyDecomposition
ROOT=${REPO}/results/fp32/paper_final/block_collapse_ce7690b_20260718T050406Z
PERF=${ROOT}/performance_retry1

if [[ -e "${PERF}" ]]; then
    echo "Refusing duplicate launch: ${PERF} already exists" >&2
    exit 17
fi

mkdir -p "${PERF}"
date -Is > "${PERF}/precheck_time.txt"
/usr/local/hyhal/bin/hy-smi > "${PERF}/device_precheck.txt"
/opt/dtk/bin/rocm-smi --showpids > "${PERF}/kfd_precheck.txt"

if ! grep -qF "No KFD PIDs currently running!" "${PERF}/kfd_precheck.txt"; then
    echo "KFD activity detected; launch cancelled" \
        | tee "${PERF}/launch_cancelled.txt" >&2
    exit 18
fi

if ! awk '
    /^[0-7][[:space:]]/ {
        devices++
        if ($6 != "0%" || $7 != "0.0%") busy=1
    }
    END { exit !(devices == 8 && !busy) }
' "${PERF}/device_precheck.txt"; then
    echo "Non-idle HCU state detected; launch cancelled" \
        | tee "${PERF}/launch_cancelled.txt" >&2
    exit 19
fi

git -C "${REPO}" status --short > "${PERF}/git_status.txt"
if [[ -s "${PERF}/git_status.txt" ]]; then
    echo "Dirty worktree; launch cancelled" \
        | tee "${PERF}/launch_cancelled.txt" >&2
    exit 20
fi

git -C "${REPO}" rev-parse HEAD > "${PERF}/git_head.txt"
git -C "${REPO}" diff --check > "${PERF}/git_diff_check.txt"
git -C "${REPO}" check-ignore -v "${PERF#${REPO}/}" \
    > "${PERF}/git_ignore_evidence.txt"
sha256sum \
    "${REPO}/build/benchmark_potrf" \
    "${REPO}/build/benchmark_getrf" \
    > "${PERF}/binary_sha256.txt"

cat > "${PERF}/run_driver.sh" <<EOF
#!/usr/bin/env bash

set +e

env \\
    PAPER_SECTIONS=block-collapse \\
    PAPER_DEVICE=1 \\
    PAPER_CONFIRM_IDLE=1 \\
    PAPER_RESULT_DIR=${PERF} \\
    PAPER_BLOCK_COLLAPSE_NS="8192 16384 24576 32768 40960" \\
    PAPER_BLOCK_COLLAPSE_WARMUP=2 \\
    PAPER_BLOCK_COLLAPSE_ITERS=10 \\
    bash ${REPO}/run_paper_experiments.sh
rc=\$?
printf '%s\n' "\${rc}" > "${PERF}/driver.exit"
if [[ \${rc} -eq 0 ]]; then
    touch "${PERF}/completed"
else
    touch "${PERF}/failed"
fi
exit "\${rc}"
EOF

chmod 0755 "${PERF}/run_driver.sh"
cp "$0" "${PERF}/launch_wrapper.sh"
printf '%s\n' \
    "PAPER_SECTIONS=block-collapse" \
    "PAPER_DEVICE=1" \
    "PAPER_CONFIRM_IDLE=1" \
    "PAPER_RESULT_DIR=${PERF}" \
    "PAPER_BLOCK_COLLAPSE_NS=8192 16384 24576 32768 40960" \
    "PAPER_BLOCK_COLLAPSE_WARMUP=2" \
    "PAPER_BLOCK_COLLAPSE_ITERS=10" \
    > "${PERF}/launch.parameters"

nohup "${PERF}/run_driver.sh" \
    > "${PERF}/driver.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" | tee "${PERF}/driver.pid"
