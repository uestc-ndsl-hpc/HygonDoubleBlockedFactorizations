#!/usr/bin/env bash

set -euo pipefail

ROOT=/home/uestc_zss/HygonCholeskyDecomposition
OUT=${ROOT}/results/fp32/paper_cholesky_coupling_kbsyrk_nomtile_6098e38
STAGE=${ROOT}/results/server_staging/paper_cholesky_coupling_kbsyrk_nomtile_6098e38
EXPECTED_HEAD=6098e38e1f8f5e56beaeb1446df4bc207455cb12

if [[ -e "${OUT}" ]]; then
    echo "Refusing duplicate launch: ${OUT} already exists" >&2
    exit 10
fi

mkdir -p "${OUT}"
date -Is > "${OUT}/launcher.precheck.timestamp"
/usr/local/hyhal/bin/hy-smi > "${OUT}/launch.hy-smi.txt"
/opt/dtk/bin/rocm-smi --showpids > "${OUT}/launch.rocm-smi-pids.txt"
fuser /dev/kfd > "${OUT}/launch.kfd.txt" 2>/dev/null || true
git -C "${ROOT}" rev-parse HEAD > "${OUT}/launch.commit.txt"
git -C "${ROOT}" status --short > "${OUT}/launch.git-status.txt"
sha256sum "${STAGE}/cholesky_coupling_kbsyrk_nomtile.driver.sh" \
    "${STAGE}/cholesky_coupling_kbsyrk_nomtile.launch.sh" \
    > "${OUT}/staged_scripts.sha256"

if [[ -s "${OUT}/launch.kfd.txt" ]]; then
    date -Is > "${OUT}/launcher.blocked-kfd"
    exit 11
fi
if [[ -s "${OUT}/launch.git-status.txt" ]]; then
    date -Is > "${OUT}/launcher.blocked-git"
    exit 12
fi
if [[ "$(cat "${OUT}/launch.commit.txt")" != "${EXPECTED_HEAD}" ]]; then
    date -Is > "${OUT}/launcher.blocked-head"
    exit 13
fi

cp "${STAGE}/cholesky_coupling_kbsyrk_nomtile.driver.sh" "${OUT}/driver.sh"
cp "${STAGE}/cholesky_coupling_kbsyrk_nomtile.launch.sh" "${OUT}/launch.sh"
printf 'setsid -f bash %q > %q 2>&1 < /dev/null\n' \
    "${OUT}/driver.sh" "${OUT}/launcher.log" > "${OUT}/launch.command"

setsid -f bash "${OUT}/driver.sh" > "${OUT}/launcher.log" 2>&1 < /dev/null

for _ in $(seq 1 100); do
    if [[ -s "${OUT}/driver.pid" ]]; then
        cat "${OUT}/driver.pid"
        exit 0
    fi
    sleep 0.1
done

echo "Driver PID was not recorded" >&2
exit 14

