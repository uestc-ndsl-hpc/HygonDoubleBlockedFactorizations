#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/scripts/benchmark_env.sh"

setup_dtk_environment
if command -v module >/dev/null 2>&1; then
    module load compiler/openmp/50 2>/dev/null || true
fi

ROOT_DIR=${SCRIPT_DIR}
BENCHMARK=${ROOT_DIR}/build/benchmark_potrf
PROFILER=${ROOT_DIR}/build/profile_tiled_right_looking_potrf
RESULT_DIR=${POTRF_TILED_RL_RESULT_DIR:-${ROOT_DIR}/results/fp32/potrf_tiled_rl}
SUMMARY_LOG=${RESULT_DIR}/summary.log

N=${POTRF_N:-40960}
read -r -a OUTER_BLOCKS <<< "${POTRF_OUTER_BLOCKS:-8192}"
read -r -a UPDATE_TILES <<< "${POTRF_UPDATE_TILES:-2048 4096}"
read -r -a POTRF_LEAVES <<< "${POTRF_LEAVES:-512 1024}"

PROFILE_KB=${POTRF_PROFILE_KB:-8192}
PROFILE_TILE=${POTRF_PROFILE_TILE:-4096}
PROFILE_LEAF=${POTRF_PROFILE_LEAF:-1024}
TRSM_LEAF=${POTRF_TRSM_LEAF:-1024}
K_TILE=${POTRF_K_TILE:-8192}
ERROR_MODE=${POTRF_ERROR_MODE:-2}
PRECISION=${POTRF_PRECISION:-0}
STAGE_WARMUP=${POTRF_STAGE_WARMUP:-1}
STAGE_ITERS=${POTRF_STAGE_ITERS:-3}

mkdir -p "${RESULT_DIR}"

{
    echo "=================================================="
    echo "Tiled-RL DoubleBlock POTRF experiment"
    echo "Benchmark          : ${BENCHMARK}"
    echo "Profiler           : ${PROFILER}"
    echo "Result directory   : ${RESULT_DIR}"
    echo "Matrix size        : ${N}"
    echo "Outer blocks       : ${OUTER_BLOCKS[*]}"
    echo "Update tiles       : ${UPDATE_TILES[*]}"
    echo "POTRF leaves       : ${POTRF_LEAVES[*]}"
    echo "Profile KB/tile    : ${PROFILE_KB}/${PROFILE_TILE}"
    echo "Profile leaves     : ${PROFILE_LEAF}/${TRSM_LEAF}"
    echo "Profile K tile     : ${K_TILE}"
    echo "Error verify mode  : ${ERROR_MODE}"
    echo "Stage warmup/iters : ${STAGE_WARMUP}/${STAGE_ITERS}"
} | tee "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${BENCHMARK}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
check_benchmark_binary \
    "${PROFILER}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

BENCHMARK_LOG=${RESULT_DIR}/benchmark_N${N}.log
echo "" | tee -a "${SUMMARY_LOG}"
echo ">>> Running benchmark sweep" | tee -a "${SUMMARY_LOG}"

"${LAUNCH_PREFIX[@]}" "${BENCHMARK}" \
    "${N}" \
    0 \
    0 \
    "${ERROR_MODE}" \
    "${PRECISION}" \
    0 \
    1 \
    256 \
    --double-nb "${OUTER_BLOCKS[@]}" \
    --b "${POTRF_LEAVES[@]}" \
    --rl-tile "${UPDATE_TILES[@]}" \
    2>&1 | tee "${BENCHMARK_LOG}"

cat "${BENCHMARK_LOG}" >> "${SUMMARY_LOG}"

PROFILE_LOG=${RESULT_DIR}/profile_N${N}_KB${PROFILE_KB}_T${PROFILE_TILE}_IB${PROFILE_LEAF}.log
echo "" | tee -a "${SUMMARY_LOG}"
echo ">>> Running stage profile" | tee -a "${SUMMARY_LOG}"

"${LAUNCH_PREFIX[@]}" "${PROFILER}" \
    --n "${N}" \
    --kb "${PROFILE_KB}" \
    --tile "${PROFILE_TILE}" \
    --leaf "${PROFILE_LEAF}" \
    --trsm-leaf "${TRSM_LEAF}" \
    --k-tile "${K_TILE}" \
    --warmup "${STAGE_WARMUP}" \
    --iters "${STAGE_ITERS}" \
    2>&1 | tee "${PROFILE_LOG}"

cat "${PROFILE_LOG}" >> "${SUMMARY_LOG}"

echo "" | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"
