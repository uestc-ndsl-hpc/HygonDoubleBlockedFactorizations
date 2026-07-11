#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-7}
setup_dtk_environment
module load compiler/openmp/50 2>/dev/null

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_potrf
RESULT_DIR=${ROOT_DIR}/results/fp32/potrf_kblocked_syrk_sweep
SUMMARY_LOG=${RESULT_DIR}/summary.log

if [ -n "${POTRF_NS:-}" ]; then
    read -r -a NS <<< "${POTRF_NS}"
else
    NS=(16384 24576 32768 40960 49152 57344 65536)
fi

BLOCKED_NB=${BLOCKED_NB:-2048}
DOUBLE_NB=${DOUBLE_NB:-8192}
MICRO_BLOCK=${MICRO_BLOCK:-512}
ERROR_MODE=2
PRECISION=0

mkdir -p "${RESULT_DIR}"
rm -f "${SUMMARY_LOG}"

echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "POTRF K-blocked SYRK size sweep"                  | tee -a "${SUMMARY_LOG}"
echo "Executable         : ${EXEC}"                    | tee -a "${SUMMARY_LOG}"
echo "Result directory   : ${RESULT_DIR}"              | tee -a "${SUMMARY_LOG}"
echo "HIP device         : ${HIP_VISIBLE_DEVICES}"      | tee -a "${SUMMARY_LOG}"
echo "Matrix sizes       : ${NS[*]}"                    | tee -a "${SUMMARY_LOG}"
echo "Blocked-RL NB      : ${BLOCKED_NB}"               | tee -a "${SUMMARY_LOG}"
echo "DoubleBlock NB/B   : ${DOUBLE_NB}/${MICRO_BLOCK}" | tee -a "${SUMMARY_LOG}"
echo "KBSYRK K tile      : ${DOUBLE_NB}"               | tee -a "${SUMMARY_LOG}"
echo "Adaptive M-GEMM    : tile 8192 for M>32768 and K>=5632" | tee -a "${SUMMARY_LOG}"
echo "Deferred TRSM      : outer panel, micro block 1024" | tee -a "${SUMMARY_LOG}"
echo "Lookahead          : diagonal factor / outer GEMM on two streams" | tee -a "${SUMMARY_LOG}"
echo "Error verify mode  : backward"                   | tee -a "${SUMMARY_LOG}"
echo "NUMA binding       : disabled"                   | tee -a "${SUMMARY_LOG}"
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"

check_benchmark_binary \
    "${EXEC}" \
    "${SUMMARY_LOG}" \
    "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"

LAUNCH_PREFIX=""

for N in "${NS[@]}"
do
    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> N=${N}: rocSOLVER and Blocked-RL" | tee -a "${SUMMARY_LOG}"

    BASELINE_LOG=${RESULT_DIR}/potrf_baseline_N${N}.log
    ${LAUNCH_PREFIX} "${EXEC}" \
        "${N}" \
        1 \
        0 \
        "${ERROR_MODE}" \
        "${PRECISION}" \
        1 \
        0 \
        "${BLOCKED_NB}" \
        2>&1 | tee "${BASELINE_LOG}"
    cat "${BASELINE_LOG}" >> "${SUMMARY_LOG}"

    echo "" | tee -a "${SUMMARY_LOG}"
    echo ">>> N=${N}: DoubleBlock variants" | tee -a "${SUMMARY_LOG}"

    DOUBLE_LOG=${RESULT_DIR}/potrf_double_block_N${N}.log
    ${LAUNCH_PREFIX} "${EXEC}" \
        "${N}" \
        0 \
        0 \
        "${ERROR_MODE}" \
        "${PRECISION}" \
        0 \
        1 \
        "${DOUBLE_NB}" \
        --b "${MICRO_BLOCK}" \
        2>&1 | tee "${DOUBLE_LOG}"
    cat "${DOUBLE_LOG}" >> "${SUMMARY_LOG}"
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
echo "POTRF K-blocked SYRK sweep finished"              | tee -a "${SUMMARY_LOG}"
echo "Summary: ${SUMMARY_LOG}"                         | tee -a "${SUMMARY_LOG}"
echo "==================================================" | tee -a "${SUMMARY_LOG}"
