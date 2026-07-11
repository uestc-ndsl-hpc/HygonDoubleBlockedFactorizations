#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"

setup_dtk_environment
module load compiler/openmp/50 2>/dev/null

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_potrf
RESULT_DIR=${ROOT_DIR}/results/fp32/potrf_double_block_tune
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}
rm -f ${SUMMARY_LOG}

NS=(
    8192
    16384
    24576
    32768
    40960
)

NBS="4096 8192 12288"
MICRO_BLOCKS="512 1024 2048"

ERROR_MODE=2
PRECISION=0

TEST_ROCSOLVER=0
TEST_RECURSIVE=0
TEST_BLOCKED=0
TEST_DOUBLE_BLOCKED=1

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF DoubleBlock nb/b tuning"                    | tee -a ${SUMMARY_LOG}
echo "Executable         : ${EXEC}"                     | tee -a ${SUMMARY_LOG}
echo "Result directory   : ${RESULT_DIR}"               | tee -a ${SUMMARY_LOG}
echo "Precision Mode     : ${PRECISION} (0=FP32)"       | tee -a ${SUMMARY_LOG}
echo "Error Verify Mode  : ${ERROR_MODE} (2=Backward)"  | tee -a ${SUMMARY_LOG}
echo "NB Sweep           : ${NBS}"                      | tee -a ${SUMMARY_LOG}
echo "b Sweep            : ${MICRO_BLOCKS}"             | tee -a ${SUMMARY_LOG}
echo "Methods            : DoubleBlock, DoubleBlock+DBTRSM, DoubleBlock+KBSYRK, DoubleBlock+KBSYRK+MGEMM" | tee -a ${SUMMARY_LOG}
echo "DBTRSM trsm_b      : 1024"                        | tee -a ${SUMMARY_LOG}
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a ${SUMMARY_LOG}

check_benchmark_binary "${EXEC}" "${SUMMARY_LOG}" "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"
do
    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running DoubleBlock tuning: N=${N}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/potrf_double_block_tune_N${N}.log
    echo "Output Detail Target -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    ${LAUNCH_PREFIX} ${EXEC} \
        ${N} \
        ${TEST_ROCSOLVER} \
        ${TEST_RECURSIVE} \
        ${ERROR_MODE} \
        ${PRECISION} \
        ${TEST_BLOCKED} \
        ${TEST_DOUBLE_BLOCKED} \
        ${NBS} \
        --b ${MICRO_BLOCKS} \
        2>&1 | tee ${OUT_FILE}

    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF DoubleBlock nb/b tuning finished"           | tee -a ${SUMMARY_LOG}
echo "Final summary written to: ${SUMMARY_LOG}"          | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
