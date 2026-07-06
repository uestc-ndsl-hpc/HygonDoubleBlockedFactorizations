#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"
setup_dtk_environment

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_syrk

RESULT_DIR=${ROOT_DIR}/results/fp32/syrk
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}
rm -f ${SUMMARY_LOG}

NS=(
    8192
    16384
    24576
    32768
    40960
    49152
    57344
    65536
)

USE_EQUAL_K=1
NBS="256 512 1024 2048"

ERROR_MODE=2
TEST_RECURSIVE=1
TEST_ROCBLAS=1
TEST_BATCHED=1
TEST_PAPER=1
TEST_MIXED=0

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "SYRK Pure FP32 Benchmark Start (Direct Interactive Run)" | tee -a ${SUMMARY_LOG}
echo "Error Mode         : ${ERROR_MODE}" | tee -a ${SUMMARY_LOG}
echo "Test rocBLAS (FP32): ${TEST_ROCBLAS}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP32)    : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Batch (FP32)  : ${TEST_BATCHED}" | tee -a ${SUMMARY_LOG}
echo "Test Batch (Mixed) : ${TEST_MIXED}" | tee -a ${SUMMARY_LOG}
echo "Test PaperSched    : ${TEST_PAPER}" | tee -a ${SUMMARY_LOG}
echo "NBs Sweep Sequence : ${NBS}" | tee -a ${SUMMARY_LOG}
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a ${SUMMARY_LOG}

check_benchmark_binary "${EXEC}" "${SUMMARY_LOG}" "rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"
do
    if [ ${USE_EQUAL_K} -eq 1 ]; then
        K=${N}
    else
        K=8192
    fi

    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running SYRK Benchmark: N=${N}, K=${K}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/syrk_N${N}_K${K}.log
    echo "Output Detail -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    ${LAUNCH_PREFIX} ${EXEC} \
        ${N} \
        ${K} \
        ${ERROR_MODE} \
        ${TEST_RECURSIVE} \
        ${TEST_ROCBLAS} \
        ${TEST_BATCHED} \
        ${TEST_MIXED} \
        ${TEST_PAPER} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "SYRK Pure FP32 Benchmark Finished"                   | tee -a ${SUMMARY_LOG}
echo "Final Summary written to: ${SUMMARY_LOG}"            | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
