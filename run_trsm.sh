#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"
setup_dtk_environment

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_trsm
RESULT_DIR=${ROOT_DIR}/results/fp32/trsm
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

USE_EQUAL_M=1
NBS="256 512 1024 2048"

ERROR_MODE=2
TEST_RECURSIVE=1
TEST_ROCBLAS=1
TEST_ITERATIVE=1
TEST_HALF=0
TEST_STREAMED=1

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "TRSM Pure FP32 Benchmark Start (Direct Parameter Run)" | tee -a ${SUMMARY_LOG}
echo "Error Mode         : ${ERROR_MODE}" | tee -a ${SUMMARY_LOG}
echo "Test rocBLAS (FP32): ${TEST_ROCBLAS}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP32)    : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Streamed RL   : ${TEST_STREAMED}" | tee -a ${SUMMARY_LOG}
echo "Test Iter (FP32)   : ${TEST_ITERATIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP16)    : ${TEST_HALF}" | tee -a ${SUMMARY_LOG}
echo "NBs Sweep Sequence : ${NBS}" | tee -a ${SUMMARY_LOG}
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a ${SUMMARY_LOG}

check_benchmark_binary "${EXEC}" "${SUMMARY_LOG}" "rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"
do
    if [ ${USE_EQUAL_M} -eq 1 ]; then
        M=${N}
    else
        M=8192
    fi

    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running TRSM Benchmark: N=${N}, M=${M}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/trsm_N${N}_M${M}.log
    echo "Output Detail -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    ${LAUNCH_PREFIX} ${EXEC} \
        ${N} \
        ${M} \
        ${ERROR_MODE} \
        ${TEST_RECURSIVE} \
        ${TEST_ROCBLAS} \
        ${TEST_ITERATIVE} \
        ${TEST_HALF} \
        ${TEST_STREAMED} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "TRSM Pure FP32 Benchmark Finished"                   | tee -a ${SUMMARY_LOG}
echo "Final Summary written to: ${SUMMARY_LOG}"            | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
