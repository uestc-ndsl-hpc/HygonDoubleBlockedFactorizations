#!/bin/bash

export ROCM_PATH=/public/software/compiler/rocm/dtk-24.04
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$PATH

export ROCBLAS_LAYER=0
export TENSILE_DB=0

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
TEST_MIXED=0

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "SYRK Pure FP32 Benchmark Start (Direct Interactive Run)" | tee -a ${SUMMARY_LOG}
echo "Error Mode         : ${ERROR_MODE}" | tee -a ${SUMMARY_LOG}
echo "Test rocBLAS (FP32): ${TEST_ROCBLAS}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP32)    : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Batch (FP32)  : ${TEST_BATCHED}" | tee -a ${SUMMARY_LOG}
echo "Test Batch (Mixed) : ${TEST_MIXED}" | tee -a ${SUMMARY_LOG}
echo "NBs Sweep Sequence : ${NBS}" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}

echo "" | tee -a ${SUMMARY_LOG}
echo "Checking binary dependencies..." | tee -a ${SUMMARY_LOG}

if [ -f "${EXEC}" ]; then
    ldd ${EXEC} | grep "not found" | tee -a ${SUMMARY_LOG}
else
    echo "Error: Executable ${EXEC} not found! Please build first." | tee -a ${SUMMARY_LOG}
    exit 1
fi

NUMA_NODE=$(cat /sys/class/drm/card0/device/numa_node 2>/dev/null)
if [ "$NUMA_NODE" != "-1" ] && [ ! -z "$NUMA_NODE" ]; then
    LAUNCH_PREFIX="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    echo "NUMA Binding detected and enabled: ${LAUNCH_PREFIX}" | tee -a ${SUMMARY_LOG}
else
    LAUNCH_PREFIX=""
    echo "Warning: No direct NUMA affinity mapping found. Running without numactl." | tee -a ${SUMMARY_LOG}
fi

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
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "SYRK Pure FP32 Benchmark Finished"                   | tee -a ${SUMMARY_LOG}
echo "Final Summary written to: ${SUMMARY_LOG}"            | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}