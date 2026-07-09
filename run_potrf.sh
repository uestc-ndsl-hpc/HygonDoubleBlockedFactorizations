#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"

# DTK runtime setup. The default ROCM_PATH is /opt/dtk, but callers can
# override it, for example: ROCM_PATH=/opt/dtk-26.04 ./run_potrf.sh
setup_dtk_environment

# Some server shells expose OpenMP through environment modules. Keep this
# optional so direct interactive runs still work when module is unavailable.
module load compiler/openmp/50 2>/dev/null

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_potrf
RESULT_DIR=${ROOT_DIR}/results/fp32/potrf
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
NBS="256 512 1024 2048"

ERROR_MODE=2       # 2 = sampled backward error
TEST_ROCSOLVER=1   # 1 = use rocSOLVER as baseline
TEST_BLOCKED=1     # 1 = test blocked right-looking POTRF
TEST_DOUBLE_BLOCKED=1 # 1 = test double-block POTRF
TEST_RECURSIVE=1   # 1 = test recursive POTRF
PRECISION=0        # 0 = FP32

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF benchmark"                                    | tee -a ${SUMMARY_LOG}
echo "Executable         : ${EXEC}"                       | tee -a ${SUMMARY_LOG}
echo "Result directory   : ${RESULT_DIR}"                 | tee -a ${SUMMARY_LOG}
echo "Precision Mode     : ${PRECISION} (0=FP32)"         | tee -a ${SUMMARY_LOG}
echo "Error Verify Mode  : ${ERROR_MODE} (2=Backward)"    | tee -a ${SUMMARY_LOG}
echo "Test rocSOLVER     : ${TEST_ROCSOLVER}"             | tee -a ${SUMMARY_LOG}
echo "Test Blocked RL    : ${TEST_BLOCKED}"               | tee -a ${SUMMARY_LOG}
echo "Test DoubleBlock   : ${TEST_DOUBLE_BLOCKED}"        | tee -a ${SUMMARY_LOG}
echo "Test Recursive     : ${TEST_RECURSIVE}"             | tee -a ${SUMMARY_LOG}
echo "NBs Sweep Sequence : ${NBS}"                        | tee -a ${SUMMARY_LOG}
echo "DoubleBlock params : nb=N/2, b=N/4 (OSLA default)"  | tee -a ${SUMMARY_LOG}
log_toolchain "${SUMMARY_LOG}"
echo "==================================================" | tee -a ${SUMMARY_LOG}

check_benchmark_binary "${EXEC}" "${SUMMARY_LOG}" "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64"
configure_numa_launch_prefix "${SUMMARY_LOG}"

for N in "${NS[@]}"
do
    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running POTRF Benchmark: N=${N}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/potrf_N${N}.log
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
        2>&1 | tee ${OUT_FILE}

    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF benchmark finished"                           | tee -a ${SUMMARY_LOG}
echo "Final summary written to: ${SUMMARY_LOG}"            | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
