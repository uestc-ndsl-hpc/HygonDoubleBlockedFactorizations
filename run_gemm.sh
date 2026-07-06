#!/bin/bash

source "$(dirname "$0")/scripts/benchmark_env.sh"

# DTK runtime setup. The default ROCM_PATH is /opt/dtk, but callers can
# override it, for example: ROCM_PATH=/opt/dtk-26.04 ./run_gemm.sh
setup_dtk_environment

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_gemm
RESULT_DIR=${ROOT_DIR}/results/fp32/gemm
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}
rm -f ${SUMMARY_LOG}

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "GEMM shape benchmark"                              | tee -a ${SUMMARY_LOG}
echo "Executable         : ${EXEC}"                       | tee -a ${SUMMARY_LOG}
echo "Result directory   : ${RESULT_DIR}"                 | tee -a ${SUMMARY_LOG}
log_toolchain "${SUMMARY_LOG}"
echo "Target Device ID   : HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" | tee -a ${SUMMARY_LOG}
echo "Total Matrix Size  : 32768"                         | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}

check_benchmark_binary "${EXEC}" "${SUMMARY_LOG}" "rocblas|hipblas|amdhip64|hsa-runtime64"

echo "" | tee -a ${SUMMARY_LOG}
echo "Starting right-looking GEMM shape sweep..." | tee -a ${SUMMARY_LOG}

${EXEC} 2>&1 | tee -a ${SUMMARY_LOG}

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "GEMM shape benchmark finished"                      | tee -a ${SUMMARY_LOG}
echo "Summary written to: ${SUMMARY_LOG}"                  | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
