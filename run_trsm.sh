#!/bin/bash
#SBATCH --job-name=trsm_fp16
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=dcu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# =========================================================
# ROCm + OpenMP Environment
# =========================================================

export ROCM_PATH=/public/software/compiler/rocm/dtk-23.10

module load compiler/openmp/50

export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:$LD_LIBRARY_PATH
export PATH=$ROCM_PATH/bin:$PATH

# =========================================================
# Project Paths
# =========================================================

ROOT_DIR=$(pwd)

BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_trsm

# 修改结果目录为 fp16
RESULT_DIR=${ROOT_DIR}/results/trsm_fp16_fp32
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}

# =========================================================
# Benchmark Config
# =========================================================

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

# M=N
USE_EQUAL_M=1

# 多个 NB
NBS="256 512 1024 2048"

# 功能开关（调整为：仅进行半精度 + 误差测试）
ERROR_MODE=2
TEST_RECURSIVE=1
TEST_ROCBLAS=0
TEST_ITERATIVE=0
TEST_HALF=1

echo "==================================================" | tee ${SUMMARY_LOG}
echo "TRSM FP16 Benchmark Start" | tee -a ${SUMMARY_LOG}
echo "Error Mode         : ${ERROR_MODE} (3=Forward+Backward)" | tee -a ${SUMMARY_LOG}
echo "Test rocBLAS (FP32): ${TEST_ROCBLAS}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP32)    : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Iter (FP32)   : ${TEST_ITERATIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Rec (FP16)    : ${TEST_HALF}" | tee -a ${SUMMARY_LOG}
echo "NBs                : ${NBS}" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}

# =========================================================
# Dependency Check
# =========================================================

echo "" | tee -a ${SUMMARY_LOG}
echo "Checking binary dependencies..." | tee -a ${SUMMARY_LOG}

ldd ${EXEC} | grep "not found" | tee -a ${SUMMARY_LOG}

# =========================================================
# Run Benchmark
# =========================================================

for N in "${NS[@]}"
do
    if [ ${USE_EQUAL_M} -eq 1 ]; then
        M=${N}
    else
        M=8192
    fi

    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running Benchmark: N=${N}, M=${M}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/trsm_N${N}_M${M}.log

    echo "Output -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    # 按照新 main 函数的 argv 映射顺序传参：
    # 1:N, 2:M, 3:ERROR_MODE, 4:TEST_RECURSIVE, 5:TEST_ROCBLAS, 6:TEST_ITERATIVE, 7:TEST_HALF, 8+:NBS
    ${EXEC} \
        ${N} \
        ${M} \
        ${ERROR_MODE} \
        ${TEST_RECURSIVE} \
        ${TEST_ROCBLAS} \
        ${TEST_ITERATIVE} \
        ${TEST_HALF} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "TRSM FP16 Benchmark Finished" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}