#!/bin/bash
#SBATCH --job-name=potrf_backward_fp64
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
EXEC=${BUILD_DIR}/benchmark_potrf

RESULT_DIR=${ROOT_DIR}/results/potrf_backward_fp64
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}

# =========================================================
# Benchmark Config
# =========================================================

NS=($(seq 8192 8192 40960))

NBS="256 512 1024 2048"

ENABLE_ERROR=1
TEST_ROCSOLVER=1
TEST_RECURSIVE=1

# 误差类型:
# 0 -> forward
# 1 -> backward
# 2 -> both
ERROR_TYPE=1

# 精度:
# 0 -> float
# 1 -> double
PRECISION=1

echo "==================================================" | tee ${SUMMARY_LOG}
echo "POTRF Backward Error Benchmark (FP64) Start" | tee -a ${SUMMARY_LOG}
echo "Enable Error Check : ${ENABLE_ERROR}" | tee -a ${SUMMARY_LOG}
echo "Error Type         : ${ERROR_TYPE} (Backward)" | tee -a ${SUMMARY_LOG}
echo "Precision          : ${PRECISION} (Double)" | tee -a ${SUMMARY_LOG}
echo "Test rocSOLVER     : ${TEST_ROCSOLVER}" | tee -a ${SUMMARY_LOG}
echo "Test Recursive     : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
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
    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running Benchmark: N=${N}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/potrf_N${N}.log

    echo "Output -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    ${EXEC} \
        ${N} \
        ${ENABLE_ERROR} \
        ${TEST_ROCSOLVER} \
        ${TEST_RECURSIVE} \
        ${ERROR_TYPE} \
        ${PRECISION} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF Benchmark Finished" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}