#!/bin/bash
#SBATCH --job-name=trsm_benchmark
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

RESULT_DIR=${ROOT_DIR}/results/trsm
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

# 功能开关
ENABLE_ERROR=0
TEST_RECURSIVE=1
TEST_ROCBLAS=1
TEST_ITERATIVE=1

echo "==================================================" | tee ${SUMMARY_LOG}
echo "TRSM Benchmark Start" | tee -a ${SUMMARY_LOG}
echo "Enable Error Check : ${ENABLE_ERROR}" | tee -a ${SUMMARY_LOG}
echo "Test rocBLAS       : ${TEST_ROCBLAS}" | tee -a ${SUMMARY_LOG}
echo "Test Recursive     : ${TEST_RECURSIVE}" | tee -a ${SUMMARY_LOG}
echo "Test Iterative     : ${TEST_ITERATIVE}" | tee -a ${SUMMARY_LOG}
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

    ${EXEC} \
        ${N} \
        ${M} \
        ${ENABLE_ERROR} \
        ${TEST_RECURSIVE} \
        ${TEST_ROCBLAS} \
        ${TEST_ITERATIVE} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "TRSM Benchmark Finished" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}