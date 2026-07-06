#!/bin/bash

# =========================================================
# ROCm Environment Setup (适配 dtk-26.04 环境)
# =========================================================

export ROCM_PATH=${ROCM_PATH:-/opt/dtk}

# 显式指定使用的 DCU 设备，工作节点虽有 8 张，测试单精度通常绑定 1 张进行精确压测
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=$ROCM_PATH/bin:$PATH

# 关闭所有底层的 Debug 追踪与内核名称打印，释放极致运行性能
export ROCBLAS_LAYER=0
export TENSILE_DB=0

# =========================================================
# Project Paths
# =========================================================

ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_gemm
RESULT_DIR=${ROOT_DIR}/results/fp32/gemm
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}

# 每次重新运行前，清空旧的汇总日志，确保数据干净
rm -f ${SUMMARY_LOG}

# =========================================================
# Test Configuration (与 C++ 源码默认序列保持对齐)
# =========================================================

# 按照论文 5.1.1 的设计意图，构建消元步数（K）序列
# 支持密集测试低阶部分，以精细观察高瘦面板的硬件利用率爬升趋势
K_STEPS=(64 128 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384)

echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "GEMM Shape Benchmark Start (Interactive on Node)"   | tee -a ${SUMMARY_LOG}
echo "Executable         : ${EXEC}"                        | tee -a ${SUMMARY_LOG}
echo "DTK Environment    : ${ROCM_PATH}"                  | tee -a ${SUMMARY_LOG}
echo "hipcc path         : $(command -v hipcc 2>/dev/null || echo unavailable)" | tee -a ${SUMMARY_LOG}
echo "hipcc version      : $(hipcc --version 2>/dev/null | head -n 1 || echo unavailable)" | tee -a ${SUMMARY_LOG}
echo "Target Device ID   : HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}" | tee -a ${SUMMARY_LOG}
echo "Total Matrix Size  : 32768"                          | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}

# =========================================================
# Dependency Check
# =========================================================

echo "" | tee -a ${SUMMARY_LOG}
echo "Checking binary dependencies..." | tee -a ${SUMMARY_LOG}

if [ -f "${EXEC}" ]; then
    ldd ${EXEC} | grep "not found" | tee -a ${SUMMARY_LOG}
    echo "Resolved ROCm libraries:" | tee -a ${SUMMARY_LOG}
    ldd ${EXEC} | grep -E "rocblas|hipblas|amdhip64|hsa-runtime64" | tee -a ${SUMMARY_LOG}
else
    echo "Error: Executable ${EXEC} not found! Please build first." | tee -a ${SUMMARY_LOG}
    exit 1
fi

# =========================================================
# Run Benchmark Loops (依次传入不同的 K 尺寸进行细粒度压测)
# =========================================================

echo "" | tee -a ${SUMMARY_LOG}
echo "Starting Right-Looking GEMM Shapes sweeping..." | tee -a ${SUMMARY_LOG}

# 为了打印统一漂亮的表格头，先不传参数空跑一次，让其打印完整的 C++ 表头
# （或者通过循环直接逐个传入参数，这里通过逐个传入 K 值触发精细测试）
# 注意：C++ 主程序内部带有完整的循环，如果你想一气呵成跑完所有 K_STEPS，
# 直接无需带参数执行一次 ${EXEC} 即可完美输出。
# 下面这种设计支持你后续对特定异常点（如 K=8192）进行单独重测。

${EXEC} 2>&1 | tee -a ${SUMMARY_LOG}

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "GEMM Shape Benchmark Finished"                       | tee -a ${SUMMARY_LOG}
echo "Summary written to: ${SUMMARY_LOG}"                  | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
