#!/bin/bash

# =========================================================
# DTK 运行环境配置 (适配 dtk-26.04 环境)
# =========================================================
export ROCM_PATH=${ROCM_PATH:-/opt/dtk}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

# 如果运行提示找不到 openmp 模块，可以直接注释掉这一行
module load compiler/openmp/50 2>/dev/null

export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}
export PATH=$ROCM_PATH/bin:$PATH

# 关闭底层复杂的 Trace 追踪与内核调试打印，确保释放全部硬件计算吞吐
export ROCBLAS_LAYER=0
export TENSILE_DB=0

# =========================================================
# 项目路径与输出管理
# =========================================================
ROOT_DIR=$(pwd)
BUILD_DIR=${ROOT_DIR}/build
EXEC=${BUILD_DIR}/benchmark_potrf
RESULT_DIR=${ROOT_DIR}/results/fp32/potrf
SUMMARY_LOG=${RESULT_DIR}/summary.log

mkdir -p ${RESULT_DIR}
rm -f ${SUMMARY_LOG}

# =========================================================
# 测试参数矩阵配置
# =========================================================
NS=(
    8192
    16384
    24576
    32768
    40960
)
NBS="256 512 1024 2048"

ERROR_MODE=2       # 2 = Backward 采样后向误差
TEST_ROCSOLVER=1   # 1 = 测试原生系统库作为 Baseline
TEST_RECURSIVE=1   # 1 = 测试自定义递归分块算子
PRECISION=0        # 0 = 纯 FP32 精度测试

# =========================================================
# 基础日志输出 (打印测试元数据)
# =========================================================
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF High-Performance Benchmark (Direct Run)"       | tee -a ${SUMMARY_LOG}
echo "DTK Environment    : ${ROCM_PATH}"                  | tee -a ${SUMMARY_LOG}
echo "Precision Mode     : ${PRECISION} (0=FP32)"         | tee -a ${SUMMARY_LOG}
echo "Error Verify Mode  : ${ERROR_MODE} (2=Backward)"    | tee -a ${SUMMARY_LOG}
echo "Test rocSOLVER     : ${TEST_ROCSOLVER}"             | tee -a ${SUMMARY_LOG}
echo "Test Recursive     : ${TEST_RECURSIVE}"             | tee -a ${SUMMARY_LOG}
echo "NBs Sweep Sequence : ${NBS}"                        | tee -a ${SUMMARY_LOG}
echo "hipcc path         : $(command -v hipcc 2>/dev/null || echo unavailable)" | tee -a ${SUMMARY_LOG}
echo "hipcc version      : $(hipcc --version 2>/dev/null | head -n 1 || echo unavailable)" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}

# =========================================================
# 依赖完整性校验
# =========================================================
echo "" | tee -a ${SUMMARY_LOG}
echo "Checking binary dependencies..." | tee -a ${SUMMARY_LOG}

if [ -f "${EXEC}" ]; then
    ldd ${EXEC} | grep "not found" | tee -a ${SUMMARY_LOG}
    echo "Resolved ROCm libraries:" | tee -a ${SUMMARY_LOG}
    ldd ${EXEC} | grep -E "rocsolver|rocblas|hipblas|amdhip64|hsa-runtime64" | tee -a ${SUMMARY_LOG}
else
    echo "Error: Executable ${EXEC} not found! Please build the target first." | tee -a ${SUMMARY_LOG}
    exit 1
fi

# =========================================================
# NUMA 硬件架构亲和性绑定
# =========================================================
NUMA_NODE=$(cat /sys/class/drm/card0/device/numa_node 2>/dev/null)
if [ "$NUMA_NODE" != "-1" ] && [ ! -z "$NUMA_NODE" ]; then
    LAUNCH_PREFIX="numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    echo "NUMA Binding detected and enabled: ${LAUNCH_PREFIX}" | tee -a ${SUMMARY_LOG}
else
    LAUNCH_PREFIX=""
    echo "Warning: No direct NUMA affinity mapping found. Running without numactl constraints." | tee -a ${SUMMARY_LOG}
fi

# =========================================================
# 循环迭代测试核心流水线
# =========================================================
for N in "${NS[@]}"
do
    echo "" | tee -a ${SUMMARY_LOG}
    echo ">>> Running POTRF Benchmark: N=${N}" | tee -a ${SUMMARY_LOG}

    OUT_FILE=${RESULT_DIR}/potrf_N${N}.log
    echo "Output Detail Target -> ${OUT_FILE}" | tee -a ${SUMMARY_LOG}

    # 直接在前台执行
    ${LAUNCH_PREFIX} ${EXEC} \
        ${N} \
        ${TEST_ROCSOLVER} \
        ${TEST_RECURSIVE} \
        ${ERROR_MODE} \
        ${PRECISION} \
        ${NBS} \
        2>&1 | tee ${OUT_FILE}

    # 将详细输出完整导入到汇总总日志中
    cat ${OUT_FILE} >> ${SUMMARY_LOG}
done

echo "" | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
echo "POTRF High-Performance Benchmark Finished"           | tee -a ${SUMMARY_LOG}
echo "Final Master Summary written to: ${SUMMARY_LOG}"      | tee -a ${SUMMARY_LOG}
echo "==================================================" | tee -a ${SUMMARY_LOG}
