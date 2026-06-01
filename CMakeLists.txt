cmake_minimum_required(VERSION 3.21)
# 1. 声明 CXX 和 HIP 语言支持
project(TRSM_Benchmark LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定中科曙光/海光 DTK 环境路径
set(ROCM_PATH /public/software/compiler/rocm/dtk-24.04)

# =========================================================
# Include & Link Directories
# =========================================================

include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

link_directories(
    ${ROCM_PATH}/lib
)

# =========================================================
# Common Compile Settings
# =========================================================

add_compile_definitions(
    __HIP_PLATFORM_AMD__
)

# =========================================================
# Define Kernel Sources
# =========================================================

set(KERNEL_SOURCES
    src/kernels/add_damping.hip
    src/kernels/fp16_convert.hip
    src/kernels/matrix_scale.hip
)

# 显式指示 CMake 将这些文件作为 HIP 代码编译
set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)

# =========================================================
# Define Common Utility Sources
# =========================================================

set(COMMON_UTIL_SOURCES
    src/utils/memory/stream_pool.cpp             # 移动后的新流管理器路径
    src/utils/benchmark/benchmark_printer.cpp   # 拆分后的结果打印实现
    src/utils/matrix/matrix_generator.cpp       # 拆分并显式实例化后的矩阵生成实现
)

# =========================================================
# Define GEMM Sources (补齐混合精度 GEMM 源文件)
# =========================================================

set(GEMM_SOURCES
    src/gemm/mixed_gemm.cpp                      # 拆分后的混合精度 GEMM 实现
)

# =========================================================
# TRSM Sources
# =========================================================

set(TRSM_SOURCES
    src/trsm/iterative_trsm.cpp
    src/trsm/recursive_trsm.cpp
    src/trsm/test_trsm.cpp
    ${COMMON_UTIL_SOURCES} # 打包引入重构后的公共工具源文件（替代旧 stream_pool.cpp）
    ${GEMM_SOURCES}        # 引入混合精度 GEMM 实现
    ${KERNEL_SOURCES}
)

# =========================================================
# SYRK Sources
# =========================================================

set(SYRK_SOURCES
    src/syrk/batched_syrk.cpp
    src/syrk/recursive_syrk.cpp
    src/syrk/test_syrk.cpp
    ${COMMON_UTIL_SOURCES} # 打包引入重构后的公共工具源文件（替代旧 stream_pool.cpp）
    ${GEMM_SOURCES}        # 引入混合精度 GEMM 实现
    ${KERNEL_SOURCES}
)

# =========================================================
# POTRF Sources
# =========================================================

set(POTRF_SOURCES
    src/potrf/recursive_potrf.cpp
    src/potrf/test_potrf.cpp
    src/trsm/recursive_trsm.cpp
    src/syrk/batched_syrk.cpp
    ${GEMM_SOURCES}        # 引入混合精度 GEMM 实现 (分块计算需要调用 mixed_gemm)
    ${COMMON_UTIL_SOURCES} # 打包引入重构后的公共工具源文件（替代旧 stream_pool.cpp）
    ${KERNEL_SOURCES}
)

# =========================================================
# TRSM Executable
# =========================================================

add_executable(benchmark_trsm
    ${TRSM_SOURCES}
)

# 针对 CXX 和 HIP 语言分别进行安全优化配置
target_compile_options(benchmark_trsm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

target_link_libraries(benchmark_trsm
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

# =========================================================
# SYRK Executable
# =========================================================

add_executable(benchmark_syrk
    ${SYRK_SOURCES}
)

target_compile_options(benchmark_syrk PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

target_link_libraries(benchmark_syrk
    rocblas
    hipblas
    amdhip64
    hsa-runtime64
)

# =========================================================
# POTRF Executable
# =========================================================

add_executable(benchmark_potrf
    ${POTRF_SOURCES}
)

target_compile_options(benchmark_potrf PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
)

target_link_libraries(benchmark_potrf
    rocblas
    rocsolver
    hipblas
    amdhip64
    hsa-runtime64
)