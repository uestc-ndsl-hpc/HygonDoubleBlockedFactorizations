cmake_minimum_required(VERSION 3.10)

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
# Define Kernel Sources (核心：添加缺失的 HIP 物理源文件)
# =========================================================

set(KERNEL_SOURCES
    src/kernels/add_damping.hip
    src/kernels/fp16_convert.hip
    src/kernels/matrix_scale.hip
    # 如果后续需要使用带自适应缩放的转换，可以取消注释下面这行：
    # src/kernels/fp16_convert_with_matrix_scale.hip
)

# 显式指示 CMake 将这些文件作为 HIP 代码编译（兼容旧版 CMake）
set_source_files_properties(${KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)

# =========================================================
# TRSM Sources
# =========================================================

set(TRSM_SOURCES
    src/trsm/iterative_trsm.cpp
    src/trsm/recursive_trsm.cpp
    src/utils/stream_pool.cpp
    src/trsm/test_trsm.cpp
    ${KERNEL_SOURCES} # 将核函数打包编译进可执行文件
)

# =========================================================
# SYRK Sources
# =========================================================

set(SYRK_SOURCES
    src/syrk/batched_syrk.cpp
    src/syrk/recursive_syrk.cpp
    src/utils/stream_pool.cpp
    src/syrk/test_syrk.cpp
    ${KERNEL_SOURCES} # 将核函数打包编译进可执行文件
)

# =========================================================
# POTRF Sources
# =========================================================

set(POTRF_SOURCES
    src/potrf/recursive_potrf.cpp
    src/potrf/test_potrf.cpp
    src/trsm/recursive_trsm.cpp
    src/utils/stream_pool.cpp
    src/syrk/batched_syrk.cpp
    ${KERNEL_SOURCES} # 将核函数打包编译进可执行文件
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