cmake_minimum_required(VERSION 3.16)

project(HygonCholesky LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =========================================================
# ROCm / DTK 路径配置
# =========================================================
set(ROCM_PATH /public/software/compiler/rocm/dtk-24.04)

# 定义库文件的绝对路径，彻底解决 "cannot find -lxxx" 问题
set(ROCBLAS_LIB   ${ROCM_PATH}/lib/librocblas.so)
set(ROCSOLVER_LIB ${ROCM_PATH}/lib/librocsolver.so)
set(HIP_LIB       ${ROCM_PATH}/lib/libamdhip64.so)

# =========================================================
# 包含路径
# =========================================================
include_directories(
    ${ROCM_PATH}/include
    ${PROJECT_SOURCE_DIR}/include
)

# =========================================================
# 创建核心算法静态库 (cholesky_lib)
# =========================================================
set(LIB_SRCS
    src/kernels/add_damping.hip
    src/kernels/fp16_convert.hip
    src/kernels/matrix_scale.hip
    src/utils/stream_pool.cpp
    src/trsm/recursive_trsm.cpp
    src/trsm/iterative_trsm.cpp  
    src/syrk/batched_syrk.cpp
    src/syrk/recursive_syrk.cpp  
    src/potrf/recursive_potrf.cpp
)
add_library(cholesky_lib STATIC ${LIB_SRCS})

# 设置编译宏和选项
target_compile_definitions(cholesky_lib PUBLIC __HIP_PLATFORM_AMD__)
target_compile_options(cholesky_lib PRIVATE -O3 -Wall)
set_target_properties(cholesky_lib PROPERTIES HIP_STANDARD 17)

# =========================================================
# 构建各模块 Benchmark 可执行文件 (已删除 test_kernels)
# =========================================================

# --- 1. TRSM ---
add_executable(benchmark_trsm src/trsm/test_trsm.cpp)
target_link_libraries(benchmark_trsm PRIVATE 
    cholesky_lib 
    ${ROCBLAS_LIB} 
    ${HIP_LIB}
)

# --- 2. SYRK ---
add_executable(benchmark_syrk src/syrk/test_syrk.cpp)
target_link_libraries(benchmark_syrk PRIVATE 
    cholesky_lib 
    ${ROCBLAS_LIB} 
    ${HIP_LIB}
)

# --- 3. POTRF ---
add_executable(benchmark_potrf src/potrf/test_potrf.cpp)
target_link_libraries(benchmark_potrf PRIVATE 
    cholesky_lib 
    ${ROCBLAS_LIB} 
    ${ROCSOLVER_LIB} 
    ${HIP_LIB}
)

# --- 4. GEMM (新增：测试两种 launch_mixed_gemm 速度的测试程序) ---
# 注：根据您的实际目录结构，如果您的代码文件放在项目根目录下的 utils/ 目录，
# 请将下方的路径修改为 utils/test_gemm.cpp
add_executable(benchmark_gemm src/utils/test_gemm.cpp)
target_link_libraries(benchmark_gemm PRIVATE 
    cholesky_lib 
    ${ROCBLAS_LIB} 
    ${HIP_LIB}
)

# 统一设置优化选项 (包含新增的 benchmark_gemm)
foreach(exe benchmark_trsm benchmark_syrk benchmark_potrf benchmark_gemm)
    target_compile_options(${exe} PRIVATE -O3)
endforeach()