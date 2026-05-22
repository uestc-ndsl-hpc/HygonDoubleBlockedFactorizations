#pragma once

#include <random>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "utils/error_check.h"
#include "kernels/add_damping.h"

/**
 * @brief 生成对称正定（SPD）测试矩阵： A = R * R^T + I
 * 其中 R 的元素服从均值为 0，方差为 1 的标准正态分布。
 * * @tparam T 矩阵元素类型 (float 或 double)
 * @param handle rocBLAS 句柄
 * @param d_A 设备端指针，存储最终生成的 n x n 测试矩阵 (外部已分配显存)
 * @param n 矩阵的维度
 * @param lda 矩阵的领先维度，通常等于 n
 * @param seed 随机数种子
 */
template <typename T>
void generate_spd_matrix(rocblas_handle handle, T* d_A, int n, int lda, unsigned int seed = 2026)
{
    size_t num_elements = (size_t)n * (size_t)lda;
    size_t bytes = num_elements * sizeof(T);

    // 1. 在 CPU 上生成随机正态分布数据 R
    std::vector<T> h_R(num_elements);
    std::mt19937 gen(seed);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (size_t i = 0; i < num_elements; ++i) {
        h_R[i] = static_cast<T>(norm_dist(gen));
    }

    // 2. 将随机矩阵 R 拷贝到设备端
    T* d_R = nullptr;
    HIP_CHECK(hipMalloc(&d_R, bytes));
    HIP_CHECK(hipMemcpy(d_R, h_R.data(), bytes, hipMemcpyHostToDevice));

    // 3. 在 GPU 上利用 GEMM 计算外积 d_A = d_R * d_R^T
    T alpha = static_cast<T>(1.0);
    T beta  = static_cast<T>(0.0);
    
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(handle, 
                                    rocblas_operation_none, rocblas_operation_transpose,
                                    n, n, n, 
                                    &alpha, 
                                    d_R, lda, 
                                    d_R, lda, 
                                    &beta, 
                                    d_A, lda));
    } else if constexpr (std::is_same_v<T, double>) {
        ROCBLAS_CHECK(rocblas_dgemm(handle, 
                                    rocblas_operation_none, rocblas_operation_transpose,
                                    n, n, n, 
                                    &alpha, 
                                    d_R, lda, 
                                    d_R, lda, 
                                    &beta, 
                                    d_A, lda));
    }

    // 释放临时矩阵 d_R
    HIP_CHECK(hipFree(d_R));

    // 4. 获取当前 rocBLAS 句柄内部使用的 stream，保证 Kernel 按顺序正确执行
    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    // 5. 直接在 GPU 上给对角线元素加 1.0f (加上单位矩阵 I)
    if constexpr (std::is_same_v<T, float>) {
        // 引用自 kernels/add_damping.h 的 C 接口
        launch_add_damping(d_A, n, lda, 1.0f, stream);
    } else {
        // 处理 double 类型的分支
        launch_add_damping_double(d_A, n, lda, 1.0, stream);
    }
    
    // 确保流同步，防止后续算法在使用 d_A 时数据还未就绪
    HIP_CHECK(hipStreamSynchronize(stream));
}


/**
 * @tparam T 矩阵元素类型 (float 或 double)
 * @param handle rocBLAS 句柄
 * @param d_A 设备端指针，存储生成的 n x k 矩阵 A (外部已分配显存)
 * @param d_C 设备端指针，存储生成的 n x n 对称矩阵 C (外部已分配显存)
 * @param n 矩阵的行数 (对于A) 暨 矩阵维度 (对于C)
 * @param k 矩阵 A 的列数 (秩更新级数)
 * @param lda A 矩阵的领先维度，通常等于 n
 * @param ldc C 矩阵的领先维度，通常等于 n
 * @param seed 随机数种子
 */
template <typename T>
void generate_syrk_matrices(rocblas_handle handle, T* d_A, T* d_C, int n, int k, int lda, int ldc, unsigned int seed = 2026)
{
    // 调用上面优化后的函数生成符合论文要求的正定矩阵 C
    generate_spd_matrix<T>(handle, d_C, n, ldc, seed);

    // 生成 SYRK 专用的测试矩阵 A (n x k)
    size_t num_elements_A = (size_t)lda * (size_t)k; 
    size_t bytes_A = num_elements_A * sizeof(T);
    std::vector<T> h_A(num_elements_A);
    
    std::mt19937 gen(seed + 1);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (size_t i = 0; i < num_elements_A; ++i) {
        h_A[i] = static_cast<T>(norm_dist(gen));
    }

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes_A, hipMemcpyHostToDevice));
}

// ============================================================================
// TRSM MATRIX GENERATORS
// ============================================================================

/**
 * @brief 生成稳定的下三角矩阵（Host 端）
 * 对角线元素加上 dim 保证矩阵具有严格的对角占优特性，从而确保求逆/三角解的数值稳定性。
 * 
 * @tparam T 矩阵元素类型 (float 或 double)
 * @param A 主机端指针，存储生成的 dim x dim 下三角矩阵
 * @param dim 矩阵的维度
 * @param lda 矩阵的领先维度
 * @param seed 随机数种子（默认 2026）
 */
template <typename T>
void generate_stable_lower_triangular(T* A, int dim, int lda, unsigned int seed = 2026)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {
            if (i < j) {
                A[j * lda + i] = static_cast<T>(0.0);
            }
            else if (i == j) {
                // 对角线显式加 dim 保证数值条件数稳定
                A[j * lda + i] = static_cast<T>(dim + dist(gen));
            }
            else {
                A[j * lda + i] = static_cast<T>(dist(gen));
            }
        }
    }
}

/**
 * @brief 生成普通的随机稠密矩阵（Host 端）
 * 元素分布在 [-0.5, 0.5] 区间内。
 * 
 * @tparam T 矩阵元素类型 (float 或 double)
 * @param B 主机端指针，存储生成的 n x m 矩阵
 * @param n 矩阵的行数
 * @param m 矩阵的列数
 * @param ldb 矩阵的领先维度
 * @param seed 随机数种子（默认 2026）
 */
template <typename T>
void generate_random_matrix(T* B, int n, int m, int ldb, unsigned int seed = 2026)
{
    std::mt19937 gen(seed + 1);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            B[j * ldb + i] = static_cast<T>(dist(gen));
        }
    }
}