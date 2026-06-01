#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

/**
 * @brief 生成对称正定（SPD）测试矩阵： A = R * R^T + I
 */
template <typename T>
void generate_spd_matrix(rocblas_handle handle, T* d_A, int n, int lda, unsigned int seed = 2026);

/**
 * @brief 生成 SYRK 专用的测试矩阵 A (n x k) 暨 对称矩阵 C (n x n)
 */
template <typename T>
void generate_syrk_matrices(rocblas_handle handle, T* d_A, T* d_C, int n, int k, int lda, int ldc, unsigned int seed = 2026);

/**
 * @brief 生成稳定的下三角矩阵（Host 端）
 */
template <typename T>
void generate_stable_lower_triangular(T* A, int dim, int lda, unsigned int seed = 2026);

/**
 * @brief 生成普通的随机稠密矩阵（Host 端）
 */
template <typename T>
void generate_random_matrix(T* B, int n, int m, int ldb, unsigned int seed = 2026);