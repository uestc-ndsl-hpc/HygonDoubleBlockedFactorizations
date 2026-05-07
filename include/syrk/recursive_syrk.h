#ifndef RECURSIVE_SYRK_H
#define RECURSIVE_SYRK_H

#include <rocblas/rocblas.h>

extern "C" {

/**
 * @brief 并行递归 SYRK (单精度)
 * @param handle 原始 handle (主要用于同步和默认流)
 * @param n 矩阵阶数
 * @param k 秩
 * @param nb 递归截止的分块大小
 */
void recursive_syrk_parallel(
    rocblas_handle handle,
    float* A, int lda,
    float* C, int ldc,
    int n, int k,
    int nb);

/**
 * @brief 并行递归 SYRK (双精度)
 */
void recursive_syrk_parallel_double(
    rocblas_handle handle,
    double* A, int lda,
    double* C, int ldc,
    int n, int k,
    int nb);

}

#endif // RECURSIVE_SYRK_H