#ifndef BATCHED_SYRK_H
#define BATCHED_SYRK_H

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 自底向上的 Batched SYRK 算法 (单精度)
 *        对应论文中 算法3：递归 SYRK 算法
 * 
 * @param handle rocBLAS 句柄
 * @param A 输入矩阵 A，大小为 n x k，列主序
 * @param lda 矩阵 A 的 leading dimension
 * @param C 目标对称矩阵 C，大小为 n x n，列主序
 * @param ldc 矩阵 C 的 leading dimension
 * @param n 矩阵行数
 * @param k 矩阵 A 的列数 (秩 k)
 * @param nb 基础分块大小 (Tile size)
 */
void batched_syrk_bottom_up(rocblas_handle handle,
                            float* A, int lda,
                            float* C, int ldc,
                            int n, int k, int nb);

/**
 * @brief 自底向上的 Batched SYRK 算法 (双精度)
 */
void batched_syrk_bottom_up_double(rocblas_handle handle,
                                   double* A, int lda,
                                   double* C, int ldc,
                                   int n, int k, int nb);

#ifdef __cplusplus
}
#endif

#endif // BATCHED_SYRK_H