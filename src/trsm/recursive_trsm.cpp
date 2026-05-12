#include "trsm/recursive_trsm.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

extern "C" {

// ======================================================
// 单精度 Right-TRSM: 求解 X * L^T = B
// n: 目标矩阵 B 的行数
// m: 目标矩阵 B 的列数 (同时也是下三角矩阵 L 的维度)
// ======================================================
void recursive_trsm(rocblas_handle handle,
                    float* A, int lda,
                    float* B, int ldb,
                    int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    // 递归终止条件：当规模小于阈值时，调用 rocblas 原生接口
    if (m <= nb) {
        float alpha = 1.0f;
        rocblas_strsm(handle,
                      rocblas_side_right,          // 右乘 (侧重点)
                      rocblas_fill_lower,          // 下三角矩阵
                      rocblas_operation_transpose, // 转置 (L^T)
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    // 对于 Right-TRSM，我们是对三角矩阵的维度 m 进行切分
    int m1 = m / 2;
    int m2 = m - m1;

    // 数学推导：X * [L11^T, L21^T; 0, L22^T] = [B1, B2]
    // 拆分后产生两个方程：
    // 1) X1 * L11^T = B1
    // 2) X1 * L21^T + X2 * L22^T = B2  ==>  X2 * L22^T = B2 - X1 * L21^T

    // 步骤 1: 求解 X1 * L11^T = B1
    // B1 占据 B 的前 m1 列，维度为 n x m1
    recursive_trsm(handle, A, lda, B, ldb, n, m1, nb);

    // 步骤 2: 更新 B2 = B2 - X1 * L21^T (使用 GEMM)
    // B2 位于 B 的后 m2 列。指针偏移 = B + m1 * ldb
    // X1 刚刚在 B1 的位置被求出，指针直接是 B
    // L21 位于 A 的左下角，指针偏移 = A + m1
    float alpha_gemm = -1.0f;
    float beta_gemm = 1.0f;
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose, // 核心：对 L21 进行转置
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,                      // X1
                  A + m1, lda,                 // L21
                  &beta_gemm,
                  B + (size_t)m1 * ldb, ldb);  // B2

    // 步骤 3: 求解 X2 * L22^T = B2
    // L22 位于 A 的右下角。指针偏移 = A + m1 * lda + m1
    recursive_trsm(handle, A + (size_t)m1 * lda + m1, lda, B + (size_t)m1 * ldb, ldb, n, m2, nb);
}

// ======================================================
// 双精度 Right-TRSM
// ======================================================
void recursive_trsm_double(rocblas_handle handle,
                           double* A, int lda,
                           double* B, int ldb,
                           int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        double alpha = 1.0;
        rocblas_dtrsm(handle,
                      rocblas_side_right,
                      rocblas_fill_lower,
                      rocblas_operation_transpose,
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_double(handle, A, lda, B, ldb, n, m1, nb);

    double alpha_gemm = -1.0;
    double beta_gemm = 1.0;
    rocblas_dgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose,
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,
                  A + m1, lda,
                  &beta_gemm,
                  B + (size_t)m1 * ldb, ldb);

    recursive_trsm_double(handle, A + (size_t)m1 * lda + m1, lda, B + (size_t)m1 * ldb, ldb, n, m2, nb);
}

} // extern "C"