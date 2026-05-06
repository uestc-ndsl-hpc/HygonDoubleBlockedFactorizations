#include "trsm/recursive_trsm.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

extern "C" {

// 单精度递归 TRSM
void recursive_trsm(rocblas_handle handle,
                    float* A, int lda,
                    float* B, int ldb,
                    int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    // 递归终止条件：当规模小于阈值时，调用 rocblas 原生接口
    if (n <= nb) {
        float alpha = 1.0f;
        rocblas_strsm(handle,
                      rocblas_side_left,
                      rocblas_fill_lower,
                      rocblas_operation_none,
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    // 分块点：将 n 维度一分为二
    int n1 = n / 2;
    int n2 = n - n1;

    // 下三角矩阵 L 的分块逻辑：
    // [ L11  0  ] * [ X1 ] = [ B1 ]
    // [ L21 L22 ]   [ X2 ]   [ B2 ]

    // 步骤 1: 求解上半部分 L11 * X1 = B1
    recursive_trsm(handle, A, lda, B, ldb, n1, m, nb);

    // 步骤 2: 更新下半部分 B2 = B2 - L21 * X1 (使用 GEMM)
    // L21 位于 A 的第 n1 行，第 0 列。指针偏移 = A + n1
    // X1  位于 B 的第 0 行，起始位置就是 B
    // B2  位于 B 的第 n1 行。指针偏移 = B + n1
    float alpha_gemm = -1.0f;
    float beta_gemm = 1.0f;
    rocblas_sgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  n2, m, n1,
                  &alpha_gemm,
                  A + n1, lda,           // L21
                  B, ldb,                // X1
                  &beta_gemm,
                  B + n1, ldb);          // B2

    // 步骤 3: 求解下半部分 L22 * X2 = B2
    // L22 位于 A 的第 n1 行，第 n1 列。指针偏移 = A + n1 * lda + n1
    // B2  位于 B 的第 n1 行。指针偏移 = B + n1
    recursive_trsm(handle, A + n1 * lda + n1, lda, B + n1, ldb, n2, m, nb);
}

// 双精度递归 TRSM
void recursive_trsm_double(rocblas_handle handle,
                           double* A, int lda,
                           double* B, int ldb,
                           int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (n <= nb) {
        double alpha = 1.0;
        rocblas_dtrsm(handle,
                      rocblas_side_left,
                      rocblas_fill_lower,
                      rocblas_operation_none,
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    recursive_trsm_double(handle, A, lda, B, ldb, n1, m, nb);

    double alpha_gemm = -1.0;
    double beta_gemm = 1.0;
    rocblas_dgemm(handle,
                  rocblas_operation_none, rocblas_operation_none,
                  n2, m, n1,
                  &alpha_gemm,
                  A + n1, lda,
                  B, ldb,
                  &beta_gemm,
                  B + n1, ldb);

    recursive_trsm_double(handle, A + n1 * lda + n1, lda, B + n1, ldb, n2, m, nb);
}

} // extern "C"