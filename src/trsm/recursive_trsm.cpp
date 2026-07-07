#include "trsm/recursive_trsm.h"
#include "utils/common/error_check.h"
#include "kernels/fp16_convert.h"
#include "gemm/mixed_gemm.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <algorithm>

namespace {

constexpr int kTrsmUpdateTile = 8192;

void trsm_update_gemm(
    rocblas_handle handle,
    float* A,
    int lda,
    float* B,
    int ldb,
    int n,
    int m1,
    int m2)
{
    float alpha_gemm = -1.0f;
    float beta_gemm = 1.0f;

    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose,
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,
                  A + (size_t)m1, lda,
                  &beta_gemm,
                  B + (size_t)m1 * (size_t)ldb, ldb);
}

void trsm_update_gemm_tiled(
    rocblas_handle handle,
    float* A,
    int lda,
    float* B,
    int ldb,
    int n,
    int m1,
    int m2)
{
    if (n <= kTrsmUpdateTile && m1 <= kTrsmUpdateTile && m2 <= kTrsmUpdateTile) {
        trsm_update_gemm(handle, A, lda, B, ldb, n, m1, m2);
        return;
    }

    float alpha_gemm = -1.0f;
    float beta_gemm = 1.0f;

    for (int row = 0; row < n; row += kTrsmUpdateTile) {
        int rows = std::min(kTrsmUpdateTile, n - row);

        for (int col = 0; col < m2; col += kTrsmUpdateTile) {
            int cols = std::min(kTrsmUpdateTile, m2 - col);
            float* C_tile = B + (size_t)(m1 + col) * (size_t)ldb + row;

            for (int kk = 0; kk < m1; kk += kTrsmUpdateTile) {
                int k_tile = std::min(kTrsmUpdateTile, m1 - kk);

                rocblas_sgemm(
                    handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    rows,
                    cols,
                    k_tile,
                    &alpha_gemm,
                    B + (size_t)kk * (size_t)ldb + row,
                    ldb,
                    A + (size_t)kk * (size_t)lda + m1 + col,
                    lda,
                    &beta_gemm,
                    C_tile,
                    ldb);
            }
        }
    }
}

} // namespace

extern "C" {

// ======================================================
// 内部纯递归执行器 (单精度)
// ======================================================
void recursive_trsm_internal(rocblas_handle handle,
                             float* A, int lda,
                             float* B, int ldb,
                             int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        float alpha = 1.0f;
        rocblas_strsm(handle,
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

    recursive_trsm_internal(handle, A, lda, B, ldb, n, m1, nb);

    trsm_update_gemm(handle, A, lda, B, ldb, n, m1, m2);

    recursive_trsm_internal(handle, A + (size_t)m1 * (size_t)lda + m1, lda, B + (size_t)m1 * (size_t)ldb, ldb, n, m2, nb);
    
    return;
}

void recursive_trsm_tiled_update_internal(rocblas_handle handle,
                                          float* A, int lda,
                                          float* B, int ldb,
                                          int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        float alpha = 1.0f;
        rocblas_strsm(handle,
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

    recursive_trsm_tiled_update_internal(handle, A, lda, B, ldb, n, m1, nb);

    trsm_update_gemm_tiled(handle, A, lda, B, ldb, n, m1, m2);

    recursive_trsm_tiled_update_internal(
        handle,
        A + (size_t)m1 * (size_t)lda + m1,
        lda,
        B + (size_t)m1 * (size_t)ldb,
        ldb,
        n,
        m2,
        nb);

    return;
}

// ======================================================
// 内部纯递归执行器 (双精度)
// ======================================================
void recursive_trsm_double_internal(rocblas_handle handle,
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

    recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m1, nb);

    double alpha_gemm = -1.0;
    double beta_gemm = 1.0;
    rocblas_dgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose,
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,
                  A + (size_t)m1, lda,
                  &beta_gemm,
                  B + (size_t)m1 * (size_t)ldb, ldb);

    recursive_trsm_double_internal(handle, A + (size_t)m1 * (size_t)lda + m1, lda, B + (size_t)m1 * (size_t)ldb, ldb, n, m2, nb);
    
    return;
}

// ======================================================
// Public API (float)
// ======================================================
void recursive_trsm(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    recursive_trsm_internal(handle, A, lda, B, ldb, n, m, nb);
    return;
}

void recursive_trsm_tiled_update(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    recursive_trsm_tiled_update_internal(handle, A, lda, B, ldb, n, m, nb);
    return;
}

// ======================================================
// Public API (double)
// ======================================================
void recursive_trsm_double(
    rocblas_handle handle,
    double* A, int lda,
    double* B, int ldb,
    int n, int m, int nb)
{
    recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m, nb);
    return;
}

// ======================================================
// 内部纯递归执行器 (混合精度)
// ======================================================
void recursive_trsm_mixed_internal(
    rocblas_handle handle,
    float* A,
    int lda,
    float* B,
    int ldb,
    int n,
    int m,
    int nb)
{
    if (n <= 0 || m <= 0)
        return;

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    if (m <= nb)
    {
        float alpha = 1.0f;

        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            n,
            m,
            &alpha,
            A,
            lda,
            B,
            ldb));

        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_mixed_internal(
        handle,
        A,
        lda,
        B,
        ldb,
        n,
        m1,
        nb);

    launch_mixed_gemm(
        handle,
        stream,
        rocblas_operation_none,       // B1 不转置
        rocblas_operation_transpose,  // A21 转置
        n,                            // GEMM M 维
        m2,                           // GEMM N 维
        m1,                           // GEMM K 维
        -1.0f,                        // alpha = -1.0
        B, ldb,                       // 对应矩阵 A_gemm (即 B1)
        A + (size_t)m1, lda,          // 对应矩阵 B_gemm (即 A21)
        1.0f,                         // beta = 1.0
        B + (size_t)m1 * (size_t)ldb, ldb     // 对应输出矩阵 C_gemm (即 B2)
    );

    recursive_trsm_mixed_internal(
        handle,
        A + (size_t)m1 * (size_t)lda + m1,
        lda,
        B + (size_t)m1 * (size_t)ldb,
        ldb,
        n,
        m2,
        nb);
        
    return;
}

// ======================================================
// Public API (mixed)
// ======================================================
void recursive_trsm_mixed(
    rocblas_handle handle,
    float* A,
    int lda,
    float* B,
    int ldb,
    int n,
    int m,
    int nb)
{
    recursive_trsm_mixed_internal(handle, A, lda, B, ldb, n, m, nb);
    return;
}

} // extern "C"
