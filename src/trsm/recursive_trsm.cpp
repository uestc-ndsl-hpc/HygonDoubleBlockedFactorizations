#include "trsm/recursive_trsm.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>

extern "C" {

// ======================================================
// Internal recursive kernel (float)
// ======================================================
static void recursive_trsm_internal(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    if (n <= 0 || m <= 0) return;

    // leaf
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

    // --------------------------------------------------
    // Step 1
    // X1 * L11^T = B1
    // --------------------------------------------------
    recursive_trsm_internal(handle, A, lda, B, ldb, n, m1, nb);

    float alpha_gemm = -1.0f;
    float beta_gemm  =  1.0f;

    // ==================================================
    // Paper-style overlap:
    // split rows of B into:
    //   n_small = nb
    //   n_large = n - nb
    // ==================================================
    const bool use_pipeline = (n > nb && m >= 4096);

    if (use_pipeline) {

        int n_small = std::min(nb, n);
        int n_large = n - n_small;

        hipStream_t stream1, stream2;
        hipStreamCreate(&stream1);
        hipStreamCreate(&stream2);

        rocblas_handle h1, h2;
        rocblas_create_handle(&h1);
        rocblas_create_handle(&h2);

        rocblas_set_stream(h1, stream1);
        rocblas_set_stream(h2, stream2);

        // ----------------------------------------------
        // stream1:
        // small GEMM
        // small recursive TRSM
        // ----------------------------------------------
        rocblas_sgemm(
            h1,
            rocblas_operation_none,
            rocblas_operation_transpose,
            n_small, m2, m1,
            &alpha_gemm,
            B, ldb,
            A + m1, lda,
            &beta_gemm,
            B + (size_t)m1 * ldb, ldb);

        recursive_trsm_internal(
            h1,
            A + (size_t)m1 * lda + m1, lda,
            B + (size_t)m1 * ldb, ldb,
            n_small, m2, nb);

        // ----------------------------------------------
        // stream2:
        // large GEMM
        // ----------------------------------------------
        if (n_large > 0) {
            rocblas_sgemm(
                h2,
                rocblas_operation_none,
                rocblas_operation_transpose,
                n_large, m2, m1,
                &alpha_gemm,
                B + n_small, ldb,
                A + m1, lda,
                &beta_gemm,
                B + (size_t)m1 * ldb + n_small, ldb);
        }

        hipStreamSynchronize(stream2);

        // ----------------------------------------------
        // large recursive TRSM
        // ----------------------------------------------
        if (n_large > 0) {
            recursive_trsm_internal(
                handle,
                A + (size_t)m1 * lda + m1, lda,
                B + (size_t)m1 * ldb + n_small, ldb,
                n_large, m2, nb);
        }

        rocblas_destroy_handle(h1);
        rocblas_destroy_handle(h2);
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
    }
    else {
        // ----------------------------------------------
        // original path
        // ----------------------------------------------
        rocblas_sgemm(handle,
                      rocblas_operation_none,
                      rocblas_operation_transpose,
                      n, m2, m1,
                      &alpha_gemm,
                      B, ldb,
                      A + m1, lda,
                      &beta_gemm,
                      B + (size_t)m1 * ldb, ldb);

        recursive_trsm_internal(
            handle,
            A + (size_t)m1 * lda + m1, lda,
            B + (size_t)m1 * ldb, ldb,
            n, m2, nb);
    }
}

// ======================================================
// Internal recursive kernel (double)
// ======================================================
static void recursive_trsm_double_internal(
    rocblas_handle handle,
    double* A, int lda,
    double* B, int ldb,
    int n, int m, int nb)
{
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
    double beta_gemm  =  1.0;

    const bool use_pipeline = (n > nb && m >= 4096);

    if (use_pipeline) {

        int n_small = std::min(nb, n);
        int n_large = n - n_small;

        hipStream_t stream1, stream2;
        hipStreamCreate(&stream1);
        hipStreamCreate(&stream2);

        rocblas_handle h1, h2;
        rocblas_create_handle(&h1);
        rocblas_create_handle(&h2);

        rocblas_set_stream(h1, stream1);
        rocblas_set_stream(h2, stream2);

        rocblas_dgemm(
            h1,
            rocblas_operation_none,
            rocblas_operation_transpose,
            n_small, m2, m1,
            &alpha_gemm,
            B, ldb,
            A + m1, lda,
            &beta_gemm,
            B + (size_t)m1 * ldb, ldb);

        recursive_trsm_double_internal(
            h1,
            A + (size_t)m1 * lda + m1, lda,
            B + (size_t)m1 * ldb, ldb,
            n_small, m2, nb);

        if (n_large > 0) {
            rocblas_dgemm(
                h2,
                rocblas_operation_none,
                rocblas_operation_transpose,
                n_large, m2, m1,
                &alpha_gemm,
                B + n_small, ldb,
                A + m1, lda,
                &beta_gemm,
                B + (size_t)m1 * ldb + n_small, ldb);
        }

        hipStreamSynchronize(stream2);

        if (n_large > 0) {
            recursive_trsm_double_internal(
                handle,
                A + (size_t)m1 * lda + m1, lda,
                B + (size_t)m1 * ldb + n_small, ldb,
                n_large, m2, nb);
        }

        rocblas_destroy_handle(h1);
        rocblas_destroy_handle(h2);
        hipStreamDestroy(stream1);
        hipStreamDestroy(stream2);
    }
    else {
        rocblas_dgemm(handle,
                      rocblas_operation_none,
                      rocblas_operation_transpose,
                      n, m2, m1,
                      &alpha_gemm,
                      B, ldb,
                      A + m1, lda,
                      &beta_gemm,
                      B + (size_t)m1 * ldb, ldb);

        recursive_trsm_double_internal(
            handle,
            A + (size_t)m1 * lda + m1, lda,
            B + (size_t)m1 * ldb, ldb,
            n, m2, nb);
    }
}

// ======================================================
// External API
// ======================================================
void recursive_trsm(rocblas_handle handle,
                    float* A, int lda,
                    float* B, int ldb,
                    int n, int m, int nb)
{
    recursive_trsm_internal(handle, A, lda, B, ldb, n, m, nb);
}

void recursive_trsm_double(rocblas_handle handle,
                           double* A, int lda,
                           double* B, int ldb,
                           int n, int m, int nb)
{
    recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m, nb);
}

} // extern "C"