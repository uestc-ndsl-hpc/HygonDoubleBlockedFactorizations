#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

// ======================================================
// Recursive TRSM (float)
// ======================================================

void recursive_trsm(
    rocblas_handle handle,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb);

// ======================================================
// Recursive TRSM (double)
// ======================================================

void recursive_trsm_double(
    rocblas_handle handle,
    double* A, int lda,
    double* B, int ldb,
    int n, int m, int nb);

void recursive_trsm_mixed(
    rocblas_handle handle,
    float* A,
    int lda,

    float* B,
    int ldb,

    int n,
    int m,
    int nb);

#ifdef __cplusplus
}
#endif
