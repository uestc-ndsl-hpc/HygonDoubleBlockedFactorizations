#pragma once

#include <rocblas/rocblas.h>
#include "utils/stream_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// ======================================================
// Recursive TRSM (float)
// ======================================================

void recursive_trsm(
    rocblas_handle handle,
    StreamPool* pool,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb);

// ======================================================
// Recursive TRSM (double)
// ======================================================

void recursive_trsm_double(
    rocblas_handle handle,
    StreamPool* pool,
    double* A, int lda,
    double* B, int ldb,
    int n, int m, int nb);

// ======================================================
// Recursive TRSM (half)
// ======================================================

void recursive_trsm_half(
    rocblas_handle handle,
    StreamPool* pool,
    rocblas_half* A, int lda,
    rocblas_half* B, int ldb,
    int n, int m, int nb);


// ============================================================================
// 供单独测试使用的 FP32 包装接口
// ============================================================================    
void recursive_trsm_half_f32_wrapper(
    rocblas_handle handle,
    StreamPool* pool,
    float* A_f32, int lda,
    float* B_f32, int ldb,
    int n, int m, int nb);

#ifdef __cplusplus
}
#endif