#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void double_blocked_left_looking_potrf(rocblas_handle handle,
                                       float* A,
                                       int lda,
                                       int n,
                                       int nb,
                                       int b,
                                       int* info);

void double_blocked_left_looking_potrf_optimized(rocblas_handle handle,
                                                 float* A,
                                                 int lda,
                                                 int n,
                                                 int nb,
                                                 int b,
                                                 int syrk_k_tile,
                                                 int gemm_m_tile,
                                                 int* info);

void double_blocked_left_looking_potrf_double(rocblas_handle handle,
                                              double* A,
                                              int lda,
                                              int n,
                                              int nb,
                                              int b,
                                              int* info);

void double_blocked_left_looking_potrf_double_optimized(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int gemm_m_tile,
    int* info);

#ifdef __cplusplus
}
#endif
