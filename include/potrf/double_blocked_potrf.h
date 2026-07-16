#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void double_blocked_potrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int nb,
                          int b,
                          int* info);

void double_blocked_potrf_dbtrsm(rocblas_handle handle,
                                 float* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info);

void double_blocked_potrf_kblocked_syrk(rocblas_handle handle,
                                        float* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int syrk_k_tile,
                                        int* info);

void double_blocked_potrf_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info);

void double_blocked_potrf_kblocked_syrk_adaptive_gemm_configured(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int gemm_m_tile,
    int gemm_m_threshold,
    int gemm_k_threshold,
    int* info);

void double_blocked_potrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info);

void double_blocked_potrf_double_dbtrsm(rocblas_handle handle,
                                        double* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int* info);

void double_blocked_potrf_double_kblocked_syrk(rocblas_handle handle,
                                               double* A,
                                               int lda,
                                               int n,
                                               int nb,
                                               int b,
                                               int syrk_k_tile,
                                               int* info);

void double_blocked_potrf_double_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info);

void double_blocked_potrf_double_kblocked_syrk_adaptive_gemm_configured(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int gemm_m_tile,
    int gemm_m_threshold,
    int gemm_k_threshold,
    int* info);

#ifdef __cplusplus
}
#endif
