#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct getrf_batched_pivot_workspace {
    int* device_sources;
    int* device_targets;
    void* device_swap_buffer;
    int* host_pivots;
    int* host_permutation;
    int* host_sources;
    int* host_targets;
    int capacity;
    int column_tile;
} getrf_batched_pivot_workspace;

typedef enum getrf_outer_trsm_mode {
    getrf_outer_trsm_direct = 0,
    getrf_outer_trsm_iterative = 1
} getrf_outer_trsm_mode;

void double_blocked_getrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int outer_nb,
                          int inner_nb,
                          getrf_outer_trsm_mode outer_trsm_mode,
                          int trsm_block,
                          int* ipiv,
                          int* info);

void double_blocked_getrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int outer_nb,
                                 int inner_nb,
                                 getrf_outer_trsm_mode outer_trsm_mode,
                                 int trsm_block,
                                 int* ipiv,
                                 int* info);

void double_blocked_getrf_batched_pivot(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_nb,
    int inner_nb,
    getrf_outer_trsm_mode outer_trsm_mode,
    int trsm_block,
    int inner_gemm_m_tile,
    int outer_gemm_m_tile,
    int* ipiv,
    int* info,
    getrf_batched_pivot_workspace* workspace);

void double_blocked_getrf_double_batched_pivot(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int outer_nb,
    int inner_nb,
    getrf_outer_trsm_mode outer_trsm_mode,
    int trsm_block,
    int inner_gemm_m_tile,
    int outer_gemm_m_tile,
    int* ipiv,
    int* info,
    getrf_batched_pivot_workspace* workspace);

#ifdef __cplusplus
}
#endif
