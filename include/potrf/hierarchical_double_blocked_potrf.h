#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void hierarchical_double_blocked_potrf(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int panel_block_size,
    int potrf_leaf_size,
    int syrk_k_tile,
    int gemm_m_tile,
    int* info);

#ifdef __cplusplus
}
#endif
