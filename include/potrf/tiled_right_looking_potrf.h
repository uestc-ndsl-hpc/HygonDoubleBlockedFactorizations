#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void tiled_right_looking_double_blocked_potrf(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info);

void tiled_right_looking_double_blocked_potrf_double(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info);

#ifdef __cplusplus
}
#endif
