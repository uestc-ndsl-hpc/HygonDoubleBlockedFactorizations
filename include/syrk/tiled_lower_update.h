#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void tiled_lower_rank_k_update(rocblas_handle handle,
                               int n,
                               int k,
                               const float* A,
                               int lda,
                               float* C,
                               int ldc,
                               int n_tile,
                               int k_tile);

void tiled_lower_rank_k_update_double(rocblas_handle handle,
                                      int n,
                                      int k,
                                      const double* A,
                                      int lda,
                                      double* C,
                                      int ldc,
                                      int n_tile,
                                      int k_tile);

#ifdef __cplusplus
}
#endif
