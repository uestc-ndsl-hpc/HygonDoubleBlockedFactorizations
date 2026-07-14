#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void gemm_nt_update_strassen_tiled(rocblas_handle handle,
                                   int rows,
                                   int cols,
                                   int k,
                                   const float* A,
                                   int lda,
                                   const float* B,
                                   int ldb,
                                   float* C,
                                   int ldc,
                                   int strassen_min_size);

void gemm_nt_update_strassen_tiled_double(rocblas_handle handle,
                                          int rows,
                                          int cols,
                                          int k,
                                          const double* A,
                                          int lda,
                                          const double* B,
                                          int ldb,
                                          double* C,
                                          int ldc,
                                          int strassen_min_size);

#ifdef __cplusplus
}
#endif
