#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void gemm_nt_update(rocblas_handle handle,
                    int rows,
                    int cols,
                    int k,
                    const float* A,
                    int lda,
                    const float* B,
                    int ldb,
                    float* C,
                    int ldc);

void gemm_nt_update_double(rocblas_handle handle,
                           int rows,
                           int cols,
                           int k,
                           const double* A,
                           int lda,
                           const double* B,
                           int ldb,
                           double* C,
                           int ldc);

void gemm_nt_update_m_tiled(rocblas_handle handle,
                            int rows,
                            int cols,
                            int k,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            float* C,
                            int ldc,
                            int m_tile);

void gemm_nt_update_m_tiled_double(rocblas_handle handle,
                                   int rows,
                                   int cols,
                                   int k,
                                   const double* A,
                                   int lda,
                                   const double* B,
                                   int ldb,
                                   double* C,
                                   int ldc,
                                   int m_tile);

#ifdef __cplusplus
}
#endif
