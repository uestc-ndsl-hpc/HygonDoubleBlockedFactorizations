#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void syrk_lower_update(rocblas_handle handle,
                       int n,
                       int k,
                       const float* A,
                       int lda,
                       float* C,
                       int ldc);

void syrk_lower_update_double(rocblas_handle handle,
                              int n,
                              int k,
                              const double* A,
                              int lda,
                              double* C,
                              int ldc);

void syrk_lower_update_kblocked(rocblas_handle handle,
                                int n,
                                int k,
                                const float* A,
                                int lda,
                                float* C,
                                int ldc,
                                int k_tile);

void syrk_lower_update_kblocked_double(rocblas_handle handle,
                                       int n,
                                       int k,
                                       const double* A,
                                       int lda,
                                       double* C,
                                       int ldc,
                                       int k_tile);

#ifdef __cplusplus
}
#endif
