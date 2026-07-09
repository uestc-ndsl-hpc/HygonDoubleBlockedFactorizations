#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void double_blocked_trsm(rocblas_handle handle,
                         float* A,
                         int lda,
                         float* B,
                         int ldb,
                         int n,
                         int m,
                         int nb,
                         int b);

void double_blocked_trsm_double(rocblas_handle handle,
                                double* A,
                                int lda,
                                double* B,
                                int ldb,
                                int n,
                                int m,
                                int nb,
                                int b);

#ifdef __cplusplus
}
#endif
