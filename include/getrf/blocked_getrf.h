#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void blocked_right_looking_getrf(rocblas_handle handle,
                                 float* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int* ipiv,
                                 int* info);

void blocked_right_looking_getrf_double(rocblas_handle handle,
                                        double* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int* ipiv,
                                        int* info);

#ifdef __cplusplus
}
#endif
