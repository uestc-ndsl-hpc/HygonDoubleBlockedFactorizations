#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void recursive_potrf(rocblas_handle handle,
                     float* A,
                     int lda,
                     int n,
                     int nb,
                     int* info);

void recursive_potrf_double(rocblas_handle handle,
                            double* A,
                            int lda,
                            int n,
                            int nb,
                            int* info);

#ifdef __cplusplus
}
#endif