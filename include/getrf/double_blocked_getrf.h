#pragma once

#include "utils/common/rocblas_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

void double_blocked_getrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int outer_nb,
                          int inner_nb,
                          int trsm_leaf,
                          int* ipiv,
                          int* info);

void double_blocked_getrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int outer_nb,
                                 int inner_nb,
                                 int trsm_leaf,
                                 int* ipiv,
                                 int* info);

#ifdef __cplusplus
}
#endif
