#pragma once

#include "utils/common/rocblas_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

void double_blocked_getrf_npvt(rocblas_handle handle,
                               float* A,
                               int lda,
                               int n,
                               int nb,
                               int b,
                               int* info);

void double_blocked_getrf_npvt_double(rocblas_handle handle,
                                      double* A,
                                      int lda,
                                      int n,
                                      int nb,
                                      int b,
                                      int* info);

#ifdef __cplusplus
}
#endif
