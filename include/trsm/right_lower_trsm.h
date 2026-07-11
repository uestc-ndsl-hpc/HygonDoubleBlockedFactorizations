#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void trsm_right_lower_transpose(rocblas_handle handle,
                                int rows,
                                int cols,
                                const float* A,
                                int lda,
                                float* B,
                                int ldb);

void trsm_right_lower_transpose_double(rocblas_handle handle,
                                       int rows,
                                       int cols,
                                       const double* A,
                                       int lda,
                                       double* B,
                                       int ldb);

#ifdef __cplusplus
}
#endif
