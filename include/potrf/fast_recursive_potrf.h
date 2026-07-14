#pragma once

#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void fast_recursive_strassen_potrf(rocblas_handle handle,
                                   float* A,
                                   int lda,
                                   int n,
                                   int leaf_size,
                                   int base_nb,
                                   int strassen_min_size,
                                   int syrk_k_tile,
                                   int* info);

void fast_recursive_strassen_potrf_double(rocblas_handle handle,
                                          double* A,
                                          int lda,
                                          int n,
                                          int leaf_size,
                                          int base_nb,
                                          int strassen_min_size,
                                          int syrk_k_tile,
                                          int* info);

#ifdef __cplusplus
}
#endif
