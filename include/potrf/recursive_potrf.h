#pragma once

#include "utils/memory/stream_pool.h"
#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void recursive_potrf(rocblas_handle handle, StreamPool* pool,
                     float* A,
                     int lda,
                     int n,
                     int nb,
                     int* info);

void recursive_potrf_double(rocblas_handle handle, StreamPool* pool,
                            double* A,
                            int lda,
                            int n,
                            int nb,
                            int* info);

void recursive_potrf_mixed_driver(rocblas_handle handle,
                                StreamPool* pool,
                                float* A,
                                int lda,
                                int n,
                                int nb,
                                int* info);

#ifdef __cplusplus
}
#endif