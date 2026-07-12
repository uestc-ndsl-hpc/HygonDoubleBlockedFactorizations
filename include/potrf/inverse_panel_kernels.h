#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void pack_lower_triangular_for_inverse(const float* source,
                                       int source_ld,
                                       float* destination,
                                       int destination_ld,
                                       int n,
                                       hipStream_t stream);

void pack_lower_triangular_for_inverse_double(const double* source,
                                              int source_ld,
                                              double* destination,
                                              int destination_ld,
                                              int n,
                                              hipStream_t stream);

void zero_strict_upper_triangle(float* matrix,
                                int lda,
                                int n,
                                hipStream_t stream);

void zero_strict_upper_triangle_double(double* matrix,
                                       int lda,
                                       int n,
                                       hipStream_t stream);

void record_first_factorization_error(const int* local_info,
                                      int* global_info,
                                      int global_offset,
                                      hipStream_t stream);

#ifdef __cplusplus
}
#endif
