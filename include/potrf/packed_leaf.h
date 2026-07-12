#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void pack_potrf_leaf(const float* source,
                     int source_ld,
                     float* destination,
                     int destination_ld,
                     int n,
                     hipStream_t stream);

void unpack_potrf_leaf(const float* source,
                       int source_ld,
                       float* destination,
                       int destination_ld,
                       int n,
                       hipStream_t stream);

void pack_potrf_leaf_double(const double* source,
                            int source_ld,
                            double* destination,
                            int destination_ld,
                            int n,
                            hipStream_t stream);

void unpack_potrf_leaf_double(const double* source,
                              int source_ld,
                              double* destination,
                              int destination_ld,
                              int n,
                              hipStream_t stream);

#ifdef __cplusplus
}
#endif
