#pragma once

#include <hip/hip_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_apply_getrf_panel_pivots(float* A,
                                     int lda,
                                     int n,
                                     int panel,
                                     int panel_size,
                                     int* pivots,
                                     hipStream_t stream);

void launch_apply_getrf_panel_pivots_double(double* A,
                                            int lda,
                                            int n,
                                            int panel,
                                            int panel_size,
                                            int* pivots,
                                            hipStream_t stream);

void launch_apply_getrf_permutation(float* A,
                                    int lda,
                                    int column_count,
                                    const int* sources,
                                    const int* targets,
                                    int move_count,
                                    float* temporary,
                                    int column_tile,
                                    hipStream_t stream);

void launch_apply_getrf_permutation_double(double* A,
                                           int lda,
                                           int column_count,
                                           const int* sources,
                                           const int* targets,
                                           int move_count,
                                           double* temporary,
                                           int column_tile,
                                           hipStream_t stream);

#ifdef __cplusplus
}
#endif
