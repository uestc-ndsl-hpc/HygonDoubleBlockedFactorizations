#pragma once

#include <rocblas/rocblas.h>

class StageBreakdownProfiler;

void double_blocked_potrf_inverse_panel_profiled(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    float* inverse_workspace,
    int inverse_ld,
    float* panel_workspace,
    int panel_workspace_rows,
    int* local_info,
    int* info,
    StageBreakdownProfiler& profiler);
