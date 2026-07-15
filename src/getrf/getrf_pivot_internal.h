#pragma once

#include <algorithm>
#include <type_traits>

#include <rocsolver/rocsolver.h>

#include "getrf/getrf_pivot_kernels.h"
#include "getrf_npvt_internal.h"

namespace getrf_pivot_internal {

using getrf_npvt_internal::block_ptr;
using getrf_npvt_internal::gemm_nn_update;
using getrf_npvt_internal::trsm_left_lower_unit;

template <typename T>
void factor_panel(rocblas_handle handle,
                  int rows,
                  int cols,
                  T* A,
                  int lda,
                  int* ipiv,
                  int* info)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(
            rocsolver_sgetrf(handle, rows, cols, A, lda, ipiv, info));
    } else {
        ROCBLAS_CHECK(
            rocsolver_dgetrf(handle, rows, cols, A, lda, ipiv, info));
    }

    return;
}

template <typename T>
void apply_panel_pivots(rocblas_handle handle,
                        T* A,
                        int lda,
                        int n,
                        int panel,
                        int panel_size,
                        int* pivots)
{
    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    if constexpr (std::is_same_v<T, float>) {
        launch_apply_getrf_panel_pivots(
            A, lda, n, panel, panel_size, pivots, stream);
    } else {
        launch_apply_getrf_panel_pivots_double(
            A, lda, n, panel, panel_size, pivots, stream);
    }

    return;
}

template <typename T>
void trsm_left_lower_unit_recursive(rocblas_handle handle,
                                    int rows,
                                    int cols,
                                    const T* L,
                                    int ldl,
                                    T* B,
                                    int ldb,
                                    int leaf_size)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if (leaf_size <= 0 || rows <= leaf_size) {
        trsm_left_lower_unit(handle, rows, cols, L, ldl, B, ldb);
        return;
    }

    const int top_rows = rows / 2;
    const int bottom_rows = rows - top_rows;

    trsm_left_lower_unit_recursive(
        handle, top_rows, cols, L, ldl, B, ldb, leaf_size);

    gemm_nn_update(
        handle,
        bottom_rows,
        cols,
        top_rows,
        block_ptr(L, ldl, top_rows, 0),
        ldl,
        B,
        ldb,
        B + top_rows,
        ldb);

    trsm_left_lower_unit_recursive(
        handle,
        bottom_rows,
        cols,
        block_ptr(L, ldl, top_rows, top_rows),
        ldl,
        B + top_rows,
        ldb,
        leaf_size);

    return;
}

} // namespace getrf_pivot_internal
