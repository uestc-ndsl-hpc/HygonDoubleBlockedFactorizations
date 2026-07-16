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
void trsm_left_lower_unit_iterative(rocblas_handle handle,
                                    int rows,
                                    int cols,
                                    const T* L,
                                    int ldl,
                                    T* B,
                                    int ldb,
                                    int block_size)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }
    if (block_size <= 0 || rows <= block_size) {
        trsm_left_lower_unit(handle, rows, cols, L, ldl, B, ldb);
        return;
    }

    for (int row = 0; row < rows; row += block_size) {
        const int diagonal_rows = std::min(block_size, rows - row);
        trsm_left_lower_unit(
            handle,
            diagonal_rows,
            cols,
            block_ptr(L, ldl, row, row),
            ldl,
            B + row,
            ldb);

        const int trailing_rows = rows - row - diagonal_rows;
        if (trailing_rows > 0) {
            gemm_nn_update(
                handle,
                trailing_rows,
                cols,
                diagonal_rows,
                block_ptr(L, ldl, row + diagonal_rows, row),
                ldl,
                B + row,
                ldb,
                B + row + diagonal_rows,
                ldb);
        }
    }

    return;
}

} // namespace getrf_pivot_internal
