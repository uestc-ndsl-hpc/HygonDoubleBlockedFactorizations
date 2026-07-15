#pragma once

#include <cstddef>
#include <type_traits>

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"

namespace getrf_npvt_internal {

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void factor_panel(rocblas_handle handle,
                  int rows,
                  int cols,
                  T* A,
                  int lda,
                  int* info)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(
            rocsolver_sgetrf_npvt(handle, rows, cols, A, lda, info));
    } else {
        ROCBLAS_CHECK(
            rocsolver_dgetrf_npvt(handle, rows, cols, A, lda, info));
    }

    return;
}

template <typename T>
void trsm_left_lower_unit(rocblas_handle handle,
                          int rows,
                          int cols,
                          const T* L,
                          int ldl,
                          T* B,
                          int ldb)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_left,
            rocblas_fill_lower,
            rocblas_operation_none,
            rocblas_diagonal_unit,
            rows,
            cols,
            &one,
            L,
            ldl,
            B,
            ldb));
    } else {
        ROCBLAS_CHECK(rocblas_dtrsm(
            handle,
            rocblas_side_left,
            rocblas_fill_lower,
            rocblas_operation_none,
            rocblas_diagonal_unit,
            rows,
            cols,
            &one,
            L,
            ldl,
            B,
            ldb));
    }

    return;
}

template <typename T>
void gemm_nn_update(rocblas_handle handle,
                    int rows,
                    int cols,
                    int k,
                    const T* L,
                    int ldl,
                    const T* U,
                    int ldu,
                    T* C,
                    int ldc)
{
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            rows,
            cols,
            k,
            &neg_one,
            L,
            ldl,
            U,
            ldu,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_none,
            rows,
            cols,
            k,
            &neg_one,
            L,
            ldl,
            U,
            ldu,
            &one,
            C,
            ldc));
    }

    return;
}

} // namespace getrf_npvt_internal
