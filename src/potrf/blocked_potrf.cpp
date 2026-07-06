#include "potrf/blocked_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"

namespace {

template <typename T>
void rocsolver_potrf(rocblas_handle handle, int n, T* A, int lda, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            A,
            lda,
            info));
    }
}

template <typename T>
void rocblas_trsm_right_lower_transpose(
    rocblas_handle handle,
    int rows,
    int cols,
    const T* A,
    int lda,
    T* B,
    int ldb)
{
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    } else {
        ROCBLAS_CHECK(rocblas_dtrsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            rows,
            cols,
            &one,
            A,
            lda,
            B,
            ldb));
    }
}

template <typename T>
void rocblas_syrk_lower_update(
    rocblas_handle handle,
    int n,
    int k,
    const T* A,
    int lda,
    T* C,
    int ldc)
{
    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_ssyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dsyrk(
            handle,
            rocblas_fill_lower,
            rocblas_operation_none,
            n,
            k,
            &neg_one,
            A,
            lda,
            &one,
            C,
            ldc));
    }
}

template <typename T>
void blocked_right_looking_potrf_impl(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int nb,
    int* info)
{
    if (n <= 0) {
        return;
    }

    if (nb <= 0) {
        nb = n;
    }

    for (int k = 0; k < n; k += nb) {
        const int panel_size = std::min(nb, n - k);
        const int trailing_size = n - k - panel_size;

        T* A_kk = A + static_cast<size_t>(k) * static_cast<size_t>(lda) + k;

        rocsolver_potrf(handle, panel_size, A_kk, lda, info);

        if (trailing_size <= 0) {
            continue;
        }

        T* A_21 = A + static_cast<size_t>(k) * static_cast<size_t>(lda) + k + panel_size;
        T* A_22 = A + static_cast<size_t>(k + panel_size) * static_cast<size_t>(lda) + k + panel_size;

        rocblas_trsm_right_lower_transpose(
            handle,
            trailing_size,
            panel_size,
            A_kk,
            lda,
            A_21,
            lda);

        rocblas_syrk_lower_update(
            handle,
            trailing_size,
            panel_size,
            A_21,
            lda,
            A_22,
            lda);
    }
}

} // namespace

extern "C" {

void blocked_right_looking_potrf(rocblas_handle handle,
                                 float* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int* info)
{
    blocked_right_looking_potrf_impl(handle, A, lda, n, nb, info);
}

void blocked_right_looking_potrf_double(rocblas_handle handle,
                                        double* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int* info)
{
    blocked_right_looking_potrf_impl(handle, A, lda, n, nb, info);
}

} // extern "C"
