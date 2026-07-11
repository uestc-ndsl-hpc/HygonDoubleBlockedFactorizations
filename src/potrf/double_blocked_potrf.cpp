#include "potrf/double_blocked_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <rocsolver/rocsolver.h>

#include "trsm/double_blocked_trsm.h"
#include "utils/common/error_check.h"

namespace {

constexpr int kPotrfDbTrsmMicroBlock = 1024;

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
const T* block_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

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

    return;
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
    if (rows <= 0 || cols <= 0) {
        return;
    }

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

    return;
}

template <typename T>
void double_blocked_trsm_right_lower_transpose(
    rocblas_handle handle,
    int rows,
    int cols,
    const T* A,
    int lda,
    T* B,
    int ldb)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int trsm_b = std::min(kPotrfDbTrsmMicroBlock, cols);

    if constexpr (std::is_same_v<T, float>) {
        double_blocked_trsm(
            handle,
            const_cast<T*>(A),
            lda,
            B,
            ldb,
            rows,
            cols,
            cols,
            trsm_b);
    } else {
        double_blocked_trsm_double(
            handle,
            const_cast<T*>(A),
            lda,
            B,
            ldb,
            rows,
            cols,
            cols,
            trsm_b);
    }

    return;
}

template <typename T>
void rocblas_gemm_nt_update(
    rocblas_handle handle,
    int rows,
    int cols,
    int k,
    const T* A,
    int lda,
    const T* B,
    int ldb,
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
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            rows,
            cols,
            k,
            &neg_one,
            A,
            lda,
            B,
            ldb,
            &one,
            C,
            ldc));
    }

    return;
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
    if (n <= 0 || k <= 0) {
        return;
    }

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

    return;
}

template <typename T>
void rocblas_syrk_lower_update_kblocked(
    rocblas_handle handle,
    int n,
    int k,
    const T* A,
    int lda,
    T* C,
    int ldc,
    int k_tile)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    if (k_tile <= 0 || k_tile >= k) {
        rocblas_syrk_lower_update(handle, n, k, A, lda, C, ldc);
        return;
    }

    for (int k_offset = 0; k_offset < k; k_offset += k_tile) {
        const int chunk = std::min(k_tile, k - k_offset);
        rocblas_syrk_lower_update(
            handle,
            n,
            chunk,
            block_ptr(A, lda, 0, k_offset),
            lda,
            C,
            ldc);
    }
}

template <typename T>
void update_next_inner_panel(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int outer,
    int panel_start,
    int outer_end,
    int b)
{
    if (panel_start >= outer_end) {
        return;
    }

    const int panel_cols = std::min(b, outer_end - panel_start);
    const int rows = n - panel_start;
    const int k = panel_start - outer;

    rocblas_gemm_nt_update(
        handle,
        rows,
        panel_cols,
        k,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, panel_start),
        lda);

    return;
}

template <typename T>
void prepare_next_outer_panel(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int next_outer,
    int nb,
    int syrk_k_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

    rocblas_syrk_lower_update_kblocked(
        handle,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda,
        syrk_k_tile);

    rocblas_gemm_nt_update(
        handle,
        below_rows,
        next_size,
        next_outer,
        block_ptr(A, lda, below_start, 0),
        lda,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, below_start, next_outer),
        lda);

    return;
}

template <typename T>
void double_blocked_potrf_impl(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int nb,
    int b,
    int* info,
    bool use_double_blocked_trsm,
    int syrk_k_tile)
{
    if (n <= 0) {
        return;
    }

    if (nb <= 0) {
        nb = n;
    }

    if (b <= 0) {
        b = nb;
    }

    b = std::min(b, nb);

    for (int outer = 0; outer < n; outer += nb) {
        const int outer_size = std::min(nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += b) {
            const int inner_size = std::min(b, outer_end - inner);
            const int panel_start = inner + inner_size;
            const int trailing_rows = n - panel_start;

            T* A_ii = block_ptr(A, lda, inner, inner);

            rocsolver_potrf(handle, inner_size, A_ii, lda, info);

            if (use_double_blocked_trsm) {
                double_blocked_trsm_right_lower_transpose(
                    handle,
                    trailing_rows,
                    inner_size,
                    A_ii,
                    lda,
                    block_ptr(A, lda, panel_start, inner),
                    lda);
            } else {
                rocblas_trsm_right_lower_transpose(
                    handle,
                    trailing_rows,
                    inner_size,
                    A_ii,
                    lda,
                    block_ptr(A, lda, panel_start, inner),
                    lda);
            }

            update_next_inner_panel(
                handle,
                A,
                lda,
                n,
                outer,
                panel_start,
                outer_end,
                b);
        }

        prepare_next_outer_panel(
            handle,
            A,
            lda,
            n,
            outer_end,
            nb,
            syrk_k_tile);
    }

    return;
}

} // namespace

extern "C" {

void double_blocked_potrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int nb,
                          int b,
                          int* info)
{
    double_blocked_potrf_impl(handle, A, lda, n, nb, b, info, false, 0);

    return;
}

void double_blocked_potrf_dbtrsm(rocblas_handle handle,
                                 float* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info)
{
    double_blocked_potrf_impl(handle, A, lda, n, nb, b, info, true, 0);

    return;
}

void double_blocked_potrf_kblocked_syrk(rocblas_handle handle,
                                        float* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int syrk_k_tile,
                                        int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, syrk_k_tile);

    return;
}

void double_blocked_potrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int nb,
                                 int b,
                                 int* info)
{
    double_blocked_potrf_impl(handle, A, lda, n, nb, b, info, false, 0);

    return;
}

void double_blocked_potrf_double_dbtrsm(rocblas_handle handle,
                                        double* A,
                                        int lda,
                                        int n,
                                        int nb,
                                        int b,
                                        int* info)
{
    double_blocked_potrf_impl(handle, A, lda, n, nb, b, info, true, 0);

    return;
}

void double_blocked_potrf_double_kblocked_syrk(rocblas_handle handle,
                                               double* A,
                                               int lda,
                                               int n,
                                               int nb,
                                               int b,
                                               int syrk_k_tile,
                                               int* info)
{
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, syrk_k_tile);

    return;
}

} // extern "C"
