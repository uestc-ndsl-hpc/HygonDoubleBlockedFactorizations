#include "trsm/double_blocked_trsm.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "utils/common/error_check.h"

namespace {

template <typename T>
T* matrix_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
const T* matrix_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void rocblas_trsm_right_lower_transpose(rocblas_handle handle,
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
void rocblas_gemm_nt_update(rocblas_handle handle,
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
void update_next_inner_panel(rocblas_handle handle,
                             T* A,
                             int lda,
                             T* B,
                             int ldb,
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
    const int solved_cols = panel_start - outer;

    rocblas_gemm_nt_update(
        handle,
        n,
        panel_cols,
        solved_cols,
        matrix_ptr(B, ldb, 0, outer),
        ldb,
        matrix_ptr(A, lda, panel_start, outer),
        lda,
        matrix_ptr(B, ldb, 0, panel_start),
        ldb);

    return;
}

template <typename T>
void double_blocked_trsm_impl(rocblas_handle handle,
                              T* A,
                              int lda,
                              T* B,
                              int ldb,
                              int n,
                              int m,
                              int nb,
                              int b)
{
    if (n <= 0 || m <= 0) {
        return;
    }

    if (nb <= 0) {
        nb = m;
    }

    if (b <= 0) {
        b = nb;
    }

    b = std::min(b, nb);

    for (int outer = 0; outer < m; outer += nb) {
        const int outer_size = std::min(nb, m - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += b) {
            const int inner_size = std::min(b, outer_end - inner);
            const int panel_start = inner + inner_size;

            rocblas_trsm_right_lower_transpose(
                handle,
                n,
                inner_size,
                matrix_ptr(A, lda, inner, inner),
                lda,
                matrix_ptr(B, ldb, 0, inner),
                ldb);

            update_next_inner_panel(
                handle,
                A,
                lda,
                B,
                ldb,
                n,
                outer,
                panel_start,
                outer_end,
                b);
        }

        const int trailing_cols = m - outer_end;
        if (trailing_cols > 0) {
            rocblas_gemm_nt_update(
                handle,
                n,
                trailing_cols,
                outer_size,
                matrix_ptr(B, ldb, 0, outer),
                ldb,
                matrix_ptr(A, lda, outer_end, outer),
                lda,
                matrix_ptr(B, ldb, 0, outer_end),
                ldb);
        }
    }

    return;
}

} // namespace

extern "C" {

void double_blocked_trsm(rocblas_handle handle,
                         float* A,
                         int lda,
                         float* B,
                         int ldb,
                         int n,
                         int m,
                         int nb,
                         int b)
{
    double_blocked_trsm_impl(handle, A, lda, B, ldb, n, m, nb, b);

    return;
}

void double_blocked_trsm_double(rocblas_handle handle,
                                double* A,
                                int lda,
                                double* B,
                                int ldb,
                                int n,
                                int m,
                                int nb,
                                int b)
{
    double_blocked_trsm_impl(handle, A, lda, B, ldb, n, m, nb, b);

    return;
}

} // extern "C"
