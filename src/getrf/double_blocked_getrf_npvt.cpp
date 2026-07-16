#include "getrf/double_blocked_getrf_npvt.h"

#include <algorithm>

#include "getrf_npvt_internal.h"

namespace {

using getrf_npvt_internal::block_ptr;
using getrf_npvt_internal::factor_panel;
using getrf_npvt_internal::gemm_nn_update;
using getrf_npvt_internal::trsm_left_lower_unit;

template <typename T>
void prepare_next_inner_panel(rocblas_handle handle,
                              T* A,
                              int lda,
                              int n,
                              int outer,
                              int next_inner,
                              int outer_end,
                              int b)
{
    if (next_inner >= outer_end) {
        return;
    }

    const int next_size = std::min(b, outer_end - next_inner);
    const int history_size = next_inner - outer;

    trsm_left_lower_unit(
        handle,
        history_size,
        next_size,
        block_ptr(A, lda, outer, outer),
        lda,
        block_ptr(A, lda, outer, next_inner),
        lda);

    gemm_nn_update(
        handle,
        n - next_inner,
        next_size,
        history_size,
        block_ptr(A, lda, next_inner, outer),
        lda,
        block_ptr(A, lda, outer, next_inner),
        lda,
        block_ptr(A, lda, next_inner, next_inner),
        lda);

    return;
}

template <typename T>
void prepare_next_outer_panel(rocblas_handle handle,
                              T* A,
                              int lda,
                              int n,
                              int next_outer,
                              int nb)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(nb, n - next_outer);

    trsm_left_lower_unit(
        handle,
        next_outer,
        next_size,
        A,
        lda,
        block_ptr(A, lda, 0, next_outer),
        lda);

    gemm_nn_update(
        handle,
        n - next_outer,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, 0, next_outer),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda);

    return;
}

template <typename T>
void double_blocked_getrf_npvt_impl(rocblas_handle handle,
                                    T* A,
                                    int lda,
                                    int n,
                                    int nb,
                                    int b,
                                    int* info)
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
            const int next_inner = inner + inner_size;

            factor_panel(
                handle,
                n - inner,
                inner_size,
                block_ptr(A, lda, inner, inner),
                lda,
                info);

            prepare_next_inner_panel(
                handle,
                A,
                lda,
                n,
                outer,
                next_inner,
                outer_end,
                b);
        }

        prepare_next_outer_panel(handle, A, lda, n, outer_end, nb);
    }

    return;
}

} // namespace

extern "C" {

void double_blocked_getrf_npvt(rocblas_handle handle,
                               float* A,
                               int lda,
                               int n,
                               int nb,
                               int b,
                               int* info)
{
    double_blocked_getrf_npvt_impl(handle, A, lda, n, nb, b, info);

    return;
}

void double_blocked_getrf_npvt_double(rocblas_handle handle,
                                      double* A,
                                      int lda,
                                      int n,
                                      int nb,
                                      int b,
                                      int* info)
{
    double_blocked_getrf_npvt_impl(handle, A, lda, n, nb, b, info);

    return;
}

} // extern "C"
