#include "getrf/double_blocked_getrf.h"

#include <algorithm>
#include <iostream>

#include "getrf_batched_pivot_internal.h"
#include "getrf_pivot_internal.h"
#include "utils/common/error_check.h"

namespace {

using getrf_pivot_internal::apply_panel_pivots;
using getrf_pivot_internal::block_ptr;
using getrf_pivot_internal::factor_panel;
using getrf_pivot_internal::gemm_nn_update;
using getrf_pivot_internal::trsm_left_lower_unit;
using getrf_pivot_internal::trsm_left_lower_unit_iterative;

void mark_getrf_failure(rocblas_handle handle, int* info)
{
    if (info == nullptr) {
        return;
    }

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));
    HIP_CHECK(hipMemsetAsync(info, 0xff, sizeof(int), stream));

    return;
}

template <typename T>
void prepare_next_inner_panel(rocblas_handle handle,
                              T* A,
                              int lda,
                              int n,
                              int outer,
                              int next_inner,
                              int outer_end,
                              int inner_nb)
{
    if (next_inner >= outer_end) {
        return;
    }

    const int next_size = std::min(inner_nb, outer_end - next_inner);
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
                              int outer_nb,
                              getrf_outer_trsm_mode outer_trsm_mode,
                              int trsm_block)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(outer_nb, n - next_outer);

    if (outer_trsm_mode == getrf_outer_trsm_direct) {
        trsm_left_lower_unit(
            handle,
            next_outer,
            next_size,
            A,
            lda,
            block_ptr(A, lda, 0, next_outer),
            lda);
    } else {
        trsm_left_lower_unit_iterative(
            handle,
            next_outer,
            next_size,
            A,
            lda,
            block_ptr(A, lda, 0, next_outer),
            lda,
            trsm_block);
    }

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
void double_blocked_getrf_impl(rocblas_handle handle,
                               T* A,
                               int lda,
                               int n,
                               int outer_nb,
                               int inner_nb,
                               getrf_outer_trsm_mode outer_trsm_mode,
                               int trsm_block,
                               int* ipiv,
                               int* info)
{
    if (n <= 0) {
        return;
    }
    if (outer_nb <= 0) {
        outer_nb = n;
    }
    if (inner_nb <= 0) {
        inner_nb = outer_nb;
    }
    inner_nb = std::min(inner_nb, outer_nb);

    for (int outer = 0; outer < n; outer += outer_nb) {
        const int outer_size = std::min(outer_nb, n - outer);
        const int outer_end = outer + outer_size;

        for (int inner = outer; inner < outer_end; inner += inner_nb) {
            const int inner_size = std::min(inner_nb, outer_end - inner);
            const int next_inner = inner + inner_size;

            factor_panel(
                handle,
                n - inner,
                inner_size,
                block_ptr(A, lda, inner, inner),
                lda,
                ipiv + inner,
                info);

            apply_panel_pivots(
                handle, A, lda, n, inner, inner_size, ipiv + inner);

            prepare_next_inner_panel(
                handle,
                A,
                lda,
                n,
                outer,
                next_inner,
                outer_end,
                inner_nb);
        }

        prepare_next_outer_panel(
            handle,
            A,
            lda,
            n,
            outer_end,
            outer_nb,
            outer_trsm_mode,
            trsm_block);
    }

    return;
}

} // namespace

extern "C" {

void double_blocked_getrf(rocblas_handle handle,
                          float* A,
                          int lda,
                          int n,
                          int outer_nb,
                          int inner_nb,
                          getrf_outer_trsm_mode outer_trsm_mode,
                          int trsm_block,
                          int* ipiv,
                          int* info)
{
    double_blocked_getrf_impl(
        handle,
        A,
        lda,
        n,
        outer_nb,
        inner_nb,
        outer_trsm_mode,
        trsm_block,
        ipiv,
        info);

    return;
}

void double_blocked_getrf_double(rocblas_handle handle,
                                 double* A,
                                 int lda,
                                 int n,
                                 int outer_nb,
                                 int inner_nb,
                                 getrf_outer_trsm_mode outer_trsm_mode,
                                 int trsm_block,
                                 int* ipiv,
                                 int* info)
{
    double_blocked_getrf_impl(
        handle,
        A,
        lda,
        n,
        outer_nb,
        inner_nb,
        outer_trsm_mode,
        trsm_block,
        ipiv,
        info);

    return;
}

void double_blocked_getrf_batched_pivot(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_nb,
    int inner_nb,
    getrf_outer_trsm_mode outer_trsm_mode,
    int trsm_block,
    int inner_gemm_m_tile,
    int* ipiv,
    int* info,
    getrf_batched_pivot_workspace* workspace)
{
    if (!getrf_batched_pivot_internal::workspace_is_valid(workspace, n)) {
        std::cerr << "Invalid batched-pivot GETRF workspace\n";
        mark_getrf_failure(handle, info);
        return;
    }

    getrf_batched_pivot_internal::NullObserver observer;
    if (!getrf_batched_pivot_internal::double_blocked_getrf(
            handle,
            observer,
            A,
            lda,
            n,
            outer_nb,
            inner_nb,
            outer_trsm_mode,
            trsm_block,
            inner_gemm_m_tile,
            ipiv,
            info,
            *workspace)) {
        std::cerr << "Invalid pivot returned by panel GETRF\n";
        mark_getrf_failure(handle, info);
    }

    return;
}

void double_blocked_getrf_double_batched_pivot(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int outer_nb,
    int inner_nb,
    getrf_outer_trsm_mode outer_trsm_mode,
    int trsm_block,
    int inner_gemm_m_tile,
    int* ipiv,
    int* info,
    getrf_batched_pivot_workspace* workspace)
{
    if (!getrf_batched_pivot_internal::workspace_is_valid(workspace, n)) {
        std::cerr << "Invalid batched-pivot GETRF workspace\n";
        mark_getrf_failure(handle, info);
        return;
    }

    getrf_batched_pivot_internal::NullObserver observer;
    if (!getrf_batched_pivot_internal::double_blocked_getrf(
            handle,
            observer,
            A,
            lda,
            n,
            outer_nb,
            inner_nb,
            outer_trsm_mode,
            trsm_block,
            inner_gemm_m_tile,
            ipiv,
            info,
            *workspace)) {
        std::cerr << "Invalid pivot returned by panel GETRF\n";
        mark_getrf_failure(handle, info);
    }

    return;
}

} // extern "C"
