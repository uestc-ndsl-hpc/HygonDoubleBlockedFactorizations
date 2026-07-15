#include "getrf/blocked_getrf_npvt.h"

#include <algorithm>

#include "getrf_npvt_internal.h"

namespace {

using getrf_npvt_internal::block_ptr;
using getrf_npvt_internal::factor_panel;
using getrf_npvt_internal::gemm_nn_update;
using getrf_npvt_internal::trsm_left_lower_unit;

template <typename T>
void blocked_right_looking_getrf_npvt_impl(rocblas_handle handle,
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

    for (int panel = 0; panel < n; panel += nb) {
        const int panel_size = std::min(nb, n - panel);
        const int next = panel + panel_size;
        const int trailing = n - next;

        factor_panel(
            handle,
            n - panel,
            panel_size,
            block_ptr(A, lda, panel, panel),
            lda,
            info);

        trsm_left_lower_unit(
            handle,
            panel_size,
            trailing,
            block_ptr(A, lda, panel, panel),
            lda,
            block_ptr(A, lda, panel, next),
            lda);

        gemm_nn_update(
            handle,
            trailing,
            trailing,
            panel_size,
            block_ptr(A, lda, next, panel),
            lda,
            block_ptr(A, lda, panel, next),
            lda,
            block_ptr(A, lda, next, next),
            lda);
    }

    return;
}

} // namespace

extern "C" {

void blocked_right_looking_getrf_npvt(rocblas_handle handle,
                                      float* A,
                                      int lda,
                                      int n,
                                      int nb,
                                      int* info)
{
    blocked_right_looking_getrf_npvt_impl(handle, A, lda, n, nb, info);

    return;
}

void blocked_right_looking_getrf_npvt_double(rocblas_handle handle,
                                             double* A,
                                             int lda,
                                             int n,
                                             int nb,
                                             int* info)
{
    blocked_right_looking_getrf_npvt_impl(handle, A, lda, n, nb, info);

    return;
}

} // extern "C"
