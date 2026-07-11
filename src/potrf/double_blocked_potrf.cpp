#include "potrf/double_blocked_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocsolver/rocsolver.h>

#include "trsm/double_blocked_trsm.h"
#include "utils/common/error_check.h"

namespace {

constexpr int kPotrfDbTrsmMicroBlock = 1024;
constexpr int kInnerGemmMTile = 8192;
constexpr int kInnerGemmMThreshold = 32768;
constexpr int kInnerGemmKThreshold = 5632;

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
    int ldb,
    int trsm_b)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (trsm_b <= 0) {
        trsm_b = cols;
    }
    trsm_b = std::min(trsm_b, cols);

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
void rocblas_gemm_nt_update_m_tiled(
    rocblas_handle handle,
    int rows,
    int cols,
    int k,
    const T* A,
    int lda,
    const T* B,
    int ldb,
    T* C,
    int ldc,
    int m_tile)
{
    if (m_tile <= 0 || m_tile >= rows) {
        rocblas_gemm_nt_update(
            handle, rows, cols, k, A, lda, B, ldb, C, ldc);
        return;
    }

    for (int row = 0; row < rows; row += m_tile) {
        const int tile_rows = std::min(m_tile, rows - row);
        rocblas_gemm_nt_update(
            handle,
            tile_rows,
            cols,
            k,
            A + row,
            lda,
            B,
            ldb,
            C + row,
            ldc);
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
    int row_end,
    int outer,
    int panel_start,
    int outer_end,
    int b,
    int gemm_m_tile,
    int gemm_m_threshold,
    int gemm_k_threshold)
{
    if (panel_start >= outer_end || panel_start >= row_end) {
        return;
    }

    const int panel_cols = std::min(b, outer_end - panel_start);
    const int rows = row_end - panel_start;
    const int k = panel_start - outer;

    const bool use_m_tiling =
        gemm_m_tile > 0 && rows > gemm_m_threshold && k >= gemm_k_threshold;

    rocblas_gemm_nt_update_m_tiled(
        handle,
        rows,
        panel_cols,
        k,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, outer),
        lda,
        block_ptr(A, lda, panel_start, panel_start),
        lda,
        use_m_tiling ? gemm_m_tile : 0);

    return;
}

template <typename T>
void update_next_outer_diagonal(
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

    rocblas_syrk_lower_update_kblocked(
        handle,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda,
        syrk_k_tile);

    return;
}

template <typename T>
void update_next_outer_panel_below(
    rocblas_handle handle,
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
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;

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
void prepare_next_outer_panel(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int next_outer,
    int nb,
    int syrk_k_tile)
{
    update_next_outer_diagonal(
        handle, A, lda, n, next_outer, nb, syrk_k_tile);
    update_next_outer_panel_below(
        handle, A, lda, n, next_outer, nb);

    return;
}

template <typename T>
void factor_outer_diagonal_block(
    rocblas_handle handle,
    T* A,
    int lda,
    int outer,
    int outer_end,
    int b,
    int* info)
{
    for (int inner = outer; inner < outer_end; inner += b) {
        const int inner_size = std::min(b, outer_end - inner);
        const int panel_start = inner + inner_size;

        T* A_ii = block_ptr(A, lda, inner, inner);
        rocsolver_potrf(handle, inner_size, A_ii, lda, info);

        rocblas_trsm_right_lower_transpose(
            handle,
            outer_end - panel_start,
            inner_size,
            A_ii,
            lda,
            block_ptr(A, lda, panel_start, inner),
            lda);

        update_next_inner_panel(
            handle,
            A,
            lda,
            outer_end,
            outer,
            panel_start,
            outer_end,
            b,
            0,
            0,
            0);
    }

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
    int syrk_k_tile,
    int inner_gemm_m_tile,
    int inner_gemm_m_threshold,
    int inner_gemm_k_threshold,
    bool defer_trailing_trsm,
    int deferred_trsm_b)
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
            const int inner_row_end = defer_trailing_trsm ? outer_end : n;
            const int trailing_rows = inner_row_end - panel_start;

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
                    lda,
                    kPotrfDbTrsmMicroBlock);
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
                inner_row_end,
                outer,
                panel_start,
                outer_end,
                b,
                inner_gemm_m_tile,
                inner_gemm_m_threshold,
                inner_gemm_k_threshold);
        }

        // The inner loop factors only L_oo; solve the untouched panel below it
        // once the complete outer diagonal block is available.
        if (defer_trailing_trsm) {
            double_blocked_trsm_right_lower_transpose(
                handle,
                n - outer_end,
                outer_size,
                block_ptr(A, lda, outer, outer),
                lda,
                block_ptr(A, lda, outer_end, outer),
                lda,
                deferred_trsm_b);
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

template <typename T>
void double_blocked_potrf_lookahead_impl(
    rocblas_handle factor_handle,
    rocblas_handle update_handle,
    T* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int trsm_b,
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

    hipStream_t factor_stream = nullptr;
    hipStream_t update_stream = nullptr;
    ROCBLAS_CHECK(rocblas_get_stream(factor_handle, &factor_stream));
    ROCBLAS_CHECK(rocblas_get_stream(update_handle, &update_stream));
    ROCBLAS_CHECK(
        rocblas_set_pointer_mode(update_handle, rocblas_pointer_mode_host));

    hipEvent_t diagonal_ready;
    hipEvent_t panel_ready;
    HIP_CHECK(hipEventCreateWithFlags(&diagonal_ready, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&panel_ready, hipEventDisableTiming));

    bool panel_update_pending = false;

    for (int outer = 0; outer < n; outer += nb) {
        const int outer_size = std::min(nb, n - outer);
        const int outer_end = outer + outer_size;

        factor_outer_diagonal_block(
            factor_handle, A, lda, outer, outer_end, b, info);

        // The local diagonal factorization can overlap the previous outer
        // GEMM. The panel below is consumed only by the deferred TRSM.
        if (panel_update_pending) {
            HIP_CHECK(hipStreamWaitEvent(factor_stream, panel_ready, 0));
            panel_update_pending = false;
        }

        double_blocked_trsm_right_lower_transpose(
            factor_handle,
            n - outer_end,
            outer_size,
            block_ptr(A, lda, outer, outer),
            lda,
            block_ptr(A, lda, outer_end, outer),
            lda,
            trsm_b);

        if (outer_end >= n) {
            continue;
        }

        update_next_outer_diagonal(
            factor_handle, A, lda, n, outer_end, nb, syrk_k_tile);

        HIP_CHECK(hipEventRecord(diagonal_ready, factor_stream));
        HIP_CHECK(hipStreamWaitEvent(update_stream, diagonal_ready, 0));

        // The next iteration factors the ready diagonal block on the main
        // stream while this update prepares the panel immediately below it.
        update_next_outer_panel_below(
            update_handle, A, lda, n, outer_end, nb);
        HIP_CHECK(hipEventRecord(panel_ready, update_stream));

        panel_update_pending = true;
    }

    HIP_CHECK(hipStreamSynchronize(factor_stream));
    HIP_CHECK(hipEventDestroy(diagonal_ready));
    HIP_CHECK(hipEventDestroy(panel_ready));

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
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, 0, 0, 0, 0, false, 0);

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
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, true, 0, 0, 0, 0, false, 0);

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
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        0,
        0,
        0,
        false,
        0);

    return;
}

void double_blocked_potrf_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        false,
        0);

    return;
}

void double_blocked_potrf_deferred_trsm(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int trsm_b,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        0,
        0,
        0,
        true,
        trsm_b);

    return;
}

void double_blocked_potrf_lookahead(
    rocblas_handle factor_handle,
    rocblas_handle update_handle,
    float* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int trsm_b,
    int* info)
{
    double_blocked_potrf_lookahead_impl(
        factor_handle,
        update_handle,
        A,
        lda,
        n,
        nb,
        b,
        syrk_k_tile,
        trsm_b,
        info);

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
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, false, 0, 0, 0, 0, false, 0);

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
    double_blocked_potrf_impl(
        handle, A, lda, n, nb, b, info, true, 0, 0, 0, 0, false, 0);

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
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        0,
        0,
        0,
        false,
        0);

    return;
}

void double_blocked_potrf_double_kblocked_syrk_adaptive_gemm(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        kInnerGemmMTile,
        kInnerGemmMThreshold,
        kInnerGemmKThreshold,
        false,
        0);

    return;
}

void double_blocked_potrf_double_deferred_trsm(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int trsm_b,
    int* info)
{
    double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        nb,
        b,
        info,
        false,
        syrk_k_tile,
        0,
        0,
        0,
        true,
        trsm_b);

    return;
}

void double_blocked_potrf_double_lookahead(
    rocblas_handle factor_handle,
    rocblas_handle update_handle,
    double* A,
    int lda,
    int n,
    int nb,
    int b,
    int syrk_k_tile,
    int trsm_b,
    int* info)
{
    double_blocked_potrf_lookahead_impl(
        factor_handle,
        update_handle,
        A,
        lda,
        n,
        nb,
        b,
        syrk_k_tile,
        trsm_b,
        info);

    return;
}

} // extern "C"
