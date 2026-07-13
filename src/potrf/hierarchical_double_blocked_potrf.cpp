#include "potrf/hierarchical_double_blocked_potrf.h"

#include <algorithm>
#include <cstddef>

#include "gemm/update_gemm.h"
#include "potrf/blocked_potrf.h"
#include "syrk/kblocked_syrk.h"
#include "trsm/recursive_trsm.h"

namespace {

constexpr int kGemmMThreshold = 32768;

float* block_ptr(float* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

void update_panel_from_current_outer(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer,
    int panel_start,
    int panel_size,
    int syrk_k_tile,
    int gemm_m_tile)
{
    const int k = panel_start - outer;
    if (k <= 0 || panel_size <= 0) {
        return;
    }

    const float* panel_history = block_ptr(A, lda, panel_start, outer);
    syrk_lower_update_kblocked(
        handle,
        panel_size,
        k,
        panel_history,
        lda,
        block_ptr(A, lda, panel_start, panel_start),
        lda,
        syrk_k_tile);

    const int below_start = panel_start + panel_size;
    const int below_rows = n - below_start;
    if (below_rows <= 0) {
        return;
    }

    const int effective_m_tile =
        below_rows > kGemmMThreshold ? gemm_m_tile : 0;

    gemm_nt_update_m_tiled(
        handle,
        below_rows,
        panel_size,
        k,
        block_ptr(A, lda, below_start, outer),
        lda,
        panel_history,
        lda,
        block_ptr(A, lda, below_start, panel_start),
        lda,
        effective_m_tile);

    return;
}

void factor_and_solve_panel(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int panel_start,
    int panel_size,
    int potrf_leaf_size,
    int* info)
{
    float* diagonal = block_ptr(A, lda, panel_start, panel_start);

    blocked_right_looking_potrf(
        handle,
        diagonal,
        lda,
        panel_size,
        potrf_leaf_size,
        info);

    const int below_start = panel_start + panel_size;
    const int below_rows = n - below_start;
    if (below_rows <= 0) {
        return;
    }

    recursive_trsm_tiled_update(
        handle,
        diagonal,
        lda,
        block_ptr(A, lda, below_start, panel_start),
        lda,
        below_rows,
        panel_size,
        potrf_leaf_size);

    return;
}

void prepare_next_outer_panel(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int next_outer,
    int outer_block_size,
    int syrk_k_tile,
    int gemm_m_tile)
{
    if (next_outer >= n) {
        return;
    }

    const int next_size = std::min(outer_block_size, n - next_outer);
    const int below_start = next_outer + next_size;
    const int below_rows = n - below_start;
    const int effective_m_tile =
        below_rows > kGemmMThreshold ? gemm_m_tile : 0;

    syrk_lower_update_kblocked(
        handle,
        next_size,
        next_outer,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, next_outer, next_outer),
        lda,
        syrk_k_tile);

    gemm_nt_update_m_tiled(
        handle,
        below_rows,
        next_size,
        next_outer,
        block_ptr(A, lda, below_start, 0),
        lda,
        block_ptr(A, lda, next_outer, 0),
        lda,
        block_ptr(A, lda, below_start, next_outer),
        lda,
        effective_m_tile);

    return;
}

} // namespace

extern "C" {

void hierarchical_double_blocked_potrf(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int panel_block_size,
    int potrf_leaf_size,
    int syrk_k_tile,
    int gemm_m_tile,
    int* info)
{
    if (n <= 0) {
        return;
    }

    if (outer_block_size <= 0) {
        outer_block_size = n;
    }
    if (panel_block_size <= 0) {
        panel_block_size = outer_block_size;
    }

    panel_block_size = std::min(panel_block_size, outer_block_size);

    if (potrf_leaf_size <= 0) {
        potrf_leaf_size = panel_block_size;
    }
    potrf_leaf_size = std::min(potrf_leaf_size, panel_block_size);

    for (int outer = 0; outer < n; outer += outer_block_size) {
        const int outer_size = std::min(outer_block_size, n - outer);
        const int outer_end = outer + outer_size;

        for (int panel = outer; panel < outer_end; panel += panel_block_size) {
            const int panel_size =
                std::min(panel_block_size, outer_end - panel);

            update_panel_from_current_outer(
                handle,
                A,
                lda,
                n,
                outer,
                panel,
                panel_size,
                syrk_k_tile,
                gemm_m_tile);

            factor_and_solve_panel(
                handle,
                A,
                lda,
                n,
                panel,
                panel_size,
                potrf_leaf_size,
                info);
        }

        prepare_next_outer_panel(
            handle,
            A,
            lda,
            n,
            outer_end,
            outer_block_size,
            syrk_k_tile,
            gemm_m_tile);
    }

    return;
}

} // extern "C"
