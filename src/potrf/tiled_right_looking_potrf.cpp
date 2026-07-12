#include "potrf/tiled_right_looking_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "potrf/blocked_potrf.h"
#include "syrk/tiled_lower_update.h"
#include "trsm/recursive_trsm.h"

namespace {

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void dispatch_blocked_potrf(rocblas_handle handle,
                            T* A,
                            int lda,
                            int n,
                            int leaf_size,
                            int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        blocked_right_looking_potrf(handle, A, lda, n, leaf_size, info);
    } else {
        blocked_right_looking_potrf_double(
            handle, A, lda, n, leaf_size, info);
    }

    return;
}

template <typename T>
void dispatch_recursive_trsm(rocblas_handle handle,
                             T* A,
                             int lda,
                             T* B,
                             int ldb,
                             int rows,
                             int cols,
                             int leaf_size)
{
    if constexpr (std::is_same_v<T, float>) {
        recursive_trsm_tiled_update(
            handle, A, lda, B, ldb, rows, cols, leaf_size);
    } else {
        recursive_trsm_double(
            handle, A, lda, B, ldb, rows, cols, leaf_size);
    }

    return;
}

template <typename T>
void dispatch_tiled_lower_update(rocblas_handle handle,
                                 int n,
                                 int k,
                                 const T* A,
                                 int lda,
                                 T* C,
                                 int ldc,
                                 int n_tile,
                                 int k_tile)
{
    if constexpr (std::is_same_v<T, float>) {
        tiled_lower_rank_k_update(
            handle, n, k, A, lda, C, ldc, n_tile, k_tile);
    } else {
        tiled_lower_rank_k_update_double(
            handle, n, k, A, lda, C, ldc, n_tile, k_tile);
    }

    return;
}

template <typename T>
void tiled_right_looking_double_blocked_potrf_impl(
    rocblas_handle handle,
    T* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info)
{
    if (n <= 0) {
        return;
    }

    if (outer_block_size <= 0) {
        outer_block_size = n;
    }
    if (update_tile_size <= 0) {
        update_tile_size = outer_block_size;
    }
    if (potrf_leaf_size <= 0) {
        potrf_leaf_size = outer_block_size;
    }
    if (trsm_leaf_size <= 0) {
        trsm_leaf_size = potrf_leaf_size;
    }
    if (update_k_tile <= 0) {
        update_k_tile = outer_block_size;
    }

    for (int outer = 0; outer < n; outer += outer_block_size) {
        const int outer_size = std::min(outer_block_size, n - outer);
        const int trailing_start = outer + outer_size;
        const int trailing_size = n - trailing_start;
        T* diagonal = block_ptr(A, lda, outer, outer);

        dispatch_blocked_potrf(
            handle,
            diagonal,
            lda,
            outer_size,
            std::min(potrf_leaf_size, outer_size),
            info);

        if (trailing_size <= 0) {
            continue;
        }

        T* panel = block_ptr(A, lda, trailing_start, outer);

        dispatch_recursive_trsm(
            handle,
            diagonal,
            lda,
            panel,
            lda,
            trailing_size,
            outer_size,
            std::min(trsm_leaf_size, outer_size));

        dispatch_tiled_lower_update(
            handle,
            trailing_size,
            outer_size,
            panel,
            lda,
            block_ptr(A, lda, trailing_start, trailing_start),
            lda,
            update_tile_size,
            update_k_tile);
    }

    return;
}

} // namespace

extern "C" {

void tiled_right_looking_double_blocked_potrf(
    rocblas_handle handle,
    float* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info)
{
    tiled_right_looking_double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        outer_block_size,
        update_tile_size,
        potrf_leaf_size,
        trsm_leaf_size,
        update_k_tile,
        info);

    return;
}

void tiled_right_looking_double_blocked_potrf_double(
    rocblas_handle handle,
    double* A,
    int lda,
    int n,
    int outer_block_size,
    int update_tile_size,
    int potrf_leaf_size,
    int trsm_leaf_size,
    int update_k_tile,
    int* info)
{
    tiled_right_looking_double_blocked_potrf_impl(
        handle,
        A,
        lda,
        n,
        outer_block_size,
        update_tile_size,
        potrf_leaf_size,
        trsm_leaf_size,
        update_k_tile,
        info);

    return;
}

} // extern "C"
