#include "syrk/tiled_lower_update.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "gemm/update_gemm.h"
#include "syrk/kblocked_syrk.h"

namespace {

template <typename T>
const T* block_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void dispatch_syrk(rocblas_handle handle,
                   int n,
                   int k,
                   const T* A,
                   int lda,
                   T* C,
                   int ldc)
{
    if constexpr (std::is_same_v<T, float>) {
        syrk_lower_update(handle, n, k, A, lda, C, ldc);
    } else {
        syrk_lower_update_double(handle, n, k, A, lda, C, ldc);
    }

    return;
}

template <typename T>
void dispatch_gemm(rocblas_handle handle,
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
    if constexpr (std::is_same_v<T, float>) {
        gemm_nt_update(handle, rows, cols, k, A, lda, B, ldb, C, ldc);
    } else {
        gemm_nt_update_double(
            handle, rows, cols, k, A, lda, B, ldb, C, ldc);
    }

    return;
}

template <typename T>
void tiled_lower_rank_k_update_impl(rocblas_handle handle,
                                    int n,
                                    int k,
                                    const T* A,
                                    int lda,
                                    T* C,
                                    int ldc,
                                    int n_tile,
                                    int k_tile)
{
    if (n <= 0 || k <= 0) {
        return;
    }

    if (n_tile <= 0) {
        n_tile = n;
    }
    if (k_tile <= 0) {
        k_tile = k;
    }

    for (int k_offset = 0; k_offset < k; k_offset += k_tile) {
        const int chunk = std::min(k_tile, k - k_offset);

        for (int col = 0; col < n; col += n_tile) {
            const int cols = std::min(n_tile, n - col);
            const T* A_col = block_ptr(A, lda, col, k_offset);

            dispatch_syrk(
                handle,
                cols,
                chunk,
                A_col,
                lda,
                block_ptr(C, ldc, col, col),
                ldc);

            for (int row = col + cols; row < n; row += n_tile) {
                const int rows = std::min(n_tile, n - row);

                dispatch_gemm(
                    handle,
                    rows,
                    cols,
                    chunk,
                    block_ptr(A, lda, row, k_offset),
                    lda,
                    A_col,
                    lda,
                    block_ptr(C, ldc, row, col),
                    ldc);
            }
        }
    }

    return;
}

} // namespace

extern "C" {

void tiled_lower_rank_k_update(rocblas_handle handle,
                               int n,
                               int k,
                               const float* A,
                               int lda,
                               float* C,
                               int ldc,
                               int n_tile,
                               int k_tile)
{
    tiled_lower_rank_k_update_impl(
        handle, n, k, A, lda, C, ldc, n_tile, k_tile);

    return;
}

void tiled_lower_rank_k_update_double(rocblas_handle handle,
                                      int n,
                                      int k,
                                      const double* A,
                                      int lda,
                                      double* C,
                                      int ldc,
                                      int n_tile,
                                      int k_tile)
{
    tiled_lower_rank_k_update_impl(
        handle, n, k, A, lda, C, ldc, n_tile, k_tile);

    return;
}

} // extern "C"
