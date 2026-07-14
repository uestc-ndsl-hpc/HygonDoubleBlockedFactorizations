#include "potrf/fast_recursive_potrf.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "gemm/strassen_gemm.h"
#include "potrf/blocked_potrf.h"
#include "utils/common/error_check.h"

namespace {

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * lda + row;
}

template <typename T>
void blocked_potrf(rocblas_handle handle,
                   T* A,
                   int lda,
                   int n,
                   int nb,
                   int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        blocked_right_looking_potrf(handle, A, lda, n, nb, info);
    } else {
        blocked_right_looking_potrf_double(handle, A, lda, n, nb, info);
    }

    return;
}

template <typename T>
void trsm_right_lower_transpose(rocblas_handle handle,
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
void syrk_lower_update(rocblas_handle handle,
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
void syrk_lower_update_k_tiled(rocblas_handle handle,
                               int n,
                               int k,
                               const T* A,
                               int lda,
                               T* C,
                               int ldc,
                               int k_tile)
{
    if (k_tile <= 0 || k_tile >= k) {
        syrk_lower_update(handle, n, k, A, lda, C, ldc);
        return;
    }

    for (int k_offset = 0; k_offset < k; k_offset += k_tile) {
        const int tile_k = std::min(k_tile, k - k_offset);
        syrk_lower_update(
            handle,
            n,
            tile_k,
            A + static_cast<size_t>(k_offset) * lda,
            lda,
            C,
            ldc);
    }

    return;
}

template <typename T>
void strassen_nt_update(rocblas_handle handle,
                        int rows,
                        int cols,
                        int k,
                        const T* A,
                        int lda,
                        const T* B,
                        int ldb,
                        T* C,
                        int ldc,
                        int strassen_min_size)
{
    if constexpr (std::is_same_v<T, float>) {
        gemm_nt_update_strassen_tiled(
            handle,
            rows,
            cols,
            k,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            strassen_min_size);
    } else {
        gemm_nt_update_strassen_tiled_double(
            handle,
            rows,
            cols,
            k,
            A,
            lda,
            B,
            ldb,
            C,
            ldc,
            strassen_min_size);
    }

    return;
}

template <typename T>
void fast_recursive_trsm(rocblas_handle handle,
                         const T* triangular,
                         int lda,
                         T* B,
                         int ldb,
                         int rows,
                         int cols,
                         int leaf_size,
                         int strassen_min_size)
{
    if (rows <= 0 || cols <= 0) {
        return;
    }

    if (cols <= leaf_size) {
        trsm_right_lower_transpose(
            handle, rows, cols, triangular, lda, B, ldb);
        return;
    }

    const int first = cols / 2;
    const int second = cols - first;

    fast_recursive_trsm(
        handle,
        triangular,
        lda,
        B,
        ldb,
        rows,
        first,
        leaf_size,
        strassen_min_size);

    strassen_nt_update(
        handle,
        rows,
        second,
        first,
        B,
        ldb,
        triangular + first,
        lda,
        B + static_cast<size_t>(first) * ldb,
        ldb,
        strassen_min_size);

    fast_recursive_trsm(
        handle,
        triangular + static_cast<size_t>(first) * lda + first,
        lda,
        B + static_cast<size_t>(first) * ldb,
        ldb,
        rows,
        second,
        leaf_size,
        strassen_min_size);

    return;
}

template <typename T>
void fast_symmetric_update(rocblas_handle handle,
                           int n,
                           int k,
                           const T* A,
                           int lda,
                           T* C,
                           int ldc,
                           int strassen_min_size,
                           int syrk_k_tile)
{
    if (n <= 1) {
        syrk_lower_update_k_tiled(
            handle, n, k, A, lda, C, ldc, syrk_k_tile);
        return;
    }

    const int first = n / 2;
    const int second = n - first;

    syrk_lower_update_k_tiled(
        handle,
        first,
        k,
        A,
        lda,
        C,
        ldc,
        syrk_k_tile);

    strassen_nt_update(
        handle,
        second,
        first,
        k,
        A + first,
        lda,
        A,
        lda,
        C + first,
        ldc,
        strassen_min_size);

    syrk_lower_update_k_tiled(
        handle,
        second,
        k,
        A + first,
        lda,
        C + static_cast<size_t>(first) * ldc + first,
        ldc,
        syrk_k_tile);

    return;
}

template <typename T>
void fast_recursive_strassen_potrf_impl(rocblas_handle handle,
                                        T* A,
                                        int lda,
                                        int n,
                                        int leaf_size,
                                        int base_nb,
                                        int strassen_min_size,
                                        int syrk_k_tile,
                                        int* info)
{
    if (n <= 0) {
        return;
    }

    if (n <= leaf_size) {
        blocked_potrf(handle, A, lda, n, base_nb, info);
        return;
    }

    const int first = n / 2;
    const int second = n - first;
    T* A11 = A;
    T* A21 = block_ptr(A, lda, first, 0);
    T* A22 = block_ptr(A, lda, first, first);

    fast_recursive_strassen_potrf_impl(
        handle,
        A11,
        lda,
        first,
        leaf_size,
        base_nb,
        strassen_min_size,
        syrk_k_tile,
        info);

    fast_recursive_trsm(
        handle,
        A11,
        lda,
        A21,
        lda,
        second,
        first,
        base_nb,
        strassen_min_size);

    fast_symmetric_update(
        handle,
        second,
        first,
        A21,
        lda,
        A22,
        lda,
        strassen_min_size,
        syrk_k_tile);

    fast_recursive_strassen_potrf_impl(
        handle,
        A22,
        lda,
        second,
        leaf_size,
        base_nb,
        strassen_min_size,
        syrk_k_tile,
        info);

    return;
}

template <typename T>
void run_fast_recursive_strassen_potrf(rocblas_handle handle,
                                       T* A,
                                       int lda,
                                       int n,
                                       int leaf_size,
                                       int base_nb,
                                       int strassen_min_size,
                                       int syrk_k_tile,
                                       int* info)
{
    if (leaf_size <= 0) {
        leaf_size = 8192;
    }
    if (base_nb <= 0) {
        base_nb = 2048;
    }
    base_nb = std::min(base_nb, leaf_size);
    if (strassen_min_size <= 0) {
        strassen_min_size = 8192;
    }
    if (syrk_k_tile <= 0) {
        syrk_k_tile = 8192;
    }

    fast_recursive_strassen_potrf_impl(
        handle,
        A,
        lda,
        n,
        leaf_size,
        base_nb,
        strassen_min_size,
        syrk_k_tile,
        info);

    return;
}

} // namespace

extern "C" {

void fast_recursive_strassen_potrf(rocblas_handle handle,
                                   float* A,
                                   int lda,
                                   int n,
                                   int leaf_size,
                                   int base_nb,
                                   int strassen_min_size,
                                   int syrk_k_tile,
                                   int* info)
{
    run_fast_recursive_strassen_potrf(
        handle,
        A,
        lda,
        n,
        leaf_size,
        base_nb,
        strassen_min_size,
        syrk_k_tile,
        info);

    return;
}

void fast_recursive_strassen_potrf_double(rocblas_handle handle,
                                          double* A,
                                          int lda,
                                          int n,
                                          int leaf_size,
                                          int base_nb,
                                          int strassen_min_size,
                                          int syrk_k_tile,
                                          int* info)
{
    run_fast_recursive_strassen_potrf(
        handle,
        A,
        lda,
        n,
        leaf_size,
        base_nb,
        strassen_min_size,
        syrk_k_tile,
        info);

    return;
}

} // extern "C"
