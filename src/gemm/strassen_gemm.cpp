#include "gemm/strassen_gemm.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include <hip/hip_runtime.h>

#include "utils/common/error_check.h"

namespace {

template <typename T>
const T* block_ptr(const T* A,
                   int lda,
                   rocblas_operation operation,
                   int row,
                   int col)
{
    if (operation == rocblas_operation_none) {
        return A + static_cast<size_t>(col) * lda + row;
    }

    return A + static_cast<size_t>(row) * lda + col;
}

template <typename T>
T* block_ptr(T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * lda + row;
}

template <typename T>
void geam(rocblas_handle handle,
          rocblas_operation operation_a,
          rocblas_operation operation_b,
          int rows,
          int cols,
          T alpha,
          const T* A,
          int lda,
          T beta,
          const T* B,
          int ldb,
          T* C,
          int ldc)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgeam(
            handle,
            operation_a,
            operation_b,
            rows,
            cols,
            &alpha,
            A,
            lda,
            &beta,
            B,
            ldb,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgeam(
            handle,
            operation_a,
            operation_b,
            rows,
            cols,
            &alpha,
            A,
            lda,
            &beta,
            B,
            ldb,
            C,
            ldc));
    }

    return;
}

template <typename T>
void gemm(rocblas_handle handle,
          rocblas_operation operation_a,
          rocblas_operation operation_b,
          int rows,
          int cols,
          int k,
          T alpha,
          const T* A,
          int lda,
          const T* B,
          int ldb,
          T beta,
          T* C,
          int ldc)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            operation_a,
            operation_b,
            rows,
            cols,
            k,
            &alpha,
            A,
            lda,
            B,
            ldb,
            &beta,
            C,
            ldc));
    } else {
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            operation_a,
            operation_b,
            rows,
            cols,
            k,
            &alpha,
            A,
            lda,
            B,
            ldb,
            &beta,
            C,
            ldc));
    }

    return;
}

template <typename T>
void gemm_nt_update(rocblas_handle handle,
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
    const T neg_one = static_cast<T>(-1);
    const T one = static_cast<T>(1);

    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        rows,
        cols,
        k,
        neg_one,
        A,
        lda,
        B,
        ldb,
        one,
        C,
        ldc);

    return;
}

template <typename T>
void linear_combination(rocblas_handle handle,
                        int n,
                        rocblas_operation operation,
                        const T* A,
                        int lda,
                        T alpha,
                        const T* B,
                        int ldb,
                        T beta,
                        T* output)
{
    geam(
        handle,
        operation,
        operation,
        n,
        n,
        alpha,
        A,
        lda,
        beta,
        B,
        ldb,
        output,
        n);

    return;
}

template <typename T>
void accumulate_product(rocblas_handle handle,
                        int n,
                        T coefficient,
                        const T* product,
                        T* C,
                        int ldc)
{
    const T one = static_cast<T>(1);

    geam(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        n,
        n,
        coefficient,
        product,
        n,
        one,
        C,
        ldc,
        C,
        ldc);

    return;
}

template <typename T>
void strassen_square_nt_update(rocblas_handle handle,
                               int n,
                               const T* A,
                               int lda,
                               const T* B,
                               int ldb,
                               T* C,
                               int ldc)
{
    if (n <= 1 || n % 2 != 0) {
        gemm_nt_update(handle, n, n, n, A, lda, B, ldb, C, ldc);
        return;
    }

    const int half = n / 2;
    const size_t tile_elements =
        static_cast<size_t>(half) * static_cast<size_t>(half);
    const size_t workspace_elements = 9 * tile_elements;

    hipStream_t stream = nullptr;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    T* workspace = nullptr;
    HIP_CHECK(hipMallocAsync(
        &workspace, workspace_elements * sizeof(T), stream));

    T* sum_a = workspace;
    T* sum_b = sum_a + tile_elements;
    T* products[7];
    for (int index = 0; index < 7; ++index) {
        products[index] = sum_b + (index + 1) * tile_elements;
    }

    const T one = static_cast<T>(1);
    const T neg_one = static_cast<T>(-1);
    const T zero = static_cast<T>(0);

    const T* A11 = block_ptr(A, lda, rocblas_operation_none, 0, 0);
    const T* A12 = block_ptr(A, lda, rocblas_operation_none, 0, half);
    const T* A21 = block_ptr(A, lda, rocblas_operation_none, half, 0);
    const T* A22 = block_ptr(A, lda, rocblas_operation_none, half, half);

    const T* B11 = block_ptr(B, ldb, rocblas_operation_transpose, 0, 0);
    const T* B12 = block_ptr(B, ldb, rocblas_operation_transpose, 0, half);
    const T* B21 = block_ptr(B, ldb, rocblas_operation_transpose, half, 0);
    const T* B22 = block_ptr(B, ldb, rocblas_operation_transpose, half, half);

    linear_combination(
        handle, half, rocblas_operation_none, A11, lda, one, A22, lda, one, sum_a);
    linear_combination(
        handle,
        half,
        rocblas_operation_transpose,
        B11,
        ldb,
        one,
        B22,
        ldb,
        one,
        sum_b);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        half,
        half,
        half,
        one,
        sum_a,
        half,
        sum_b,
        half,
        zero,
        products[0],
        half);

    linear_combination(
        handle, half, rocblas_operation_none, A21, lda, one, A22, lda, one, sum_a);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        half,
        half,
        half,
        one,
        sum_a,
        half,
        B11,
        ldb,
        zero,
        products[1],
        half);

    linear_combination(
        handle,
        half,
        rocblas_operation_transpose,
        B12,
        ldb,
        one,
        B22,
        ldb,
        neg_one,
        sum_b);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        half,
        half,
        half,
        one,
        A11,
        lda,
        sum_b,
        half,
        zero,
        products[2],
        half);

    linear_combination(
        handle,
        half,
        rocblas_operation_transpose,
        B21,
        ldb,
        one,
        B11,
        ldb,
        neg_one,
        sum_b);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        half,
        half,
        half,
        one,
        A22,
        lda,
        sum_b,
        half,
        zero,
        products[3],
        half);

    linear_combination(
        handle, half, rocblas_operation_none, A11, lda, one, A12, lda, one, sum_a);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        half,
        half,
        half,
        one,
        sum_a,
        half,
        B22,
        ldb,
        zero,
        products[4],
        half);

    linear_combination(
        handle,
        half,
        rocblas_operation_none,
        A21,
        lda,
        one,
        A11,
        lda,
        neg_one,
        sum_a);
    linear_combination(
        handle,
        half,
        rocblas_operation_transpose,
        B11,
        ldb,
        one,
        B12,
        ldb,
        one,
        sum_b);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        half,
        half,
        half,
        one,
        sum_a,
        half,
        sum_b,
        half,
        zero,
        products[5],
        half);

    linear_combination(
        handle,
        half,
        rocblas_operation_none,
        A12,
        lda,
        one,
        A22,
        lda,
        neg_one,
        sum_a);
    linear_combination(
        handle,
        half,
        rocblas_operation_transpose,
        B21,
        ldb,
        one,
        B22,
        ldb,
        one,
        sum_b);
    gemm(
        handle,
        rocblas_operation_none,
        rocblas_operation_none,
        half,
        half,
        half,
        one,
        sum_a,
        half,
        sum_b,
        half,
        zero,
        products[6],
        half);

    T* C11 = block_ptr(C, ldc, 0, 0);
    T* C12 = block_ptr(C, ldc, 0, half);
    T* C21 = block_ptr(C, ldc, half, 0);
    T* C22 = block_ptr(C, ldc, half, half);

    accumulate_product(handle, half, neg_one, products[0], C11, ldc);
    accumulate_product(handle, half, neg_one, products[3], C11, ldc);
    accumulate_product(handle, half, one, products[4], C11, ldc);
    accumulate_product(handle, half, neg_one, products[6], C11, ldc);

    accumulate_product(handle, half, neg_one, products[2], C12, ldc);
    accumulate_product(handle, half, neg_one, products[4], C12, ldc);

    accumulate_product(handle, half, neg_one, products[1], C21, ldc);
    accumulate_product(handle, half, neg_one, products[3], C21, ldc);

    accumulate_product(handle, half, neg_one, products[0], C22, ldc);
    accumulate_product(handle, half, one, products[1], C22, ldc);
    accumulate_product(handle, half, neg_one, products[2], C22, ldc);
    accumulate_product(handle, half, neg_one, products[5], C22, ldc);

    HIP_CHECK(hipFreeAsync(workspace, stream));

    return;
}

template <typename T>
void gemm_nt_update_strassen_tiled_impl(rocblas_handle handle,
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
    if (rows <= 0 || cols <= 0 || k <= 0) {
        return;
    }

    const int square_size = cols;
    for (int k_offset = 0; k_offset < k; k_offset += square_size) {
        const int tile_k = std::min(square_size, k - k_offset);

        for (int row_offset = 0; row_offset < rows; row_offset += square_size) {
            const int tile_rows =
                std::min(square_size, rows - row_offset);
            const T* tile_a =
                A + static_cast<size_t>(k_offset) * lda + row_offset;
            const T* tile_b =
                B + static_cast<size_t>(k_offset) * ldb;
            T* tile_c = C + row_offset;

            const bool use_strassen =
                square_size >= strassen_min_size &&
                square_size % 2 == 0 &&
                tile_rows == square_size &&
                tile_k == square_size;

            if (use_strassen) {
                strassen_square_nt_update(
                    handle,
                    square_size,
                    tile_a,
                    lda,
                    tile_b,
                    ldb,
                    tile_c,
                    ldc);
            } else {
                gemm_nt_update(
                    handle,
                    tile_rows,
                    cols,
                    tile_k,
                    tile_a,
                    lda,
                    tile_b,
                    ldb,
                    tile_c,
                    ldc);
            }
        }
    }

    return;
}

} // namespace

extern "C" {

void gemm_nt_update_strassen_tiled(rocblas_handle handle,
                                   int rows,
                                   int cols,
                                   int k,
                                   const float* A,
                                   int lda,
                                   const float* B,
                                   int ldb,
                                   float* C,
                                   int ldc,
                                   int strassen_min_size)
{
    gemm_nt_update_strassen_tiled_impl(
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

    return;
}

void gemm_nt_update_strassen_tiled_double(rocblas_handle handle,
                                          int rows,
                                          int cols,
                                          int k,
                                          const double* A,
                                          int lda,
                                          const double* B,
                                          int ldb,
                                          double* C,
                                          int ldc,
                                          int strassen_min_size)
{
    gemm_nt_update_strassen_tiled_impl(
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

    return;
}

} // extern "C"
