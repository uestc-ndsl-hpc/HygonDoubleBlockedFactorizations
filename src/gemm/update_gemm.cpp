#include "gemm/update_gemm.h"

#include <algorithm>
#include <type_traits>

#include "utils/common/error_check.h"

namespace {

template <typename T>
void gemm_nt_update_impl(rocblas_handle handle,
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
void gemm_nt_update_m_tiled_impl(rocblas_handle handle,
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
        gemm_nt_update_impl(
            handle, rows, cols, k, A, lda, B, ldb, C, ldc);
        return;
    }

    for (int row = 0; row < rows; row += m_tile) {
        const int tile_rows = std::min(m_tile, rows - row);
        gemm_nt_update_impl(
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

    return;
}

} // namespace

extern "C" {

void gemm_nt_update(rocblas_handle handle,
                    int rows,
                    int cols,
                    int k,
                    const float* A,
                    int lda,
                    const float* B,
                    int ldb,
                    float* C,
                    int ldc)
{
    gemm_nt_update_impl(handle, rows, cols, k, A, lda, B, ldb, C, ldc);

    return;
}

void gemm_nt_update_double(rocblas_handle handle,
                           int rows,
                           int cols,
                           int k,
                           const double* A,
                           int lda,
                           const double* B,
                           int ldb,
                           double* C,
                           int ldc)
{
    gemm_nt_update_impl(handle, rows, cols, k, A, lda, B, ldb, C, ldc);

    return;
}

void gemm_nt_update_m_tiled(rocblas_handle handle,
                            int rows,
                            int cols,
                            int k,
                            const float* A,
                            int lda,
                            const float* B,
                            int ldb,
                            float* C,
                            int ldc,
                            int m_tile)
{
    gemm_nt_update_m_tiled_impl(
        handle, rows, cols, k, A, lda, B, ldb, C, ldc, m_tile);

    return;
}

void gemm_nt_update_m_tiled_double(rocblas_handle handle,
                                   int rows,
                                   int cols,
                                   int k,
                                   const double* A,
                                   int lda,
                                   const double* B,
                                   int ldb,
                                   double* C,
                                   int ldc,
                                   int m_tile)
{
    gemm_nt_update_m_tiled_impl(
        handle, rows, cols, k, A, lda, B, ldb, C, ldc, m_tile);

    return;
}

} // extern "C"
