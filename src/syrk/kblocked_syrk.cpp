#include "syrk/kblocked_syrk.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "utils/common/error_check.h"

namespace {

template <typename T>
const T* matrix_ptr(const T* A, int lda, int row, int col)
{
    return A + static_cast<size_t>(col) * static_cast<size_t>(lda) + row;
}

template <typename T>
void syrk_lower_update_impl(rocblas_handle handle,
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
void syrk_lower_update_kblocked_impl(rocblas_handle handle,
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
        syrk_lower_update_impl(handle, n, k, A, lda, C, ldc);
        return;
    }

    for (int k_offset = 0; k_offset < k; k_offset += k_tile) {
        const int chunk = std::min(k_tile, k - k_offset);
        syrk_lower_update_impl(
            handle,
            n,
            chunk,
            matrix_ptr(A, lda, 0, k_offset),
            lda,
            C,
            ldc);
    }

    return;
}

} // namespace

extern "C" {

void syrk_lower_update(rocblas_handle handle,
                       int n,
                       int k,
                       const float* A,
                       int lda,
                       float* C,
                       int ldc)
{
    syrk_lower_update_impl(handle, n, k, A, lda, C, ldc);

    return;
}

void syrk_lower_update_double(rocblas_handle handle,
                              int n,
                              int k,
                              const double* A,
                              int lda,
                              double* C,
                              int ldc)
{
    syrk_lower_update_impl(handle, n, k, A, lda, C, ldc);

    return;
}

void syrk_lower_update_kblocked(rocblas_handle handle,
                                int n,
                                int k,
                                const float* A,
                                int lda,
                                float* C,
                                int ldc,
                                int k_tile)
{
    syrk_lower_update_kblocked_impl(
        handle, n, k, A, lda, C, ldc, k_tile);

    return;
}

void syrk_lower_update_kblocked_double(rocblas_handle handle,
                                       int n,
                                       int k,
                                       const double* A,
                                       int lda,
                                       double* C,
                                       int ldc,
                                       int k_tile)
{
    syrk_lower_update_kblocked_impl(
        handle, n, k, A, lda, C, ldc, k_tile);

    return;
}

} // extern "C"
