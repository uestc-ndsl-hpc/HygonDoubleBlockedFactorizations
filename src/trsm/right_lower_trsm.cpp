#include "trsm/right_lower_trsm.h"

#include <type_traits>

#include "utils/common/error_check.h"

namespace {

template <typename T>
void trsm_right_lower_transpose_impl(rocblas_handle handle,
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

} // namespace

extern "C" {

void trsm_right_lower_transpose(rocblas_handle handle,
                                int rows,
                                int cols,
                                const float* A,
                                int lda,
                                float* B,
                                int ldb)
{
    trsm_right_lower_transpose_impl(handle, rows, cols, A, lda, B, ldb);

    return;
}

void trsm_right_lower_transpose_double(rocblas_handle handle,
                                       int rows,
                                       int cols,
                                       const double* A,
                                       int lda,
                                       double* B,
                                       int ldb)
{
    trsm_right_lower_transpose_impl(handle, rows, cols, A, lda, B, ldb);

    return;
}

} // extern "C"
