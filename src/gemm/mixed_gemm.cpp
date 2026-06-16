#include "gemm/mixed_gemm.h"
#include <hip/hip_fp16.h>
#include "utils/common/error_check.h"
#include "kernels/fp16_convert.h"
#include "kernels/hss_gemm.h"

extern "C" {

void launch_hss_gemm(
    rocblas_handle handle,
    hipStream_t stream,
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const float* A_f32, int lda,
    const float* B_f32, int ldb,
    float beta,
    float* C_f32, int ldc)
{
    (void)handle;

    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;

    rocblas_half* A_half = nullptr;

    int rows_A = (transA == rocblas_operation_none) ? m : k;
    int cols_A = (transA == rocblas_operation_none) ? k : m;

    size_t size_A = (size_t)rows_A * cols_A;

    if (size_A > 0) {
        HIP_CHECK(hipMallocAsync(&A_half, size_A * sizeof(rocblas_half), stream));
        launch_fp32_to_fp16(
            A_f32,
            reinterpret_cast<__half*>(A_half),
            rows_A, cols_A, lda, rows_A,
            stream);
    }

    launch_hss_gemm_kernel(
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        reinterpret_cast<const __half*>(A_half),
        rows_A,
        B_f32,
        ldb,
        beta,
        C_f32,
        ldc,
        stream);

    if (A_half != nullptr) {
        HIP_CHECK(hipFreeAsync(A_half, stream));
    }

    return;
}

void launch_hhs_gemm(
    rocblas_handle handle,
    hipStream_t stream,
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const float* A_f32, int lda,
    const float* B_f32, int ldb,
    float beta,
    float* C_f32, int ldc)
{
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;

    rocblas_half* A_half = nullptr;
    rocblas_half* B_half = nullptr;
    rocblas_half* C_half = nullptr;

    int rows_A = (transA == rocblas_operation_none) ? m : k;
    int cols_A = (transA == rocblas_operation_none) ? k : m;
    int rows_B = (transB == rocblas_operation_none) ? k : n;
    int cols_B = (transB == rocblas_operation_none) ? n : k;

    size_t size_A = (size_t)rows_A * cols_A;
    size_t size_B = (size_t)rows_B * cols_B;
    size_t size_C = (size_t)m * n;

    HIP_CHECK(hipMallocAsync(&A_half, size_A * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&B_half, size_B * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&C_half, size_C * sizeof(rocblas_half), stream));

    launch_fp32_to_fp16(
        A_f32,
        reinterpret_cast<__half*>(A_half),
        rows_A, cols_A, lda, rows_A,
        stream);

    launch_fp32_to_fp16(
        B_f32,
        reinterpret_cast<__half*>(B_half),
        rows_B, cols_B, ldb, rows_B,
        stream);

    if (beta != 0.0f) {
        launch_fp32_to_fp16(
            C_f32,
            reinterpret_cast<__half*>(C_half),
            m, n, ldc, m,
            stream);
    }

    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        transA,
        transB,
        m, n, k,
        &alpha,
        A_half,
        rocblas_datatype_f16_r,
        rows_A,
        B_half,
        rocblas_datatype_f16_r,
        rows_B,
        &beta,
        C_half,
        rocblas_datatype_f16_r,
        m,
        C_half,
        rocblas_datatype_f16_r,
        m,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0, 0));

    launch_fp16_to_fp32(
        reinterpret_cast<const __half*>(C_half),
        C_f32,
        m, n,
        m, ldc,
        stream);

    HIP_CHECK(hipFreeAsync(A_half, stream));
    HIP_CHECK(hipFreeAsync(B_half, stream));
    HIP_CHECK(hipFreeAsync(C_half, stream));

    return;
}

void launch_mixed_gemm(
    rocblas_handle handle,
    hipStream_t stream,
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const float* A_f32, int lda,
    const float* B_f32, int ldb,
    float beta,
    float* C_f32, int ldc)
{
    launch_hss_gemm(
        handle,
        stream,
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        A_f32,
        lda,
        B_f32,
        ldb,
        beta,
        C_f32,
        ldc);

    return;
}

} // extern "C"
