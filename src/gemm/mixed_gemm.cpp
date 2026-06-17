#include "gemm/mixed_gemm.h"

#include <cstdlib>
#include <cstring>

#include <hip/hip_fp16.h>

#include "kernels/fp16_convert.h"
#include "kernels/hss_gemm.h"
#include "utils/common/error_check.h"

namespace {

const char* custom_backend_for_kernel(const char* kernel)
{
    if (std::strcmp(kernel, "custom_nt_auto") == 0) {
        return "custom_true_hss_preconvert_ab_nt_auto_legacy_small_square_large";
    }
    if (std::strcmp(kernel, "custom_nt_legacy_64x64x16") == 0) {
        return "custom_true_hss_preconvert_ab_nt_legacy_64x64x16";
    }
    if (std::strcmp(kernel, "custom_nt_square_128x128x32_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_square_128x128x32_mix";
    }
    if (std::strcmp(kernel, "custom_nt_wide_64x128x32_4x4_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_wide_64x128x32_4x4_mix";
    }
    if (std::strcmp(kernel, "custom_nt_square4_128x128x32_4x4_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_square4_128x128x32_4x4_mix";
    }

    return "custom_true_hss_preconvert_ab_nt_tuned_128x64x32_mix";
}

} // namespace

extern "C" {

const char* hygon_hss_gemm_backend_name()
{
    return custom_backend_for_kernel(hygon_hss_custom_kernel_name());
}

void launch_hss_gemm_preconverted(
    rocblas_handle handle,
    hipStream_t stream,
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* A_half, int lda,
    const __half* B_half, int ldb,
    float beta,
    float* C_f32, int ldc)
{
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) {
        return;
    }
    (void)handle;

    launch_hss_gemm_true_kernel(
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        A_half,
        lda,
        B_half,
        ldb,
        beta,
        C_f32,
        ldc,
        stream);
}

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
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) {
        return;
    }

    rocblas_half* A_half = nullptr;
    rocblas_half* B_half = nullptr;

    int rows_A = (transA == rocblas_operation_none) ? m : k;
    int cols_A = (transA == rocblas_operation_none) ? k : m;
    int rows_B = (transB == rocblas_operation_none) ? k : n;
    int cols_B = (transB == rocblas_operation_none) ? n : k;

    size_t size_A = static_cast<size_t>(rows_A) * cols_A;
    size_t size_B = static_cast<size_t>(rows_B) * cols_B;

    if (size_A > 0) {
        HIP_CHECK(hipMallocAsync(&A_half, size_A * sizeof(rocblas_half), stream));
        launch_fp32_to_fp16(
            A_f32,
            reinterpret_cast<__half*>(A_half),
            rows_A,
            cols_A,
            lda,
            rows_A,
            stream);
    }

    if (size_B > 0) {
        HIP_CHECK(hipMallocAsync(&B_half, size_B * sizeof(rocblas_half), stream));
        launch_fp32_to_fp16(
            B_f32,
            reinterpret_cast<__half*>(B_half),
            rows_B,
            cols_B,
            ldb,
            rows_B,
            stream);
    }

    launch_hss_gemm_preconverted(
        handle,
        stream,
        transA,
        transB,
        m,
        n,
        k,
        alpha,
        reinterpret_cast<const __half*>(A_half),
        rows_A,
        reinterpret_cast<const __half*>(B_half),
        rows_B,
        beta,
        C_f32,
        ldc);

    if (B_half != nullptr) {
        HIP_CHECK(hipFreeAsync(B_half, stream));
    }
    if (A_half != nullptr) {
        HIP_CHECK(hipFreeAsync(A_half, stream));
    }
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
}

} // extern "C"
