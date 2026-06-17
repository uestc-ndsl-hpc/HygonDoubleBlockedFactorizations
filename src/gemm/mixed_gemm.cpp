#include "gemm/mixed_gemm.h"
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include "utils/common/error_check.h"
#include "kernels/fp16_convert.h"
#include "kernels/hss_gemm.h"

namespace {

bool env_is_truthy(const char* value)
{
    if (value == nullptr) {
        return false;
    }

    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T';
}

bool use_tile_a_conversion()
{
    const char* value = std::getenv("HYGON_HSS_A_MODE");
    if (value != nullptr &&
        (std::strcmp(value, "tile") == 0 || std::strcmp(value, "TILE") == 0 ||
         std::strcmp(value, "f32a") == 0 || std::strcmp(value, "F32A") == 0)) {
        return true;
    }

    return env_is_truthy(std::getenv("HYGON_HSS_TILE_A"));
}

bool use_legacy_half_a_single_b()
{
    const char* value = std::getenv("HYGON_HSS_INPUT_MODE");
    if (value == nullptr) {
        return false;
    }

    return std::strcmp(value, "half_a_single_b") == 0 ||
           std::strcmp(value, "HALF_A_SINGLE_B") == 0 ||
           std::strcmp(value, "hsf") == 0 ||
           std::strcmp(value, "HSF") == 0 ||
           std::strcmp(value, "legacy") == 0 ||
           std::strcmp(value, "LEGACY") == 0;
}

} // namespace

extern "C" {

const char* hygon_half_a_single_b_backend_name()
{
    const bool tile_a = use_tile_a_conversion();
    const char* kernel = hygon_hss_custom_kernel_name();

    if (std::strcmp(kernel, "custom_nt_auto") == 0) {
        return tile_a
            ? "custom_tile_a_nt_auto_legacy_small_square_large"
            : "custom_preconvert_a_nt_auto_legacy_small_square_large";
    } else if (std::strcmp(kernel, "custom_nt_legacy_64x64x16") == 0) {
        return tile_a
            ? "custom_tile_a_nt_legacy_64x64x16"
            : "custom_preconvert_a_nt_legacy_64x64x16";
    } else if (std::strcmp(kernel, "custom_nt_square_128x128x32_mix") == 0) {
        return tile_a
            ? "custom_tile_a_nt_square_128x128x32_mix"
            : "custom_preconvert_a_nt_square_128x128x32_mix";
    } else if (std::strcmp(kernel, "custom_nt_wide_64x128x32_4x4_mix") == 0) {
        return tile_a
            ? "custom_tile_a_nt_wide_64x128x32_4x4_mix"
            : "custom_preconvert_a_nt_wide_64x128x32_4x4_mix";
    } else if (std::strcmp(kernel, "custom_nt_square4_128x128x32_4x4_mix") == 0) {
        return tile_a
            ? "custom_tile_a_nt_square4_128x128x32_4x4_mix"
            : "custom_preconvert_a_nt_square4_128x128x32_4x4_mix";
    }

    return tile_a
        ? "custom_tile_a_nt_tuned_128x64x32_mix"
        : "custom_preconvert_a_nt_tuned_128x64x32_mix";
}

const char* hygon_hss_gemm_backend_name()
{
    if (use_legacy_half_a_single_b()) {
        return hygon_half_a_single_b_backend_name();
    }

    const char* kernel = hygon_hss_custom_kernel_name();

    if (std::strcmp(kernel, "custom_nt_auto") == 0) {
        return "custom_true_hss_preconvert_ab_nt_auto_legacy_small_square_large";
    } else if (std::strcmp(kernel, "custom_nt_legacy_64x64x16") == 0) {
        return "custom_true_hss_preconvert_ab_nt_legacy_64x64x16";
    } else if (std::strcmp(kernel, "custom_nt_square_128x128x32_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_square_128x128x32_mix";
    } else if (std::strcmp(kernel, "custom_nt_wide_64x128x32_4x4_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_wide_64x128x32_4x4_mix";
    } else if (std::strcmp(kernel, "custom_nt_square4_128x128x32_4x4_mix") == 0) {
        return "custom_true_hss_preconvert_ab_nt_square4_128x128x32_4x4_mix";
    }

    return "custom_true_hss_preconvert_ab_nt_tuned_128x64x32_mix";
}

void launch_half_a_single_b_gemm(
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
    (void)handle;

    if (transA == rocblas_operation_none && transB != rocblas_operation_none &&
        use_tile_a_conversion()) {
        launch_hss_gemm_f32a_kernel(
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
            ldc,
            stream);

        return;
    }

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
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;
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

    return;
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
    if (use_legacy_half_a_single_b()) {
        launch_half_a_single_b_gemm(
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

    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;
    (void)handle;

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
