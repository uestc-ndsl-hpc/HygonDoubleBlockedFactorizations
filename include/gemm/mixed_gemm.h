#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    float* C_f32, int ldc);

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
    float* C_f32, int ldc);

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
    float* C_f32, int ldc);

const char* hygon_hss_gemm_backend_name();

#ifdef __cplusplus
}
#endif
