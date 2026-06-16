#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

void launch_hss_gemm_kernel(
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const __half* A_half,
    int lda,
    const float* B_f32,
    int ldb,
    float beta,
    float* C_f32,
    int ldc,
    hipStream_t stream);

void launch_hss_gemm_f32a_kernel(
    rocblas_operation transA,
    rocblas_operation transB,
    int m,
    int n,
    int k,
    float alpha,
    const float* A_f32,
    int lda,
    const float* B_f32,
    int ldb,
    float beta,
    float* C_f32,
    int ldc,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif
