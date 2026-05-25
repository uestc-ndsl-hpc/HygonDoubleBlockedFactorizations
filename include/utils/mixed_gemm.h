#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include "utils/error_check.h"
#include "kernels/fp16_convert.h"

extern "C" {

/**
 * @brief 基于 Tensor Cores 的混合精度（FP16 计算/输入，FP32 累加/输出）通用 GEMM
 * 计算公式: C = alpha * op(A) * op(B) + beta * C
 */
inline void launch_mixed_gemm(
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
    // 如果维度无效，或者由于计算规模为0且无须累加，则直接返回
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;

    rocblas_half* A_half = nullptr;
    rocblas_half* B_half = nullptr;
    rocblas_half* C_half = nullptr;

    // 根据矩阵是否转置，计算 A 和 B 在物理内存中的实际行列数
    int rows_A = (transA == rocblas_operation_none) ? m : k;
    int cols_A = (transA == rocblas_operation_none) ? k : m;
    int rows_B = (transB == rocblas_operation_none) ? k : n;
    int cols_B = (transB == rocblas_operation_none) ? n : k;

    size_t size_A = (size_t)rows_A * cols_A;
    size_t size_B = (size_t)rows_B * cols_B;
    size_t size_C = (size_t)m * n;

    // 在当前 Stream 上异步分配临时半精度缓冲区
    HIP_CHECK(hipMallocAsync(&A_half, size_A * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&B_half, size_B * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&C_half, size_C * sizeof(rocblas_half), stream));

    // 1. 异步转换矩阵 A (FP32 -> FP16)
    launch_fp32_to_fp16(
        A_f32,
        reinterpret_cast<__half*>(A_half),
        rows_A, cols_A, lda, rows_A,
        stream);

    // 2. 异步转换矩阵 B (FP32 -> FP16)
    launch_fp32_to_fp16(
        B_f32,
        reinterpret_cast<__half*>(B_half),
        rows_B, cols_B, ldb, rows_B,
        stream);

    // 3. 异步转换矩阵 C (FP32 -> FP16) (仅当 beta 不为 0 时需要同步初值)
    if (beta != 0.0f) {
        launch_fp32_to_fp16(
            C_f32,
            reinterpret_cast<__half*>(C_half),
            m, n, ldc, m,
            stream);
    }

    // 4. 调用 rocBLAS Tensor Core 核心（启用 FP32 内部累加器以抑制截断误差）
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
        rocblas_datatype_f32_r, // 内部乘加使用 FP32 累加器以提高稳定性
        rocblas_gemm_algo_standard,
        0, 0));

    // 5. 异步将结果转回主内存 C (FP16 -> FP32)
    launch_fp16_to_fp32(
        reinterpret_cast<const __half*>(C_half),
        C_f32,
        m, n,
        m, ldc,
        stream);

    // 6. 异步释放临时缓冲区
    HIP_CHECK(hipFreeAsync(A_half, stream));
    HIP_CHECK(hipFreeAsync(B_half, stream));
    HIP_CHECK(hipFreeAsync(C_half, stream));
}

inline void launch_mixed_gemm_test(
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
    // 如果维度无效，或者计算规模为 0 且无须累加，则直接返回
    if (m <= 0 || n <= 0 || (k <= 0 && beta == 1.0f)) return;

    rocblas_half* A_half = nullptr;
    rocblas_half* B_half = nullptr;

    // 根据矩阵是否转置，计算 A 和 B 在物理内存中的实际行列数
    int rows_A = (transA == rocblas_operation_none) ? m : k;
    int cols_A = (transA == rocblas_operation_none) ? k : m;
    int rows_B = (transB == rocblas_operation_none) ? k : n;
    int cols_B = (transB == rocblas_operation_none) ? n : k;

    size_t size_A = (size_t)rows_A * cols_A;
    size_t size_B = (size_t)rows_B * cols_B;

    // 1. 在当前流（Stream）上仅异步分配 A 和 B 的临时半精度空间
    HIP_CHECK(hipMallocAsync(&A_half, size_A * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&B_half, size_B * sizeof(rocblas_half), stream));

    // 2. 将矩阵 A 和 B 从 FP32 转换为 FP16 (A_f32 -> A_half, B_f32 -> B_half)
    // 连续线程合并读取与写入，确保转换效率高
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

    // 3. 调用 rocBLAS Tensor Core 混合精度核心。
    //    输入矩阵 C 与输出目标 D 均直接传入原有的高精度 C_f32 变量，主维度参数安全传入真实的 ldc
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        transA,
        transB,
        m, n, k,
        &alpha,
        A_half,
        rocblas_datatype_f16_r, // 输入 A 为半精度
        rows_A,
        B_half,
        rocblas_datatype_f16_r, // 输入 B 为半精度
        rows_B,
        &beta,
        C_f32,                  // 输入 C 直接在全局内存中使用 FP32 精度
        rocblas_datatype_f32_r, // 输入 C 的存储格式为 FP32
        ldc,                    // 输入 C 的 Leading Dimension 必须传入真实的 ldc
        C_f32,                  // 输出 D 直接在全局内存中使用 FP32 精度写回
        rocblas_datatype_f32_r, // 输出 D 的存储格式为 FP32
        ldc,                    // 输出 D 的 Leading Dimension 必须传入真实的 ldc
        rocblas_datatype_f32_r, // 内部计算累加器（Computation Type）指定为 FP32
        rocblas_gemm_algo_standard,
        0, 0));

    // 4. 异步释放 A 和 B 的临时缓冲区，无需任何 CPU-GPU 阻塞
    HIP_CHECK(hipFreeAsync(A_half, stream));
    HIP_CHECK(hipFreeAsync(B_half, stream));
}

} // extern "C"