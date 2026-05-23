#include "trsm/recursive_trsm.h"
#include "utils/error_check.h"
#include "utils/stream_pool.h"
#include "kernels/fp16_convert.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <algorithm>

extern "C" {

// ======================================================
// 内部纯递归执行器 (单精度)
// ======================================================
void recursive_trsm_internal(rocblas_handle handle,
                             float* A, int lda,
                             float* B, int ldb,
                             int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        float alpha = 1.0f;
        rocblas_strsm(handle,
                      rocblas_side_right,          
                      rocblas_fill_lower,          
                      rocblas_operation_transpose, 
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_internal(handle, A, lda, B, ldb, n, m1, nb);

    float alpha_gemm = -1.0f;
    float beta_gemm = 1.0f;
    rocblas_sgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose, 
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,                      
                  A + m1, lda,                 
                  &beta_gemm,
                  B + (size_t)m1 * ldb, ldb);  

    recursive_trsm_internal(handle, A + (size_t)m1 * lda + m1, lda, B + (size_t)m1 * ldb, ldb, n, m2, nb);
}

// ======================================================
// 内部纯递归执行器 (双精度)
// ======================================================
void recursive_trsm_double_internal(rocblas_handle handle,
                                    double* A, int lda,
                                    double* B, int ldb,
                                    int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        double alpha = 1.0;
        rocblas_dtrsm(handle,
                      rocblas_side_right,
                      rocblas_fill_lower,
                      rocblas_operation_transpose,
                      rocblas_diagonal_non_unit,
                      n, m, &alpha,
                      A, lda,
                      B, ldb);
        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m1, nb);

    double alpha_gemm = -1.0;
    double beta_gemm = 1.0;
    rocblas_dgemm(handle,
                  rocblas_operation_none,
                  rocblas_operation_transpose,
                  n, m2, m1,
                  &alpha_gemm,
                  B, ldb,
                  A + m1, lda,
                  &beta_gemm,
                  B + (size_t)m1 * ldb, ldb);

    recursive_trsm_double_internal(handle, A + (size_t)m1 * lda + m1, lda, B + (size_t)m1 * ldb, ldb, n, m2, nb);
}

// ======================================================
// Public API (float)
// ======================================================
void recursive_trsm(
    rocblas_handle handle,
    StreamPool* pool,
    float* A, int lda,
    float* B, int ldb,
    int n, int m, int nb)
{
    const int N_LIMIT = 8192;

    if (n <= N_LIMIT) {
        recursive_trsm_internal(handle, A, lda, B, ldb, n, m, nb);
        return;
    }

    hipStream_t orig_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &orig_stream));

    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;
    std::vector<hipEvent_t> events(num_chunks, nullptr);

    for (int i = 0; i < num_chunks; i++) {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);

        auto &ctx = pool->get(i);

        HIP_CHECK(hipEventCreate(&events[i]));

        recursive_trsm_internal(
            ctx.handle,
            A, lda,
            B + row_start, ldb,
            current_n, m, nb);

        HIP_CHECK(hipEventRecord(events[i], ctx.stream));
    }

    for (int i = 0; i < num_chunks; i++) {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    for (int i = 0; i < num_chunks; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

// ======================================================
// Public API (double)
// ======================================================
void recursive_trsm_double(
    rocblas_handle handle,
    StreamPool* pool,
    double* A, int lda,
    double* B, int ldb,
    int n, int m, int nb)
{
    const int N_LIMIT = 8192;

    if (n <= N_LIMIT) {
        recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m, nb);
        return;
    }

    hipStream_t orig_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &orig_stream));

    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;

    std::vector<hipEvent_t> events(num_chunks, nullptr);

    for (int i = 0; i < num_chunks; i++) {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);

        auto &ctx = pool->get(i);

        HIP_CHECK(hipEventCreate(&events[i]));

        recursive_trsm_double_internal(
            ctx.handle,
            A, lda,
            B + row_start, ldb,
            current_n, m, nb);

        HIP_CHECK(hipEventRecord(events[i], ctx.stream));
    }

    for (int i = 0; i < num_chunks; i++) {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    for (int i = 0; i < num_chunks; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

static inline void tc_gemm_update_fp32_output(
    rocblas_handle handle,
    hipStream_t stream,
    int n,
    int m2,
    int m1,
    float* B1_f32,
    int ldb,
    float* A21_f32,
    int lda,
    float* B2_f32)
{
    rocblas_half *B1_half = nullptr;
    rocblas_half *A21_half = nullptr;
    rocblas_half *B2_half = nullptr;

    size_t size_B1  = (size_t)n * m1;
    size_t size_A21 = (size_t)m2 * m1;
    size_t size_B2  = (size_t)n * m2;

    HIP_CHECK(hipMallocAsync(&B1_half, size_B1 * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&A21_half, size_A21 * sizeof(rocblas_half), stream));
    HIP_CHECK(hipMallocAsync(&B2_half, size_B2 * sizeof(rocblas_half), stream));

    // 1. 转换 B1
    launch_fp32_to_fp16(
        B1_f32,
        reinterpret_cast<__half*>(B1_half),
        n, m1, ldb, n,
        stream);

    // 2. 转换 A21
    launch_fp32_to_fp16(
        A21_f32,
        reinterpret_cast<__half*>(A21_half),
        m2, m1, lda, m2,
        stream);

    // ==========================================
    // 【新增修复】：将 B2_f32 的初始数据同步至 B2_half
    // ==========================================
    launch_fp32_to_fp16(
        B2_f32,
        reinterpret_cast<__half*>(B2_half),
        n, m2, ldb, n,
        stream);

    float alpha = -1.0f;
    float beta  = 1.0f;

    // 3. 执行计算 (D = alpha * B1 * A21^T + beta * B2)
    rocblas_gemm_ex(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        n, m2, m1,
        &alpha,
        B1_half,
        rocblas_datatype_f16_r,
        n,
        A21_half,
        rocblas_datatype_f16_r,
        m2,
        &beta,
        B2_half, // 输入 C 矩阵已通过上面修复被赋予正确初值
        rocblas_datatype_f16_r,
        n,
        B2_half, // 输出 D 矩阵
        rocblas_datatype_f16_r,
        n,
        rocblas_datatype_f32_r, // 使用 FP32 作为累加器提升内部乘加的数值稳定性
        rocblas_gemm_algo_standard,
        0, 0);

    // 4. 将计算完成后的数据转回 B2_f32
    launch_fp16_to_fp32(
        reinterpret_cast<const __half*>(B2_half),
        B2_f32,
        n, m2,
        n, ldb,
        stream);

    HIP_CHECK(hipFreeAsync(B1_half, stream));
    HIP_CHECK(hipFreeAsync(A21_half, stream));
    HIP_CHECK(hipFreeAsync(B2_half, stream));
}

void recursive_trsm_mixed_internal(
    rocblas_handle handle,

    float* A,
    int lda,

    float* B,
    int ldb,

    int n,
    int m,
    int nb)
{
    if (n <= 0 || m <= 0)
        return;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if (m <= nb)
    {
        float alpha = 1.0f;

        rocblas_strsm(
            handle,
            rocblas_side_right,
            rocblas_fill_lower,
            rocblas_operation_transpose,
            rocblas_diagonal_non_unit,
            n,
            m,
            &alpha,
            A,
            lda,
            B,
            ldb);

        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_mixed_internal(
        handle,
        A,
        lda,
        B,
        ldb,
        n,
        m1,
        nb);

    tc_gemm_update_fp32_output(
        handle,
        stream,
        n,
        m2,
        m1,
        B,
        ldb,
        A + m1,
        lda,
        B + (size_t)m1 * ldb);

    recursive_trsm_mixed_internal(
        handle,
        A + (size_t)m1 * lda + m1,
        lda,
        B + (size_t)m1 * ldb,
        ldb,
        n,
        m2,
        nb);
}

void recursive_trsm_mixed(
    rocblas_handle handle,
    StreamPool* pool,
    float* A,
    int lda,
    float* B,
    int ldb,
    int n,
    int m,
    int nb)
{
    const int N_LIMIT = 8192;

    if (n <= N_LIMIT)
    {
        recursive_trsm_mixed_internal(
            handle,
            A, lda,
            B, ldb,
            n, m, nb);

        return;
    }

    hipStream_t orig_stream;
    rocblas_get_stream(handle, &orig_stream);

    int num_chunks =
        (n + N_LIMIT - 1) / N_LIMIT;

    std::vector<hipEvent_t> events(num_chunks);

    for (int i = 0; i < num_chunks; i++)
    {
        int row_start = i * N_LIMIT;

        int current_n =
            std::min(N_LIMIT, n - row_start);

        auto& ctx = pool->get(i);

        hipEventCreate(&events[i]);

        recursive_trsm_mixed_internal(
            ctx.handle,
            A,
            lda,
            B + row_start,
            ldb,
            current_n,
            m,
            nb);

        hipEventRecord(events[i], ctx.stream);
    }

    for (int i = 0; i < num_chunks; i++)
    {
        hipStreamWaitEvent(
            orig_stream,
            events[i],
            0);
    }

    for (int i = 0; i < num_chunks; i++)
    {
        hipEventDestroy(events[i]);
    }
}

} // extern "C"