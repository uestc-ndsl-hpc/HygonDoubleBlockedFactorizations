#include "trsm/recursive_trsm.h"
#include "utils/error_check.h"
#include "utils/stream_pool.h"
#include "kernels/fp16_convert.h"
#include "utils/mixed_gemm.h"
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

// ======================================================
// 内部纯递归执行器 (混合精度)
// ======================================================
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
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    if (m <= nb)
    {
        float alpha = 1.0f;

        ROCBLAS_CHECK(rocblas_strsm(
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
            ldb));

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

    launch_mixed_gemm(
        handle,
        stream,
        rocblas_operation_none,       // B1 不转置
        rocblas_operation_transpose,  // A21 转置
        n,                            // GEMM M 维
        m2,                           // GEMM N 维
        m1,                           // GEMM K 维
        -1.0f,                        // alpha = -1.0
        B, ldb,                       // 对应矩阵 A_gemm (即 B1)
        A + m1, lda,                  // 对应矩阵 B_gemm (即 A21)
        1.0f,                         // beta = 1.0
        B + (size_t)m1 * ldb, ldb     // 对应输出矩阵 C_gemm (即 B2)
    );

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

// ======================================================
// Public API (mixed)
// ======================================================
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
    ROCBLAS_CHECK(rocblas_get_stream(handle, &orig_stream));

    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;

    std::vector<hipEvent_t> events(num_chunks);

    for (int i = 0; i < num_chunks; i++)
    {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);

        auto& ctx = pool->get(i);

        HIP_CHECK(hipEventCreate(&events[i]));

        recursive_trsm_mixed_internal(
            ctx.handle,
            A,
            lda,
            B + row_start,
            ldb,
            current_n,
            m,
            nb);

        HIP_CHECK(hipEventRecord(events[i], ctx.stream));
    }

    for (int i = 0; i < num_chunks; i++)
    {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    for (int i = 0; i < num_chunks; i++)
    {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

} // extern "C"