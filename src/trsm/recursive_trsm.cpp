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
// 底层回退：小规模 FP16 TRSM 转用 FP32 求解
// [优化]：通过参数直接传入预分配的全局临时 Workspace，消除显存分配开销
// ======================================================
void custom_fallback_htrsm(rocblas_handle handle,
                           rocblas_side side, rocblas_fill uplo,
                           rocblas_operation trans, rocblas_diagonal diag,
                           int n, int m, float alpha,
                           rocblas_half* A, int lda,
                           rocblas_half* B, int ldb,
                           float* temp_A, float* temp_B) 
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // 直接使用传入的 Workspace，消除了原本在此处的 hipMallocAsync/hipFreeAsync
    launch_fp16_to_fp32((const __half*)A, temp_A, m, m, lda, m, stream);
    launch_fp16_to_fp32((const __half*)B, temp_B, n, m, ldb, n, stream);

    rocblas_strsm(handle, side, uplo, trans, diag,
                  n, m, &alpha,
                  temp_A, m,
                  temp_B, n);

    launch_fp32_to_fp16(temp_B, (__half*)B, n, m, n, ldb, stream);
}


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
// 内部纯递归执行器 (半精度 FP16)
// [优化]：参数链向下传递预分配的 Workspace 指针
// ======================================================
void recursive_trsm_half_internal(rocblas_handle handle,
                                  rocblas_half* A, int lda,
                                  rocblas_half* B, int ldb,
                                  int n, int m, int nb,
                                  float* temp_A, float* temp_B) 
{
    if (n <= 0 || m <= 0) return;

    if (m <= nb) {
        float alpha = 1.0f;
        custom_fallback_htrsm(handle,
                              rocblas_side_right,          
                              rocblas_fill_lower,          
                              rocblas_operation_transpose, 
                              rocblas_diagonal_non_unit,
                              n, m, alpha,
                              A, lda,
                              B, ldb,
                              temp_A, temp_B);
        return;
    }

    int m1 = m / 2;
    int m2 = m - m1;

    recursive_trsm_half_internal(handle, A, lda, B, ldb, n, m1, nb, temp_A, temp_B);

    float alpha_gemm = -1.0f;
    float beta_gemm  = 1.0f;
    rocblas_gemm_ex(handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose, 
                    n, m2, m1,
                    &alpha_gemm,
                    B, rocblas_datatype_f16_r, ldb,                      
                    A + m1, rocblas_datatype_f16_r, lda,                 
                    &beta_gemm,
                    B + (size_t)m1 * ldb, rocblas_datatype_f16_r, ldb,
                    B + (size_t)m1 * ldb, rocblas_datatype_f16_r, ldb,
                    rocblas_datatype_f32_r,
                    rocblas_gemm_algo_standard,
                    0, 0);  

    recursive_trsm_half_internal(handle, A + (size_t)m1 * lda + m1, lda, B + (size_t)m1 * ldb, ldb, n, m2, nb, temp_A, temp_B);
}

// ======================================================
// Public API (float)
// [优化]：移除多流合并中导致 CPU 挂起阻塞的 hipStreamSynchronize
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

        if (i > 0) {
            HIP_CHECK(hipEventCreate(&events[i]));
        }

        recursive_trsm_internal(
            ctx.handle,
            A, lda,
            B + row_start, ldb,
            current_n, m, nb);

        if (i > 0) {
            HIP_CHECK(hipEventRecord(events[i], ctx.stream));
        }
    }

    // 仅通过 GPU 事件图依赖进行流同步，避免任何主机端阻塞
    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    // 延迟安全销毁事件
    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

// ======================================================
// Public API (double)
// [优化]：同步行为与 float API 保持一致，移除 hipStreamSynchronize
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

        if (i > 0) {
            HIP_CHECK(hipEventCreate(&events[i]));
        }

        recursive_trsm_double_internal(
            ctx.handle,
            A, lda,
            B + row_start, ldb,
            current_n, m, nb);

        if (i > 0) {
            HIP_CHECK(hipEventRecord(events[i], ctx.stream));
        }
    }

    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

// ======================================================
// Public API (half)
// [整体优化]：
// 1. 将显存分配提升至最外层（仅在各分块流开始处分配一次，随后递归复用）
// 2. 利用 hipFreeAsync 在流队列末尾异步回收，彻底避免 Host/Device 阻塞
// 3. 移除多流合并阶段的主机端 hipStreamSynchronize
// ======================================================
void recursive_trsm_half(
    rocblas_handle handle,
    StreamPool* pool,
    rocblas_half* A, int lda,
    rocblas_half* B, int ldb,
    int n, int m, int nb)
{
    const int N_LIMIT = 8192; 

    if (n <= N_LIMIT) {
        hipStream_t stream;
        ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

        // 单流无分块时，在最外层统一申请一次 Workspace 并在最后释放
        float *temp_A = nullptr;
        float *temp_B = nullptr;
        HIP_CHECK(hipMallocAsync((void**)&temp_A, nb * nb * sizeof(float), stream));
        HIP_CHECK(hipMallocAsync((void**)&temp_B, n * nb * sizeof(float), stream));

        recursive_trsm_half_internal(handle, A, lda, B, ldb, n, m, nb, temp_A, temp_B);

        HIP_CHECK(hipFreeAsync(temp_A, stream));
        HIP_CHECK(hipFreeAsync(temp_B, stream));
        return;
    }

    hipStream_t orig_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &orig_stream));

    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;

    std::vector<hipEvent_t> events(num_chunks, nullptr);
    std::vector<float*> temp_A_list(num_chunks, nullptr);
    std::vector<float*> temp_B_list(num_chunks, nullptr);

    for (int i = 0; i < num_chunks; i++) {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);
        auto &ctx = pool->get(i);

        if (i > 0) {
            HIP_CHECK(hipEventCreate(&events[i]));
        }

        // 1. 在每个分块计算流中各预分配一次 Workspace
        HIP_CHECK(hipMallocAsync((void**)&temp_A_list[i], nb * nb * sizeof(float), ctx.stream));
        HIP_CHECK(hipMallocAsync((void**)&temp_B_list[i], current_n * nb * sizeof(float), ctx.stream));

        // 2. 传入 Workspace 进行无内部申请的纯递归执行
        recursive_trsm_half_internal(
            ctx.handle,
            A, lda,
            B + row_start, ldb,
            current_n, m, nb,
            temp_A_list[i], temp_B_list[i]);

        // 3. 异步释放 Workspace（排队在当前计算流的末尾，完全不阻塞 CPU）
        HIP_CHECK(hipFreeAsync(temp_A_list[i], ctx.stream));
        HIP_CHECK(hipFreeAsync(temp_B_list[i], ctx.stream));

        if (i > 0) {
            HIP_CHECK(hipEventRecord(events[i], ctx.stream));
        }
    }

    // 4. 仅依靠 GPU 事件链在主流上进行非阻塞合并
    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipStreamWaitEvent(orig_stream, events[i], 0));
    }

    // 5. 销毁事件
    for (int i = 1; i < num_chunks; i++) {
        HIP_CHECK(hipEventDestroy(events[i]));
    }
}

void recursive_trsm_half_f32_wrapper(
    rocblas_handle handle,
    StreamPool* pool,
    float* A_f32, int lda,
    float* B_f32, int ldb,
    int n, int m, int nb)
{
    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    size_t size_A_half = static_cast<size_t>(m) * m * sizeof(rocblas_half);
    size_t size_B_half = static_cast<size_t>(n) * m * sizeof(rocblas_half);

    rocblas_half* d_A_half = nullptr;
    rocblas_half* d_B_half = nullptr;

    // 1. 异步分配临时 FP16 显存空间
    HIP_CHECK(hipMallocAsync((void**)&d_A_half, size_A_half, stream));
    HIP_CHECK(hipMallocAsync((void**)&d_B_half, size_B_half, stream));

    // 2. 在 GPU 端全异步执行 FP32 到 FP16 的转换（不涉及 Host/Device 数据传输）
    launch_fp32_to_fp16(A_f32, reinterpret_cast<__half*>(d_A_half), m, m, lda, m, stream);
    launch_fp32_to_fp16(B_f32, reinterpret_cast<__half*>(d_B_half), n, m, ldb, n, stream);

    // 3. 执行优化后的半精度递归 TRSM (其内部将自动递归分治并使用 Tensor Cores 进行加速计算)
    recursive_trsm_half(handle, pool, d_A_half, lda, d_B_half, ldb, n, m, nb);

    // 4. 将计算完成后的半精度结果全异步转回 FP32，并直接写回传入的 B_f32 矩阵中
    launch_fp16_to_fp32(reinterpret_cast<const __half*>(d_B_half), B_f32, n, m, n, ldb, stream);

    // 5. 异步释放临时的 FP16 空间
    HIP_CHECK(hipFreeAsync(d_A_half, stream));
    HIP_CHECK(hipFreeAsync(d_B_half, stream));
}

} // extern "C"