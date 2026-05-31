#include "syrk/batched_syrk.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include "gemm/mixed_gemm.h"
#include "utils/error_check.h"

extern "C" {

// ============================================================================
// 辅助同步函数：等待 StreamPool 中所有流完成当前层的发射（论文层间同步要求）
// ============================================================================
static inline void sync_stream_pool(StreamPool* pool) {
    if (!pool) return;
    for (int i = 0; i < pool->size(); ++i) {
        HIP_CHECK(hipStreamSynchronize(pool->get(i).stream));
    }
}

// ============================================================================
// 单精度 Bottom-Up SYRK (严格论文基准版)
// ============================================================================
void batched_syrk_bottom_up(
    rocblas_handle base_handle,
    StreamPool* pool,
    float* A,
    int lda,
    float* C,
    int ldc,
    int n,
    int k,
    int nb)
{
    if (n <= 0 || k <= 0) return;

    // 边界条件：若总体尺寸小于分块，退化为单次标准 ssyrk
    if (n <= nb)
    {
        float alpha = -1.0f;
        float beta  = 1.0f;
        rocblas_ssyrk(base_handle, rocblas_fill_lower, rocblas_operation_none,
                      n, k, &alpha, A, lda, &beta, C, ldc);
        return;
    }

    float alpha = -1.0f;
    float beta  = 1.0f;

    // -------------------------------------------------------------------------
    // 阶段1：Diagonal Leaf Updates (对角线小叶子块并行更新)
    // -------------------------------------------------------------------------
    int num_blocks = n / nb;

    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;
        auto& ctx = pool->get(b); // 轮询分发至硬件流

        // 论文思路：直接调用单次 GEMM 更新对角线小块，不进行内部的 K 轴切分
        rocblas_sgemm(
            ctx.handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            nb, nb, k,
            &alpha,
            A + offset, lda,
            A + offset, lda,
            &beta,
            C + offset + offset * ldc, ldc);
    }

    // 独立处理对角线尾部残差块
    int tail = n % nb;
    if (tail > 0)
    {
        int offset = num_blocks * nb;
        auto& ctx = pool->get(num_blocks);

        rocblas_sgemm(
            ctx.handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            tail, tail, k,
            &alpha,
            A + offset, lda,
            A + offset, lda,
            &beta,
            C + offset + offset * ldc, ldc);
    }

    // 论文要求：第一阶段对角线元素必须全部落盘，才能开始计算依赖它们的非对角线矩形
    sync_stream_pool(pool);

    // -------------------------------------------------------------------------
    // 阶段2：Bottom-Up Merge (自底向上非对角线矩形层层合并)
    // -------------------------------------------------------------------------
    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2)
    {
        int S = i * nb; // 当前层合并的跨度
        int batch_count = n / (2 * S);

        // 同一层内的非对角线块互不影响，由多流并发发射更新
        for (int b = 0; b < batch_count; b++)
        {
            int offset = b * 2 * S;
            auto& ctx = pool->get(b); 

            rocblas_sgemm(
                ctx.handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                S, S, k,
                &alpha,
                A + offset + S, lda,
                A + offset, lda,
                &beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        // 处理该层残差块的合并
        int rem = n % (2 * S);
        if (rem > S)
        {
            int offset = batch_count * 2 * S;
            int tail_M = rem - S;
            int tail_N = S;
            auto& ctx = pool->get(batch_count);

            rocblas_sgemm(
                ctx.handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                tail_M, tail_N, k,
                &alpha,
                A + offset + S, lda,
                A + offset, lda,
                &beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        // 论文的核心依赖约束：每一层循环是一个时间步，上层的计算依赖下层的结果，必须强制同步
        sync_stream_pool(pool);
    }
}


// ============================================================================
// 双精度 Bottom-Up SYRK (严格论文基准版)
// ============================================================================
void batched_syrk_bottom_up_double(
    rocblas_handle base_handle,
    StreamPool* pool,
    double* A,
    int lda,
    double* C,
    int ldc,
    int n,
    int k,
    int nb)
{
    if (n <= 0 || k <= 0) return;

    if (n <= nb)
    {
        double alpha = -1.0;
        double beta  = 1.0;
        rocblas_dsyrk(base_handle, rocblas_fill_lower, rocblas_operation_none,
                      n, k, &alpha, A, lda, &beta, C, ldc);
        return;
    }

    double alpha = -1.0;
    double beta  = 1.0;

    // 阶段1：对角线叶节点更新
    int num_blocks = n / nb;
    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;
        auto& ctx = pool->get(b);

        rocblas_dgemm(
            ctx.handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            nb, nb, k,
            &alpha,
            A + offset, lda,
            A + offset, lda,
            &beta,
            C + offset + offset * ldc, ldc);
    }

    int tail = n % nb;
    if (tail > 0)
    {
        int offset = num_blocks * nb;
        auto& ctx = pool->get(num_blocks);

        rocblas_dgemm(
            ctx.handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            tail, tail, k,
            &alpha,
            A + offset, lda,
            A + offset, lda,
            &beta,
            C + offset + offset * ldc, ldc);
    }

    sync_stream_pool(pool);

    // 阶段2：自底向上层层合并
    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2)
    {
        int S = i * nb;
        int batch_count = n / (2 * S);

        for (int b = 0; b < batch_count; b++)
        {
            int offset = b * 2 * S;
            auto& ctx = pool->get(b);

            rocblas_dgemm(
                ctx.handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                S, S, k,
                &alpha,
                A + offset + S, lda,
                A + offset, lda,
                &beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        int rem = n % (2 * S);
        if (rem > S)
        {
            int offset = batch_count * 2 * S;
            int tail_M = rem - S;
            int tail_N = S;
            auto& ctx = pool->get(batch_count);

            rocblas_dgemm(
                ctx.handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                tail_M, tail_N, k,
                &alpha,
                A + offset + S, lda,
                A + offset, lda,
                &beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        sync_stream_pool(pool);
    }
}


// ============================================================================
// 混合精度/半精度 Bottom-Up SYRK (严格论文基准版)
// ============================================================================
void batched_syrk_bottom_up_mixed(
    rocblas_handle base_handle,
    StreamPool* pool,
    float* A,
    int lda,
    float* C,
    int ldc,
    int n,
    int k,
    int nb)
{
    if (n <= 0 || k <= 0) return;

    if (n <= nb)
    {
        float alpha = -1.0f;
        float beta  = 1.0f;
        rocblas_ssyrk(base_handle, rocblas_fill_lower, rocblas_operation_none,
                      n, k, &alpha, A, lda, &beta, C, ldc);
        return;
    }

    float alpha = -1.0f;
    float beta  = 1.0f;

    // 阶段1：对角线叶节点更新
    int num_blocks = n / nb;
    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;
        auto& ctx = pool->get(b); 

        launch_mixed_gemm(
            ctx.handle,
            ctx.stream,
            rocblas_operation_none,
            rocblas_operation_transpose,
            nb, nb, k,
            alpha,
            A + offset, lda,
            A + offset, lda,
            beta,
            C + offset + offset * ldc, ldc);
    }

    int tail = n % nb;
    if (tail > 0)
    {
        int offset = num_blocks * nb;
        auto& ctx = pool->get(num_blocks);

        launch_mixed_gemm(
            ctx.handle,
            ctx.stream,
            rocblas_operation_none,
            rocblas_operation_transpose,
            tail, tail, k,
            alpha,
            A + offset, lda,
            A + offset, lda,
            beta,
            C + offset + offset * ldc, ldc);
    }

    sync_stream_pool(pool);

    // 阶段2：自底向上层层合并
    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2)
    {
        int S = i * nb;
        int batch_count = n / (2 * S);

        for (int b = 0; b < batch_count; b++)
        {
            int offset = b * 2 * S;
            auto& ctx = pool->get(b);

            launch_mixed_gemm(
                ctx.handle,
                ctx.stream,
                rocblas_operation_none,
                rocblas_operation_transpose,
                S, S, k,
                alpha,
                A + offset + S, lda,
                A + offset, lda,
                beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        int rem = n % (2 * S);
        if (rem > S)
        {
            int offset = batch_count * 2 * S;
            int tail_M = rem - S;
            int tail_N = S;
            auto& ctx = pool->get(batch_count);

            launch_mixed_gemm(
                ctx.handle,
                ctx.stream,
                rocblas_operation_none,
                rocblas_operation_transpose,
                tail_M, tail_N, k,
                alpha,
                A + offset + S, lda,
                A + offset, lda,
                beta,
                C + (offset + S) + offset * ldc, ldc);
        }

        sync_stream_pool(pool);
    }
}

} // extern "C"