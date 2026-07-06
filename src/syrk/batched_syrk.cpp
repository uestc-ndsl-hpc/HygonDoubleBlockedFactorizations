#include "syrk/batched_syrk.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cstddef>

#include "gemm/mixed_gemm.h"
#include "utils/common/error_check.h"

extern "C" {

static constexpr int BATCH_THRESHOLD = 8;
static constexpr int MAX_BATCHED_S = 4096;
static constexpr int BK = 2048;

static inline rocblas_handle select_handle(rocblas_handle base_handle, StreamPool* pool, int task_id)
{
    return (pool && pool->size() > 0) ? pool->get(task_id).handle : base_handle;
}

static inline void wait_stream_pool_on_handle(rocblas_handle base_handle, StreamPool* pool)
{
    if (!pool || pool->size() <= 0) {
        return;
    }

    hipStream_t main_stream = nullptr;
    ROCBLAS_CHECK(rocblas_get_stream(base_handle, &main_stream));

    for (int i = 0; i < pool->size(); ++i) {
        hipStream_t stream = pool->get(i).stream;
        if (stream == main_stream) {
            continue;
        }

        hipEvent_t event;
        HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        HIP_CHECK(hipEventRecord(event, stream));
        HIP_CHECK(hipStreamWaitEvent(main_stream, event, 0));
        HIP_CHECK(hipEventDestroy(event));
    }
}

static inline void sync_stream_pool(StreamPool* pool)
{
    if (!pool) {
        return;
    }

    for (int i = 0; i < pool->size(); ++i) {
        HIP_CHECK(hipStreamSynchronize(pool->get(i).stream));
    }
}

static inline void syrk_update_float(
    rocblas_handle handle,
    float* A,
    int lda,
    float* C,
    int ldc,
    int n,
    int k)
{
    const float alpha = -1.0f;
    const float beta = 1.0f;

    ROCBLAS_CHECK(rocblas_ssyrk(
        handle,
        rocblas_fill_lower,
        rocblas_operation_none,
        n,
        k,
        &alpha,
        A,
        lda,
        &beta,
        C,
        ldc));
}

static inline void syrk_update_double(
    rocblas_handle handle,
    double* A,
    int lda,
    double* C,
    int ldc,
    int n,
    int k)
{
    const double alpha = -1.0;
    const double beta = 1.0;

    ROCBLAS_CHECK(rocblas_dsyrk(
        handle,
        rocblas_fill_lower,
        rocblas_operation_none,
        n,
        k,
        &alpha,
        A,
        lda,
        &beta,
        C,
        ldc));
}

static inline void gemm_update_float(
    rocblas_handle handle,
    float* A_bottom,
    float* A_top,
    float* C,
    int lda,
    int ldc,
    int m,
    int n,
    int k)
{
    const float alpha = -1.0f;
    const float beta = 1.0f;

    for (int kk = 0; kk < k; kk += BK) {
        int cur_k = std::min(BK, k - kk);
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            cur_k,
            &alpha,
            A_bottom + static_cast<std::size_t>(kk) * lda,
            lda,
            A_top + static_cast<std::size_t>(kk) * lda,
            lda,
            &beta,
            C,
            ldc));
    }
}

static inline void gemm_update_double(
    rocblas_handle handle,
    double* A_bottom,
    double* A_top,
    double* C,
    int lda,
    int ldc,
    int m,
    int n,
    int k)
{
    const double alpha = -1.0;
    const double beta = 1.0;

    for (int kk = 0; kk < k; kk += BK) {
        int cur_k = std::min(BK, k - kk);
        ROCBLAS_CHECK(rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            cur_k,
            &alpha,
            A_bottom + static_cast<std::size_t>(kk) * lda,
            lda,
            A_top + static_cast<std::size_t>(kk) * lda,
            lda,
            &beta,
            C,
            ldc));
    }
}

static inline void strided_batched_gemm_update_float(
    rocblas_handle handle,
    float* A,
    int lda,
    float* C,
    int ldc,
    int s,
    int k,
    int batch_count)
{
    const float alpha = -1.0f;
    const float beta = 1.0f;
    const rocblas_stride stride_a = static_cast<rocblas_stride>(2LL * s);
    const rocblas_stride stride_b = static_cast<rocblas_stride>(2LL * s);
    const rocblas_stride stride_c =
        static_cast<rocblas_stride>(2LL * s + static_cast<long long>(2 * s) * ldc);

    for (int kk = 0; kk < k; kk += BK) {
        int cur_k = std::min(BK, k - kk);
        ROCBLAS_CHECK(rocblas_sgemm_strided_batched(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            s,
            s,
            cur_k,
            &alpha,
            A + s + static_cast<std::size_t>(kk) * lda,
            lda,
            stride_a,
            A + static_cast<std::size_t>(kk) * lda,
            lda,
            stride_b,
            &beta,
            C + s,
            ldc,
            stride_c,
            batch_count));
    }
}

static inline void strided_batched_gemm_update_double(
    rocblas_handle handle,
    double* A,
    int lda,
    double* C,
    int ldc,
    int s,
    int k,
    int batch_count)
{
    const double alpha = -1.0;
    const double beta = 1.0;
    const rocblas_stride stride_a = static_cast<rocblas_stride>(2LL * s);
    const rocblas_stride stride_b = static_cast<rocblas_stride>(2LL * s);
    const rocblas_stride stride_c =
        static_cast<rocblas_stride>(2LL * s + static_cast<long long>(2 * s) * ldc);

    for (int kk = 0; kk < k; kk += BK) {
        int cur_k = std::min(BK, k - kk);
        ROCBLAS_CHECK(rocblas_dgemm_strided_batched(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            s,
            s,
            cur_k,
            &alpha,
            A + s + static_cast<std::size_t>(kk) * lda,
            lda,
            stride_a,
            A + static_cast<std::size_t>(kk) * lda,
            lda,
            stride_b,
            &beta,
            C + s,
            ldc,
            stride_c,
            batch_count));
    }
}

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
    if (n <= 0 || k <= 0) {
        return;
    }

    if (n <= nb) {
        syrk_update_float(base_handle, A, lda, C, ldc, n, k);
        return;
    }

    int task_id = 0;
    int num_blocks = n / nb;

    for (int b = 0; b < num_blocks; ++b) {
        int offset = b * nb;
        syrk_update_float(
            select_handle(base_handle, pool, task_id++),
            A + offset,
            lda,
            C + offset + static_cast<std::size_t>(offset) * ldc,
            ldc,
            nb,
            k);
    }

    int tail = n % nb;
    if (tail > 0) {
        int offset = num_blocks * nb;
        syrk_update_float(
            select_handle(base_handle, pool, task_id++),
            A + offset,
            lda,
            C + offset + static_cast<std::size_t>(offset) * ldc,
            ldc,
            tail,
            k);
    }

    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2) {
        int s = i * nb;
        int batch_count = n / (2 * s);

        if (batch_count >= BATCH_THRESHOLD && s <= MAX_BATCHED_S) {
            strided_batched_gemm_update_float(base_handle, A, lda, C, ldc, s, k, batch_count);
        } else {
            for (int b = 0; b < batch_count; ++b) {
                int offset = b * 2 * s;
                gemm_update_float(
                    select_handle(base_handle, pool, task_id++),
                    A + offset + s,
                    A + offset,
                    C + (offset + s) + static_cast<std::size_t>(offset) * ldc,
                    lda,
                    ldc,
                    s,
                    s,
                    k);
            }
        }

        int rem = n % (2 * s);
        if (rem > s) {
            int offset = batch_count * 2 * s;
            gemm_update_float(
                select_handle(base_handle, pool, task_id++),
                A + offset + s,
                A + offset,
                C + (offset + s) + static_cast<std::size_t>(offset) * ldc,
                lda,
                ldc,
                rem - s,
                s,
                k);
        }
    }

    wait_stream_pool_on_handle(base_handle, pool);
}

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
    if (n <= 0 || k <= 0) {
        return;
    }

    if (n <= nb) {
        syrk_update_double(base_handle, A, lda, C, ldc, n, k);
        return;
    }

    int task_id = 0;
    int num_blocks = n / nb;

    for (int b = 0; b < num_blocks; ++b) {
        int offset = b * nb;
        syrk_update_double(
            select_handle(base_handle, pool, task_id++),
            A + offset,
            lda,
            C + offset + static_cast<std::size_t>(offset) * ldc,
            ldc,
            nb,
            k);
    }

    int tail = n % nb;
    if (tail > 0) {
        int offset = num_blocks * nb;
        syrk_update_double(
            select_handle(base_handle, pool, task_id++),
            A + offset,
            lda,
            C + offset + static_cast<std::size_t>(offset) * ldc,
            ldc,
            tail,
            k);
    }

    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2) {
        int s = i * nb;
        int batch_count = n / (2 * s);

        if (batch_count >= BATCH_THRESHOLD && s <= MAX_BATCHED_S) {
            strided_batched_gemm_update_double(base_handle, A, lda, C, ldc, s, k, batch_count);
        } else {
            for (int b = 0; b < batch_count; ++b) {
                int offset = b * 2 * s;
                gemm_update_double(
                    select_handle(base_handle, pool, task_id++),
                    A + offset + s,
                    A + offset,
                    C + (offset + s) + static_cast<std::size_t>(offset) * ldc,
                    lda,
                    ldc,
                    s,
                    s,
                    k);
            }
        }

        int rem = n % (2 * s);
        if (rem > s) {
            int offset = batch_count * 2 * s;
            gemm_update_double(
                select_handle(base_handle, pool, task_id++),
                A + offset + s,
                A + offset,
                C + (offset + s) + static_cast<std::size_t>(offset) * ldc,
                lda,
                ldc,
                rem - s,
                s,
                k);
        }
    }

    wait_stream_pool_on_handle(base_handle, pool);
}

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

    int num_blocks = n / nb;
    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;
        auto& ctx = pool->get(b);

        float* C_block = C + offset + (size_t)offset * (size_t)ldc;

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
            C_block, ldc);
    }

    int tail = n % nb;
    if (tail > 0)
    {
        int offset = num_blocks * nb;
        auto& ctx = pool->get(num_blocks);

        float* C_block = C + offset + (size_t)offset * (size_t)ldc;

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
            C_block, ldc);
    }

    sync_stream_pool(pool);

    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2)
    {
        int S = i * nb;
        int batch_count = n / (2 * S);

        for (int b = 0; b < batch_count; b++)
        {
            int offset = b * 2 * S;
            auto& ctx = pool->get(b);

            float* C_block = C + (offset + S) + (size_t)offset * (size_t)ldc;

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
                C_block, ldc);
        }

        int rem = n % (2 * S);
        if (rem > S)
        {
            int offset = batch_count * 2 * S;
            int tail_M = rem - S;
            int tail_N = S;
            auto& ctx = pool->get(batch_count);

            float* C_block = C + (offset + S) + (size_t)offset * (size_t)ldc;

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
                C_block, ldc);
        }

        sync_stream_pool(pool);
    }
}

} // extern "C"
