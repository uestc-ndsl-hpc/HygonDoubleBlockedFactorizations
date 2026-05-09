#include "syrk/batched_syrk.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <algorithm>

extern "C" {

// ============================================================================
// 核心思想
// ----------------------------------------------------------------------------
// 1. 用 GEMM 替代 SYRK
// 2. Bottom-Up Merge
// 3. 避免 strided_batched 在小 batch 下性能崩塌
// 4. Split-K Blocking
// 5. Large-S fallback
//
// 核心优化：
// ----------------------------------------------------------------------------
// batch_count >= 8
//     -> strided_batched GEMM
//
// batch_count < 8
//     -> 普通 GEMM loop
//
// K 很大
//     -> Split-K Blocking
//
// S 很大
//     -> 禁止 batched
// ============================================================================


// ============================================================================
// 参数调优
// ============================================================================

static constexpr int BATCH_THRESHOLD = 8;

// 超过该尺寸后，禁止继续 batched
static constexpr int MAX_BATCHED_S = 4096;

// Split-K Blocking
static constexpr int BK = 2048;


// ============================================================================
// 单次 GEMM Update
// C -= A_bottom * A_top^T
// ============================================================================

static inline void gemm_update_float(
    rocblas_handle handle,
    float* A_bottom,
    float* A_top,
    float* C,
    int lda,
    int ldc,
    int M,
    int N,
    int K)
{
    float alpha = -1.0f;
    float beta  = 1.0f;

    // ------------------------------------------------------------------------
    // Split-K Blocking
    // ------------------------------------------------------------------------

    bool first = true;

    for (int kk = 0; kk < K; kk += BK)
    {
        int cur_k = std::min(BK, K - kk);

        float cur_beta = first ? beta : 1.0f;

        rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            M,
            N,
            cur_k,
            &alpha,
            A_bottom + kk * lda,
            lda,
            A_top + kk * lda,
            lda,
            &cur_beta,
            C,
            ldc);

        first = false;
    }
}


// ============================================================================
// 双精度 GEMM Update
// ============================================================================

static inline void gemm_update_double(
    rocblas_handle handle,
    double* A_bottom,
    double* A_top,
    double* C,
    int lda,
    int ldc,
    int M,
    int N,
    int K)
{
    double alpha = -1.0;
    double beta  = 1.0;

    bool first = true;

    for (int kk = 0; kk < K; kk += BK)
    {
        int cur_k = std::min(BK, K - kk);

        double cur_beta = first ? beta : 1.0;

        rocblas_dgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            M,
            N,
            cur_k,
            &alpha,
            A_bottom + kk * lda,
            lda,
            A_top + kk * lda,
            lda,
            &cur_beta,
            C,
            ldc);

        first = false;
    }
}


// ============================================================================
// 单精度 Bottom-Up Recursive SYRK
// ============================================================================

void batched_syrk_bottom_up(
    rocblas_handle handle,
    float* A,
    int lda,
    float* C,
    int ldc,
    int n,
    int k,
    int nb)
{
    if (n <= 0 || k <= 0)
        return;

    // =========================================================================
    // 小矩阵直接 SYRK
    // =========================================================================

    if (n <= nb)
    {
        float alpha = -1.0f;
        float beta  = 1.0f;

        rocblas_ssyrk(
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
            ldc);

        return;
    }

    // =========================================================================
    // 阶段1：Diagonal Leaf Updates
    // =========================================================================

    int num_blocks = n / nb;

    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;

        gemm_update_float(
            handle,
            A + offset,
            A + offset,
            C + offset + offset * ldc,
            lda,
            ldc,
            nb,
            nb,
            k);
    }

    // Tail
    int tail = n % nb;

    if (tail > 0)
    {
        int offset = num_blocks * nb;

        gemm_update_float(
            handle,
            A + offset,
            A + offset,
            C + offset + offset * ldc,
            lda,
            ldc,
            tail,
            tail,
            k);
    }

    // =========================================================================
    // 阶段2：Bottom-Up Merge
    // =========================================================================

    for (int i = 1;
         (n / (2 * i * nb) > 0) ||
         (n % (2 * i * nb) > i * nb);
         i *= 2)
    {
        int S = i * nb;

        int batch_count = n / (2 * S);

        // =====================================================================
        // Case 1:
        // Large batch -> Strided Batched GEMM
        // =====================================================================

        if (batch_count >= BATCH_THRESHOLD &&
            S <= MAX_BATCHED_S)
        {
            for (int kk = 0; kk < k; kk += BK)
            {
                int cur_k = std::min(BK, k - kk);

                float alpha = -1.0f;
                float beta  = (kk == 0) ? 1.0f : 1.0f;

                rocblas_sgemm_strided_batched(
                    handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    S,
                    S,
                    cur_k,
                    &alpha,

                    A + S + kk * lda,
                    lda,
                    2 * S,

                    A + kk * lda,
                    lda,
                    2 * S,

                    &beta,

                    C + S,
                    ldc,
                    2 * S + 2 * S * ldc,

                    batch_count);
            }
        }

        // =====================================================================
        // Case 2:
        // Small batch OR large S
        // -> fallback to regular GEMM loop
        // =====================================================================

        else
        {
            for (int b = 0; b < batch_count; b++)
            {
                int offset = b * 2 * S;

                gemm_update_float(
                    handle,
                    A + offset + S,
                    A + offset,
                    C + (offset + S) + offset * ldc,
                    lda,
                    ldc,
                    S,
                    S,
                    k);
            }
        }

        // =====================================================================
        // Tail Processing
        // =====================================================================

        int rem = n % (2 * S);

        if (rem > S)
        {
            int offset = batch_count * 2 * S;

            int tail_M = rem - S;
            int tail_N = S;

            gemm_update_float(
                handle,
                A + offset + S,
                A + offset,
                C + (offset + S) + offset * ldc,
                lda,
                ldc,
                tail_M,
                tail_N,
                k);
        }
    }
}


// ============================================================================
// 双精度版本
// ============================================================================

void batched_syrk_bottom_up_double(
    rocblas_handle handle,
    double* A,
    int lda,
    double* C,
    int ldc,
    int n,
    int k,
    int nb)
{
    if (n <= 0 || k <= 0)
        return;

    if (n <= nb)
    {
        double alpha = -1.0;
        double beta  = 1.0;

        rocblas_dsyrk(
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
            ldc);

        return;
    }

    int num_blocks = n / nb;

    for (int b = 0; b < num_blocks; b++)
    {
        int offset = b * nb;

        gemm_update_double(
            handle,
            A + offset,
            A + offset,
            C + offset + offset * ldc,
            lda,
            ldc,
            nb,
            nb,
            k);
    }

    int tail = n % nb;

    if (tail > 0)
    {
        int offset = num_blocks * nb;

        gemm_update_double(
            handle,
            A + offset,
            A + offset,
            C + offset + offset * ldc,
            lda,
            ldc,
            tail,
            tail,
            k);
    }

    for (int i = 1;
         (n / (2 * i * nb) > 0) ||
         (n % (2 * i * nb) > i * nb);
         i *= 2)
    {
        int S = i * nb;

        int batch_count = n / (2 * S);

        if (batch_count >= BATCH_THRESHOLD &&
            S <= MAX_BATCHED_S)
        {
            for (int kk = 0; kk < k; kk += BK)
            {
                int cur_k = std::min(BK, k - kk);

                double alpha = -1.0;
                double beta  = (kk == 0) ? 1.0 : 1.0;

                rocblas_dgemm_strided_batched(
                    handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    S,
                    S,
                    cur_k,
                    &alpha,

                    A + S + kk * lda,
                    lda,
                    2 * S,

                    A + kk * lda,
                    lda,
                    2 * S,

                    &beta,

                    C + S,
                    ldc,
                    2 * S + 2 * S * ldc,

                    batch_count);
            }
        }
        else
        {
            for (int b = 0; b < batch_count; b++)
            {
                int offset = b * 2 * S;

                gemm_update_double(
                    handle,
                    A + offset + S,
                    A + offset,
                    C + (offset + S) + offset * ldc,
                    lda,
                    ldc,
                    S,
                    S,
                    k);
            }
        }

        int rem = n % (2 * S);

        if (rem > S)
        {
            int offset = batch_count * 2 * S;

            int tail_M = rem - S;
            int tail_N = S;

            gemm_update_double(
                handle,
                A + offset + S,
                A + offset,
                C + (offset + S) + offset * ldc,
                lda,
                ldc,
                tail_M,
                tail_N,
                k);
        }
    }
}

} // extern "C"