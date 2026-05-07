#include "syrk/batched_syrk.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

extern "C" {

// ==============================================================================
// 单精度实现
// ==============================================================================
void batched_syrk_bottom_up(rocblas_handle handle,
                            float* A, int lda,
                            float* C, int ldc,
                            int n, int k, int nb) {
    if (n <= 0 || k <= 0) return;

    // 规模极小时直接调用原生 SYRK
    if (n <= nb) {
        float alpha = -1.0f;
        float beta  = 1.0f;
        rocblas_ssyrk(handle, rocblas_fill_lower, rocblas_operation_none,
                      n, k, &alpha, A, lda, &beta, C, ldc);
        return;
    }

    // 在 Cholesky Right-looking 分解中，SYRK 通常是 C = C - A * A^T
    float alpha_gemm = -1.0f;
    float beta_gemm  = 1.0f;

    // -------------------------------------------------------------------------
    // 阶段 1：对角线叶子节点初始化 (算法3的第3行)
    // 使用 BatchedGEMM 计算所有最基础的对角线方块 C_ii = C_ii - A_i * A_i^T
    // -------------------------------------------------------------------------
    int num_blocks = n / nb;
    if (num_blocks > 0) {
        // A 的步长：向下跳 nb 行
        // C 的步长：沿对角线向下跳 nb 行，向右跳 nb 列 -> offset = nb + nb * ldc
        rocblas_sgemm_strided_batched(
            handle,
            rocblas_operation_none,       // op(A)
            rocblas_operation_transpose,  // op(A^T)
            nb, nb, k,
            &alpha_gemm,
            A, lda, nb,                   // A_i
            A, lda, nb,                   // A_i^T
            &beta_gemm,
            C, ldc, nb + nb * ldc,        // C_ii
            num_blocks
        );
    }

    // [边界处理]：当 n 不能被 nb 整除时，处理对角线上最后一个非完整方块
    int tail = n % nb;
    if (tail > 0) {
        int offset = num_blocks * nb;
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      tail, tail, k,
                      &alpha_gemm,
                      A + offset, lda,                // 尾部 A 块
                      A + offset, lda,                // 尾部 A 块的转置
                      &beta_gemm,
                      C + offset + offset * ldc, ldc); // 尾部对角线 C 块
    }

    // -------------------------------------------------------------------------
    // 阶段 2：自底向上规约合并
    // 利用 BatchedGEMM 将相邻的分块进行交叉乘积累加
    // -------------------------------------------------------------------------
    // i 控制当前处理块的规模（倍数）。当需要处理的剩余块数或边界跨度符合时，进入循环。
    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2) {
        int S = i * nb; // 当前层级单块的尺寸
        int batch_count = n / (2 * S); // 可以形成标准对拼的满格块对数

        // 2.1 批量处理标准的满格跨块交叉相乘
        if (batch_count > 0) {
            // 计算 C_{2m+1, 2m} = C_{2m+1, 2m} - A_{bottom} * A_{top}^T
            rocblas_sgemm_strided_batched(
                handle,
                rocblas_operation_none,       // A_bottom
                rocblas_operation_transpose,  // A_top 转置
                S, S, k,
                &alpha_gemm,
                A + S, lda, 2 * S,            // A_bottom, 步长跨过一整个大块 (2S)
                A, lda, 2 * S,                // A_top,    步长跨过一整个大块 (2S)
                &beta_gemm,
                C + S, ldc, 2 * S + 2 * S * ldc, // 目标位置：偏下 S 行。步长：偏下 2S 行，偏右 2S 列
                batch_count
            );
        }

        // 2.2 [边界处理]：处理无法凑满 2*S 的“落单边角料”
        int rem = n % (2 * S);
        if (rem > S) {
            // 这意味着上面的 top 块是完整的（大小为 S）
            // 但下面的 bottom 块只有残缺的尺寸 (rem - S)
            int offset = batch_count * 2 * S;
            int tail_M = rem - S; // bottom 块的实际行数
            int tail_N = S;       // top 块的实际行数(也就是转置后的列数)
            
            rocblas_sgemm(handle, 
                          rocblas_operation_none, rocblas_operation_transpose,
                          tail_M, tail_N, k,
                          &alpha_gemm,
                          A + offset + S, lda,            // 残缺的 bottom 块
                          A + offset, lda,                // 完整的 top 块
                          &beta_gemm,
                          C + (offset + S) + offset * ldc, ldc); // 定位到对应的子矩阵位置
        }
    }
}


// ==============================================================================
// 双精度实现
// ==============================================================================
void batched_syrk_bottom_up_double(rocblas_handle handle,
                                   double* A, int lda,
                                   double* C, int ldc,
                                   int n, int k, int nb) {
    if (n <= 0 || k <= 0) return;

    if (n <= nb) {
        double alpha = -1.0;
        double beta  = 1.0;
        rocblas_dsyrk(handle, rocblas_fill_lower, rocblas_operation_none,
                      n, k, &alpha, A, lda, &beta, C, ldc);
        return;
    }

    double alpha_gemm = -1.0;
    double beta_gemm  = 1.0;

    // 阶段 1
    int num_blocks = n / nb;
    if (num_blocks > 0) {
        rocblas_dgemm_strided_batched(
            handle,
            rocblas_operation_none, rocblas_operation_transpose,
            nb, nb, k,
            &alpha_gemm,
            A, lda, nb,
            A, lda, nb,
            &beta_gemm,
            C, ldc, nb + nb * ldc,
            num_blocks
        );
    }

    int tail = n % nb;
    if (tail > 0) {
        int offset = num_blocks * nb;
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      tail, tail, k,
                      &alpha_gemm,
                      A + offset, lda,
                      A + offset, lda,
                      &beta_gemm,
                      C + offset + offset * ldc, ldc);
    }

    // 阶段 2
    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2) {
        int S = i * nb;
        int batch_count = n / (2 * S);

        if (batch_count > 0) {
            rocblas_dgemm_strided_batched(
                handle,
                rocblas_operation_none, rocblas_operation_transpose,
                S, S, k,
                &alpha_gemm,
                A + S, lda, 2 * S,
                A, lda, 2 * S,
                &beta_gemm,
                C + S, ldc, 2 * S + 2 * S * ldc,
                batch_count
            );
        }

        int rem = n % (2 * S);
        if (rem > S) {
            int offset = batch_count * 2 * S;
            int tail_M = rem - S;
            int tail_N = S;
            
            rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                          tail_M, tail_N, k,
                          &alpha_gemm,
                          A + offset + S, lda,
                          A + offset, lda,
                          &beta_gemm,
                          C + (offset + S) + offset * ldc, ldc);
        }
    }
}

} // extern "C"