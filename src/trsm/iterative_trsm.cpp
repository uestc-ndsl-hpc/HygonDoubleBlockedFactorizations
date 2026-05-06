#include "trsm/iterative_trsm.h"
#include <hip/hip_runtime.h>

extern "C" {

void iterative_trsm_right_looking(rocblas_handle handle,
                                   float* A, int lda,
                                   float* B, int ldb,
                                   int n, int m, int nb) {
    if (n <= 0 || m <= 0) return;

    float one = 1.0f;
    float neg_one = -1.0f;

    for (int i = 0; i < n; i += nb) {
        // 1. 确定当前块的大小
        int current_nb = (n - i < nb) ? (n - i) : nb;

        // 2. 求解当前的对角线子问题：L_ii * X_i = B_i
        // 此时 B_i 已经被之前的步骤更新过了
        rocblas_strsm(handle,
                      rocblas_side_left,
                      rocblas_fill_lower,
                      rocblas_operation_none,
                      rocblas_diagonal_non_unit,
                      current_nb, m, &one,
                      A + i * lda + i, lda, 
                      B + i, ldb);

        // 3. 右视更新：使用求得的 X_i 更新剩余的 B 块
        // B_remaining = B_remaining - L_remaining_i * X_i
        if (i + current_nb < n) {
            int remaining_rows = n - (i + current_nb);
            
            // 这一步将 TRSM 的大部分计算量转化为 GEMM
            // 从而能够利用 GPU 的 Tensor Cores 或 Matrix Cores 提升吞吐量
            rocblas_sgemm(handle,
                          rocblas_operation_none, 
                          rocblas_operation_none,
                          remaining_rows, m, current_nb,
                          &neg_one,
                          A + i * lda + (i + current_nb), lda, // L_remaining_i
                          B + i, ldb,                          // X_i
                          &one,
                          B + (i + current_nb), ldb);         // B_remaining
        }
    }
}

}