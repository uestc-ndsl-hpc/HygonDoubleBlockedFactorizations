#include "trsm/recursive_trsm.h"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <algorithm>

extern "C" {

// ======================================================
// 内部纯递归执行器 (单精度)
// 不包含流的分配逻辑，执行最纯粹的 Right-looking 递归
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

    // 1) X1 * L11^T = B1
    recursive_trsm_internal(handle, A, lda, B, ldb, n, m1, nb);

    // 2) B2 = B2 - X1 * L21^T 
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

    // 3) X2 * L22^T = B2
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
// 外部包装器: 单精度 Right-TRSM 
// 参考原文 5.1.4 节：利用流并发以及限制 GEMM 规模防止颠簸与误差爆炸
// ======================================================
void recursive_trsm(rocblas_handle handle,
                    float* A, int lda,
                    float* B, int ldb,
                    int n, int m, int nb) {
    const int N_LIMIT = 8192; 

    // 如果未达到阈值，直接调用常规递归即可
    if (n <= N_LIMIT) {
        recursive_trsm_internal(handle, A, lda, B, ldb, n, m, nb);
        return;
    }

    // 获取原始流
    hipStream_t orig_stream;
    rocblas_get_stream(handle, &orig_stream);

    // 计算切块数量
    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;
    
    // 创建流、句柄、事件资源池
    std::vector<hipStream_t> streams(num_chunks);
    std::vector<rocblas_handle> handles(num_chunks);
    std::vector<hipEvent_t> events(num_chunks);

    for (int i = 0; i < num_chunks; i++) {
        if (i == 0) {
            handles[i] = handle;
            streams[i] = orig_stream;
        } else {
            hipStreamCreate(&streams[i]);
            rocblas_create_handle(&handles[i]);
            rocblas_set_stream(handles[i], streams[i]);
            hipEventCreate(&events[i]);
        }
    }

    for (int i = 0; i < num_chunks; i++) {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);
        
        recursive_trsm_internal(handles[i], A, lda, B + row_start, ldb, current_n, m, nb);
        
        if (i > 0) {
            hipEventRecord(events[i], streams[i]);
        }
    }

    for (int i = 1; i < num_chunks; i++) {
        hipStreamWaitEvent(orig_stream, events[i], 0);
        hipStreamSynchronize(streams[i]);
        rocblas_destroy_handle(handles[i]);
        hipStreamDestroy(streams[i]);
        hipEventDestroy(events[i]);
    }
}

// ======================================================
// 外部包装器: 双精度 Right-TRSM 
// ======================================================
void recursive_trsm_double(rocblas_handle handle,
                           double* A, int lda,
                           double* B, int ldb,
                           int n, int m, int nb) {
    const int N_LIMIT = 8192;

    if (n <= N_LIMIT) {
        recursive_trsm_double_internal(handle, A, lda, B, ldb, n, m, nb);
        return;
    }

    hipStream_t orig_stream;
    rocblas_get_stream(handle, &orig_stream);

    int num_chunks = (n + N_LIMIT - 1) / N_LIMIT;
    std::vector<hipStream_t> streams(num_chunks);
    std::vector<rocblas_handle> handles(num_chunks);
    std::vector<hipEvent_t> events(num_chunks);

    for (int i = 0; i < num_chunks; i++) {
        if (i == 0) {
            handles[i] = handle;
            streams[i] = orig_stream;
        } else {
            hipStreamCreate(&streams[i]);
            rocblas_create_handle(&handles[i]);
            rocblas_set_stream(handles[i], streams[i]);
            hipEventCreate(&events[i]);
        }
    }

    for (int i = 0; i < num_chunks; i++) {
        int row_start = i * N_LIMIT;
        int current_n = std::min(N_LIMIT, n - row_start);
        recursive_trsm_double_internal(handles[i], A, lda, B + row_start, ldb, current_n, m, nb);
        
        if (i > 0) {
            hipEventRecord(events[i], streams[i]);
        }
    }

    for (int i = 1; i < num_chunks; i++) {
        hipStreamWaitEvent(orig_stream, events[i], 0);
        hipStreamSynchronize(streams[i]);
        rocblas_destroy_handle(handles[i]);
        hipStreamDestroy(streams[i]);
        hipEventDestroy(events[i]);
    }
}

} // extern "C"