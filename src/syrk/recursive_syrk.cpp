#include "syrk/recursive_syrk.h"
#include "utils/common/error_check.h"
#include <hip/hip_runtime.h>
#include <iostream>

// ======================================================
// 内部纯递归函数 
// ======================================================
template <typename T>
void syrk_internal(rocblas_handle handle, T* A, int lda, T* C, int ldc, int n, int k, int nb) {
    if (n <= 0 || k <= 0) return;

    if (n <= nb) {
        T alpha = -1.0;
        T beta  = 1.0;
        if constexpr (std::is_same_v<T, float>) {
            rocblas_ssyrk(handle, rocblas_fill_lower, rocblas_operation_none, n, k, &alpha, A, lda, &beta, C, ldc);
        } else {
            rocblas_dsyrk(handle, rocblas_fill_lower, rocblas_operation_none, n, k, &alpha, A, lda, &beta, C, ldc);
        }
        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    // 1. 递归处理 C11
    syrk_internal(handle, A, lda, C, ldc, n1, k, nb);
    
    // 2. 递归处理 C22
    syrk_internal(handle, A + n1, lda, C + n1 * ldc + n1, ldc, n2, k, nb);

    // 3. GEMM 更新 C21: C21 = C21 - A2 * A1^T
    T alpha_g = -1.0;
    T beta_g  = 1.0;
    if constexpr (std::is_same_v<T, float>) {
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n2, n1, k, &alpha_g, A + n1, lda, A, lda, &beta_g, C + n1, ldc);
    } else {
        rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n2, n1, k, &alpha_g, A + n1, lda, A, lda, &beta_g, C + n1, ldc);
    }
}

// ======================================================
// 外部并行接口 (仅在顶层创建流)
// ======================================================
void recursive_syrk_parallel(rocblas_handle handle, float* A, int lda, float* C, int ldc, int n, int k, int nb) {
    if (n <= nb * 2) {
        syrk_internal<float>(handle, A, lda, C, ldc, n, k, nb);
        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    // 只创建两个辅助流用于并行化 C11 和 C22 的更新
    hipStream_t s1, s2;
    rocblas_handle h1, h2;
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    ROCBLAS_CHECK(rocblas_create_handle(&h1));
    ROCBLAS_CHECK(rocblas_create_handle(&h2));
    ROCBLAS_CHECK(rocblas_set_stream(h1, s1));
    ROCBLAS_CHECK(rocblas_set_stream(h2, s2));

    // 并行发射子递归任务
    syrk_internal<float>(h1, A, lda, C, ldc, n1, k, nb);
    syrk_internal<float>(h2, A + n1, lda, C + n1 * ldc + n1, ldc, n2, k, nb);

    // GEMM 在主流上执行
    float alpha = -1.0f, beta = 1.0f;
    rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                  n2, n1, k, &alpha, A + n1, lda, A, lda, &beta, C + n1, ldc);

    // 等待顶层流同步
    HIP_CHECK(hipStreamSynchronize(s1));
    HIP_CHECK(hipStreamSynchronize(s2));

    // 清理顶层资源
    ROCBLAS_CHECK(rocblas_destroy_handle(h1));
    ROCBLAS_CHECK(rocblas_destroy_handle(h2));
    HIP_CHECK(hipStreamDestroy(s1));
    HIP_CHECK(hipStreamDestroy(s2));
}

// 双精度并行接口
void recursive_syrk_parallel_double(rocblas_handle handle, double* A, int lda, double* C, int ldc, int n, int k, int nb) {
    if (n <= nb * 2) {
        syrk_internal<double>(handle, A, lda, C, ldc, n, k, nb);
        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    hipStream_t s1, s2;
    rocblas_handle h1, h2;
    HIP_CHECK(hipStreamCreate(&s1));
    HIP_CHECK(hipStreamCreate(&s2));
    ROCBLAS_CHECK(rocblas_create_handle(&h1));
    ROCBLAS_CHECK(rocblas_create_handle(&h2));
    ROCBLAS_CHECK(rocblas_set_stream(h1, s1));
    ROCBLAS_CHECK(rocblas_set_stream(h2, s2));

    syrk_internal<double>(h1, A, lda, C, ldc, n1, k, nb);
    syrk_internal<double>(h2, A + n1, lda, C + n1 * ldc + n1, ldc, n2, k, nb);

    double alpha = -1.0, beta = 1.0;
    rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                  n2, n1, k, &alpha, A + n1, lda, A, lda, &beta, C + n1, ldc);

    HIP_CHECK(hipStreamSynchronize(s1));
    HIP_CHECK(hipStreamSynchronize(s2));

    ROCBLAS_CHECK(rocblas_destroy_handle(h1));
    ROCBLAS_CHECK(rocblas_destroy_handle(h2));
    HIP_CHECK(hipStreamDestroy(s1));
    HIP_CHECK(hipStreamDestroy(s2));
}