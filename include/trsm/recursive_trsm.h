#ifndef RECURSIVE_TRSM_H
#define RECURSIVE_TRSM_H

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifdef __cplusplus
extern "C" {
#endif

// 单精度递归TRSM
// 求解 L * X = B，L为n×n下三角矩阵（列主序）
// A: L矩阵，n×n，下三角部分有效
// B: 输入B矩阵，输出被X覆盖，n×m
// n: L的维度，m: B的列数，nb: 递归停止阈值
void recursive_trsm(rocblas_handle handle,
                    float* A, int lda,
                    float* B, int ldb,
                    int n, int m, int nb);

// 双精度递归TRSM
void recursive_trsm_double(rocblas_handle handle,
                           double* A, int lda,
                           double* B, int ldb,
                           int n, int m, int nb);

// 测试函数
void test_trsm_performance(int n, int m, int nb, int warmup, int iter);
void test_trsm_accuracy(int n, int m, int nb);
double compute_residual(float* A, float* X, float* B, int n, int m, int lda, int ldb);
double compute_residual_double(double* A, double* X, double* B, int n, int m, int lda, int ldb);

#ifdef __cplusplus
}
#endif

#endif