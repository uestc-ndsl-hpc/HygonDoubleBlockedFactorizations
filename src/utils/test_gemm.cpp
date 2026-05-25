#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "utils/error_check.h"
#include "utils/mixed_gemm.h"

// ============================================================
// GFLOPS 计算
// ============================================================
double calculate_gflops(int m, int n, int k, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double flops = 2.0 * m * n * k;
    return flops / (time_ms * 1e6); // ms -> s -> GFLOPS
}

// ============================================================
// SGEMM (FP32 baseline)
// ============================================================
void launch_sgemm(
    rocblas_handle handle,
    hipStream_t stream,
    int m, int n, int k,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc)
{
    ROCBLAS_CHECK(rocblas_gemm_ex(
        handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        m, n, k,
        &alpha,
        A, rocblas_datatype_f32_r, lda,
        B, rocblas_datatype_f32_r, ldb,
        &beta,
        C, rocblas_datatype_f32_r, ldc,
        C, rocblas_datatype_f32_r, ldc,
        rocblas_datatype_f32_r,
        rocblas_gemm_algo_standard,
        0, 0));
}

// ============================================================
// Benchmark 主函数 (含精度与吞吐量双重分析)
// ============================================================
void run_benchmark_size(
    rocblas_handle handle,
    hipStream_t stream,
    int M, int N, int K)
{
    int lda = M;
    int ldb = N;
    int ldc = M;

    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)N * K;
    size_t size_C = (size_t)M * N;

    float alpha = -1.0f;
    float beta  = 1.0f;

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_init(size_C); // 专门留一份初始值用于恢复

    std::mt19937 gen(2026);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& v : h_A) v = dist(gen);
    for (auto& v : h_B) v = dist(gen);
    for (auto& v : h_C_init) v = dist(gen);

    // 分配 GPU 显存
    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(float)));

    // 用于保存各算子单次运行结果以进行误差比对的内存
    std::vector<float> h_C_sgemm(size_C);
    std::vector<float> h_C_ex(size_C);
    std::vector<float> h_C_mix(size_C);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const int warmup = 1;
    const int runs   = 1;

    double t_sgemm = 0.0;
    double t_ex    = 0.0;
    double t_mix   = 0.0;

    // 辅助闭包：恢复 C 矩阵初始状态
    auto reset_d_C = [&]() {
        HIP_CHECK(hipMemcpy(d_C, h_C_init.data(), size_C * sizeof(float), hipMemcpyHostToDevice));
    };

    // 辅助闭包：执行性能测量
    auto measure = [&](auto kernel, double &t)
    {
        reset_d_C();
        for (int i = 0; i < warmup; i++)
            kernel();

        HIP_CHECK(hipStreamSynchronize(stream));

        HIP_CHECK(hipEventRecord(start, stream));
        for (int i = 0; i < runs; i++)
            kernel();
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        t = ms / runs;
    };

    // 1. 测量 SGEMM 并捕获其最终结果作为基准
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));
    
    measure([&] {
        launch_sgemm(handle, stream, M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }, t_sgemm);
    HIP_CHECK(hipMemcpy(h_C_sgemm.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    // 2. 测量 GEMM_EX (16-32-32) 并捕获其结果
    measure([&] {
        launch_mixed_gemm_test(handle, stream, rocblas_operation_none, rocblas_operation_transpose,
                               M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }, t_ex);
    HIP_CHECK(hipMemcpy(h_C_ex.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    // 3. 测量 MIXED (16-32-16) 并捕获其结果
    measure([&] {
        launch_mixed_gemm(handle, stream, rocblas_operation_none, rocblas_operation_transpose,
                          M, N, K, alpha, d_A, lda, d_B, ldb, beta, d_C, ldc);
    }, t_mix);
    HIP_CHECK(hipMemcpy(h_C_mix.data(), d_C, size_C * sizeof(float), hipMemcpyDeviceToHost));

    // ========================================================
    // 误差计算模块 (以 SGEMM 为标准真值)
    // ========================================================
    auto compute_error = [](const std::vector<float>& baseline, const std::vector<float>& target, 
                            double& max_err, double& rel_fro_err) 
    {
        double max_absolute_error = 0.0;
        double frobenius_norm_diff = 0.0;
        double frobenius_norm_base = 0.0;

        for (size_t i = 0; i < baseline.size(); ++i) {
            double diff = std::abs(static_cast<double>(baseline[i]) - static_cast<double>(target[i]));
            if (diff > max_absolute_error) {
                max_absolute_error = diff;
            }
            frobenius_norm_diff += diff * diff;
            frobenius_norm_base += static_cast<double>(baseline[i]) * static_cast<double>(baseline[i]);
        }

        max_err = max_absolute_error;
        rel_fro_err = (frobenius_norm_base > 0.0) ? std::sqrt(frobenius_norm_diff / frobenius_norm_base) : 0.0;
    };

    double max_err_ex = 0.0, rel_err_ex = 0.0;
    double max_err_mix = 0.0, rel_err_mix = 0.0;

    compute_error(h_C_sgemm, h_C_ex, max_err_ex, rel_err_ex);
    compute_error(h_C_sgemm, h_C_mix, max_err_mix, rel_err_mix);

    // 计算吞吐量
    double g_sgemm = calculate_gflops(M, N, K, t_sgemm);
    double g_ex    = calculate_gflops(M, N, K, t_ex);
    double g_mix   = calculate_gflops(M, N, K, t_mix);

    // ========================================================
    // 打印双行输出：第一行吞吐量，第二行精度误差
    // ========================================================
    std::string size_str = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    
    // 1. 吞吐量行
    std::cout << std::setw(15) << size_str
              << " | " << std::setw(14) << std::fixed << std::setprecision(2) << g_sgemm
              << " | " << std::setw(16) << g_ex
              << " | " << std::setw(16) << g_mix << "  (GFLOPS)\n";

    // 2. 相对误差与最大绝对误差行
    std::cout << std::setw(15) << " " 
              << " | " << std::setw(14) << "0.00 (Base)"
              << " | " << "Rel:" << std::setw(9) << std::scientific << std::setprecision(2) << rel_err_ex 
              << " Max:" << std::setw(9) << max_err_ex
              << " | " << "Rel:" << std::setw(9) << rel_err_mix 
              << " Max:" << std::setw(9) << max_err_mix << "  (Error)\n";
              
    std::cout << "--------------------------------------------------------------------------------------\n";

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

// ============================================================
int main()
{
    rocblas_handle handle;
    hipStream_t stream;

    HIP_CHECK(hipStreamCreate(&stream));
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

    std::cout << "======================================================================================\n";
    std::cout << "             GEMM MIXED PRECISION PERFORMANCE & ACCURACY BENCHMARK                    \n";
    std::cout << "======================================================================================\n";

    std::cout << std::setw(15) << "M x N x K"
              << " | " << std::setw(14) << "SGEMM (FP32)"
              << " | " << std::setw(25) << "GEMM_EX (16-32-32)"
              << " | " << std::setw(25) << "MIXED (16-32-16)"
              << std::endl;

    std::cout << "--------------------------------------------------------------------------------------\n";

    std::vector<std::vector<int>> sizes = {
        {256,256,256},
        {512,512,512},
        {1024,1024,1024},
        {2048,2048,2048},
        {4096,4096,4096},
        {8192,8192,8192},
        {16384,16384,16384}
    };

    for (auto &s : sizes)
        run_benchmark_size(handle, stream, s[0], s[1], s[2]);

    std::cout << "======================================================================================\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}