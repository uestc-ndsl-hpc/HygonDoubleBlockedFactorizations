#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <iomanip>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "utils/error_check.h"
#include "utils/mixed_gemm.h"

// ============================================================
// GFLOPS
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
// benchmark
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
    std::vector<float> h_C(size_C);

    std::mt19937 gen(2026);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    for (auto& v : h_A) v = dist(gen);
    for (auto& v : h_B) v = dist(gen);
    for (auto& v : h_C) v = dist(gen);

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_B, size_B * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, size_C * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_C, h_C.data(), size_C * sizeof(float), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    const int warmup = 5;
    const int runs   = 50;

    double t_sgemm = 0.0;
    double t_ex    = 0.0;
    double t_mix   = 0.0;

    auto measure = [&](auto kernel, double &t)
    {
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

    // ========================================================
    // SGEMM
    // ========================================================
    measure([&] {
        launch_sgemm(handle, stream, M, N, K, alpha,
                     d_A, lda, d_B, ldb, beta, d_C, ldc);
    }, t_sgemm);

    // ========================================================
    // GEMM_EX (FP16 input, FP32 compute, FP32 output)
    // ========================================================
    measure([&] {
        launch_mixed_gemm_test(handle, stream,
                               rocblas_operation_none,
                               rocblas_operation_transpose,
                               M, N, K,
                               alpha,
                               d_A, lda,
                               d_B, ldb,
                               beta,
                               d_C, ldc);
    }, t_ex);

    // ========================================================
    // MIXED (FP16 input / FP32 compute / FP16 output path)
    // ========================================================
    measure([&] {
        launch_mixed_gemm(handle, stream,
                          rocblas_operation_none,
                          rocblas_operation_transpose,
                          M, N, K,
                          alpha,
                          d_A, lda,
                          d_B, ldb,
                          beta,
                          d_C, ldc);
    }, t_mix);

    // ========================================================
    // throughput
    // ========================================================
    double g_sgemm = calculate_gflops(M, N, K, t_sgemm);
    double g_ex    = calculate_gflops(M, N, K, t_ex);
    double g_mix   = calculate_gflops(M, N, K, t_mix);

    std::cout << std::setw(14)
              << (std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K))
              << " | " << std::setw(16) << g_sgemm
              << " | " << std::setw(18) << g_ex
              << " | " << std::setw(18) << g_mix
              << std::endl;

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

    std::cout << "===============================================================\n";
    std::cout << " GEMM THROUGHPUT BENCHMARK (GFLOPS)\n";
    std::cout << "===============================================================\n";

    std::cout << std::setw(14) << "M x N x K"
              << " | " << std::setw(16) << "SGEMM (FP32)"
              << " | " << std::setw(18) << "GEMM_EX (16-32-32)"
              << " | " << std::setw(18) << "MIXED (16-32-16)"
              << std::endl;

    std::cout << "---------------------------------------------------------------\n";

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

    std::cout << "===============================================================\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    HIP_CHECK(hipStreamDestroy(stream));
}