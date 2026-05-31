#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "syrk/recursive_syrk.h"
#include "syrk/batched_syrk.h"

#include "utils/error_check.h"
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_printer.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/device_buffer.h"
#include "utils/benchmark/benchmark_runner.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/stream_pool.h"

using namespace std;

// Calculate GFLOPS for SYRK
double calculate_gflops(int n, int k, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double flops = static_cast<double>(n) * (n + 1) * k;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// Relative Frobenius norm error calculation
template<typename T_REF>
double verify_relative_error_lower(
    const vector<T_REF>& ref,
    const vector<float>& res,
    int n)
{
    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            int idx = j * n + i;
            double diff = static_cast<double>(ref[idx]) - static_cast<double>(res[idx]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[idx]) * static_cast<double>(ref[idx]);
        }
    }
    return sqrt(diff_norm) / (sqrt(ref_norm) + 1e-20);
}

// Sampled backward error verification (Sampled Reconstructed Forward Error)
double verify_backward_error_sampled(
    const vector<float>& A,
    const vector<float>& C_original,
    const vector<float>& C_result,
    int n,
    int k,
    int lda,
    int ldc,
    float alpha = -1.0f,
    float beta = 1.0f,
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * (n + 1) / 2;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    std::mt19937 gen(2026);
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, n - 1);

    for (int s = 0; s < sample_count; ) {
        int row = dist_row(gen);
        int col = dist_col(gen);

        if (row < col) {
            continue; // Only sample lower triangular elements
        }

        double reconstructed = 0.0;

        for (int l = 0; l < k; ++l) {
            reconstructed +=
                static_cast<double>(A[l * lda + row]) *
                static_cast<double>(A[l * lda + col]);
        }

        reconstructed =
            static_cast<double>(alpha) * reconstructed +
            static_cast<double>(beta) * static_cast<double>(C_original[col * ldc + row]);

        double res = static_cast<double>(C_result[col * ldc + row]);
        double diff = res - reconstructed;

        diff_norm += diff * diff;
        ref_norm  += reconstructed * reconstructed;

        ++s;
    }

    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

void run_all_benchmarks(
    int n, int k, const vector<int>& nbs,
    ErrorMode error_mode,
    bool test_recursive,
    bool test_rocblas,
    bool test_batched,
    bool test_mixed)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    // 【新增】实例化流连接池 (StreamPool)
    // 这里的参数代表创建的硬件独立流数量，通常设为 8 或 16 可以带来良好的层内并发掩盖
    int num_streams = 16; 
    StreamPool pool(num_streams);

    int lda = n, ldc = n;
    size_t bytes_A = static_cast<size_t>(n) * k * sizeof(float);
    size_t bytes_C = static_cast<size_t>(n) * n * sizeof(float);

    // 1. 显存分配
    DeviceBuffer<float> d_A_f(n * k);
    DeviceBuffer<float> d_C_orig(n * n); // 原始备份
    DeviceBuffer<float> d_C_f(n * n);    // 工作区

    // 2. 使用规范化生成器初始化
    generate_syrk_matrices<float>(handle, (float*)d_A_f, (float*)d_C_orig, n, k, lda, ldc, 2026);

    // 3. 准备校验用 Host 数据
    vector<float> h_A_f(n * k), h_C_f(n * n);
    HIP_CHECK(hipMemcpy(h_A_f.data(), (float*)d_A_f, bytes_A, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_C_f.data(), (float*)d_C_orig, bytes_C, hipMemcpyDeviceToHost));

    float alpha_f = -1.0f, beta_f = 1.0f;

    // ========================================================================
    // Golden Reference Generation
    // ========================================================================
    vector<double> h_C_gold(n * n);
    vector<float>  h_C_ref_f32(n * n);
    
    bool has_fp64_ref = false;
    bool has_f32_ref  = false;

    if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
        cout << ">>> Generating Golden Reference for Error Verification...\n";
        bool do_fp64_verify = (std::max(n, k) <= 16384);
        
        if (do_fp64_verify) {
            vector<double> h_A_d(n * k);
            vector<double> h_C_d(n * n);

            for (size_t i = 0; i < h_A_f.size(); ++i) h_A_d[i] = h_A_f[i];
            for (size_t i = 0; i < h_C_f.size(); ++i) h_C_d[i] = h_C_f[i];

            DeviceBuffer<double> d_A_d(n * k);
            DeviceBuffer<double> d_C_d(n * n);
            
            HIP_CHECK(hipMemcpy((double*)d_A_d, h_A_d.data(), static_cast<size_t>(n) * k * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy((double*)d_C_d, h_C_d.data(), static_cast<size_t>(n) * n * sizeof(double), hipMemcpyHostToDevice));

            double alpha_d = -1.0;
            double beta_d  = 1.0;

            ROCBLAS_CHECK(
                rocblas_dsyrk(
                    handle, rocblas_fill_lower, rocblas_operation_none,
                    n, k, &alpha_d, (double*)d_A_d, lda, &beta_d, (double*)d_C_d, ldc));

            HIP_CHECK(hipMemcpy(h_C_gold.data(), (double*)d_C_d, static_cast<size_t>(n) * n * sizeof(double), hipMemcpyDeviceToHost));
            has_fp64_ref = true;
        } else {
            HIP_CHECK(hipMemcpy((float*)d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
            
            ROCBLAS_CHECK(
                rocblas_ssyrk(
                    handle, rocblas_fill_lower, rocblas_operation_none,
                    n, k, &alpha_f, (float*)d_A_f, lda, &beta_f, (float*)d_C_f, ldc));
                    
            HIP_CHECK(hipMemcpy(h_C_ref_f32.data(), (float*)d_C_f, bytes_C, hipMemcpyDeviceToHost));
            has_f32_ref = true;
        }
    }

    // ========================================================================
    // Benchmark Preparation
    // ========================================================================
    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_c = [&]() {
        HIP_CHECK(
            hipMemcpy(
                (float*)d_C_f, 
                h_C_f.data(), 
                bytes_C, 
                hipMemcpyHostToDevice));
    };

    auto calc_gflops_syrk = [&](double ms) {
        return calculate_gflops(n, k, ms);
    };

    auto verify_syrk = [&](double& f_err, double& b_err) {
        if (error_mode != ErrorMode::None) {
            vector<float> h_C_out(n * n);
            HIP_CHECK(hipMemcpy(h_C_out.data(), (float*)d_C_f, bytes_C, hipMemcpyDeviceToHost));

            if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
                if (has_fp64_ref) {
                    f_err = verify_relative_error_lower(h_C_gold, h_C_out, n);
                } else if (has_f32_ref) {
                    f_err = verify_relative_error_lower(h_C_ref_f32, h_C_out, n);
                }
            }

            if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both) {
                b_err = verify_backward_error_sampled(
                    h_A_f, h_C_f, h_C_out, n, k, lda, ldc, alpha_f, beta_f, 4096);
            }
        }
    };

    // ========================================================================
    // Run Benchmarks
    // ========================================================================
    cout << ">>> Running Benchmarks..." << endl;

    if (test_rocblas) {
        baseline_time = runner.execute(
            "rocBLAS (FP32)", 
            "N/A", 
            reset_c,
            [&]() {
                ROCBLAS_CHECK(
                    rocblas_ssyrk(
                        handle, rocblas_fill_lower, rocblas_operation_none,
                        n, k, &alpha_f, (float*)d_A_f, lda, &beta_f, (float*)d_C_f, ldc));
            },
            verify_syrk, 
            calc_gflops_syrk);
        
        results.back().speedup = 1.0; 
    }

    if (test_recursive) {
        for (int nb : nbs) {
            runner.execute(
                "Recursive (FP32)", 
                to_string(nb), 
                reset_c,
                [&]() {
                    // 如果该递归接口后续也需要流池，可参考 Batched 做同样更改
                    recursive_syrk_parallel(handle, (float*)d_A_f, lda, (float*)d_C_f, ldc, n, k, nb);
                },
                verify_syrk, 
                calc_gflops_syrk);
        }
    }

    if (test_batched) {
        for (int nb : nbs) {
            runner.execute(
                "Batched (FP32)", 
                to_string(nb), 
                reset_c,
                [&]() {
                    // 【已修改】传入外部实例化的流连接池指针 &pool
                    batched_syrk_bottom_up(handle, &pool, (float*)d_A_f, lda, (float*)d_C_f, ldc, n, k, nb);
                },
                verify_syrk, 
                calc_gflops_syrk);
        }
    }

    if (test_mixed) {
        for (int nb : nbs) {
            runner.execute(
                "Batched (Mixed)", 
                to_string(nb), 
                reset_c,
                [&]() {
                    // 【已修改】传入外部实例化的流连接池指针 &pool
                    batched_syrk_bottom_up_mixed(handle, &pool, (float*)d_A_f, lda, (float*)d_C_f, ldc, n, k, nb);
                },
                verify_syrk, 
                calc_gflops_syrk);
        }
    }

    print_benchmark_results(
        "SYRK Benchmark Summary (N = " + std::to_string(n) + ", K = " + std::to_string(k) + ")",
        results,
        error_mode,
        120
    );

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    int n = 4096;
    int k = 4096;

    int error_input     = 0;
    bool test_recursive = true;
    bool test_rocblas   = true;
    bool test_batched   = true;
    bool test_mixed     = true;

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) error_input = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);
    if (argc >= 6) test_rocblas = atoi(argv[5]);
    if (argc >= 7) test_batched = atoi(argv[6]);
    if (argc >= 8) test_mixed = atoi(argv[7]);

    for (int i = 8; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }
    if (nbs.empty()) {
        nbs.push_back(256);
    }

    ErrorMode error_mode = ErrorMode::None;
    if (error_input == 1)      error_mode = ErrorMode::Forward;
    else if (error_input == 2) error_mode = ErrorMode::Backward;
    else if (error_input == 3) error_mode = ErrorMode::Both;

    run_all_benchmarks(
        n, k, nbs, 
        error_mode, 
        test_recursive, 
        test_rocblas, 
        test_batched,
        test_mixed);

    return 0;
}