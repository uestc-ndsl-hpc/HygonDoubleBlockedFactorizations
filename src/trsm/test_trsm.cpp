#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "trsm/recursive_trsm.h"
#include "trsm/iterative_trsm.h"

#include "utils/common/error_check.h"
#include "utils/memory/stream_pool.h"
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/benchmark_printer.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/memory/device_buffer.h"
#include "utils/benchmark/benchmark_runner.h"

#include "utils/matrix/matrix_generator.h"

using namespace std;

// Right-TRSM: X(n x m) * L^T(m x m) = B(n x m)
double calculate_gflops(int n, int m, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double total_flops = static_cast<double>(n) * m * m;
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

// ============================================================================
// FORWARD ERROR (Sampled) - Fully Optimized to Prevent CPU Hangs
// ============================================================================
template<typename T_REF>
double verify_relative_error(const vector<T_REF>& ref, const vector<float>& res, int sample_count = 10000) {
    double diff_norm = 0.0;
    double ref_norm  = 0.0;
    size_t total_size = ref.size();
    
    if (total_size <= 1000000) {
        for (size_t i = 0; i < total_size; i++) {
            double diff = static_cast<double>(ref[i]) - static_cast<double>(res[i]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
        }
    } else {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<size_t> dist(0, total_size - 1);

        for (int s = 0; s < sample_count; ++s) {
            size_t idx = dist(gen);
            double diff = static_cast<double>(ref[idx]) - static_cast<double>(res[idx]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[idx]) * static_cast<double>(ref[idx]);
        }
    }
    return sqrt(diff_norm) / (sqrt(ref_norm) + 1e-20);
}

// ============================================================================
// BACKWARD ERROR (Sampled)
// ============================================================================
double verify_backward_error_sampled(
    const vector<float>& A,          // 系数三角矩阵 L (M x M)
    const vector<float>& B_original, // 原始右端项 B (N x M)
    const vector<float>& X_result,   // 计算出的解矩阵 X (N x M)
    int n,                           // B, X 的行数
    int m,                           // B, X 的列数，同时也是 A 的阶数
    int lda,                         // A 的主维步长
    int ldb,                         // B, X 的主维步长
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * m;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, m - 1);

    size_t lda_s = static_cast<size_t>(lda);
    size_t ldb_s = static_cast<size_t>(ldb);

    for (int s = 0; s < sample_count; ++s) {
        int row = dist_row(gen); // 目标元素的行索引 (0 ~ n-1)
        int col = dist_col(gen); // 目标元素的列索引 (0 ~ m-1)
        double reconstructed = 0.0;

        for (int k = 0; k <= col; ++k) {
            size_t idx_X = static_cast<size_t>(k) * ldb_s + static_cast<size_t>(row);
            size_t idx_A = static_cast<size_t>(k) * lda_s + static_cast<size_t>(col);
            reconstructed += static_cast<double>(X_result[idx_X]) * static_cast<double>(A[idx_A]);
        }

        size_t idx_B = static_cast<size_t>(col) * ldb_s + static_cast<size_t>(row);
        
        double ref_val = static_cast<double>(B_original[idx_B]);
        double diff = reconstructed - ref_val;
        
        diff_norm += diff * diff;
        ref_norm  += ref_val * ref_val;
    }
    return std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-20);
}

void run_all_benchmarks(
    int n,
    int m,
    const vector<int>& nbs,
    ErrorMode error_mode,
    bool test_recursive,
    bool test_rocblas,
    bool test_iterative,
    bool test_half)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    // Matrix dimensions for Right-Side TRSM: L is m x m, B is n x m
    size_t size_A = static_cast<size_t>(m) * m * sizeof(float);
    size_t size_B = static_cast<size_t>(n) * m * sizeof(float);
    int lda = m; 
    int ldb = n; 

    StreamPool pool(4);

    // Host Memory Setup
    vector<float> h_A(m * m);
    vector<float> h_B(n * m);
    vector<float> h_X(n * m);

    generate_stable_lower_triangular<float>(h_A.data(), m, lda);
    generate_random_matrix<float>(h_B.data(), n, m, ldb);

    // Device Memory Setup
    DeviceBuffer<float> d_A(m * m);
    DeviceBuffer<float> d_B(n * m);

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    float alpha = 1.0f;

    // Golden Reference Generation
    vector<double> h_X_gold(n * m);
    vector<float>  h_X_ref_f32(n * m);
    bool has_fp64_ref = false;
    bool has_f32_ref  = false;

    if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
        cout << ">>> Generating Golden Reference...\n";
        bool do_fp64_verify = (std::max(n, m) <= 16384);

        // FP64 High-Precision Reference Generation
        if (do_fp64_verify) {
            DeviceBuffer<double> d_A_d(m * m);
            DeviceBuffer<double> d_B_d(n * m);
            vector<double> h_A_d(m * m);
            vector<double> h_B_d(n * m);

            for (size_t i = 0; i < h_A.size(); ++i) h_A_d[i] = static_cast<double>(h_A[i]);
            for (size_t i = 0; i < h_B.size(); ++i) h_B_d[i] = static_cast<double>(h_B[i]);

            HIP_CHECK(hipMemcpy(d_A_d, h_A_d.data(), static_cast<size_t>(m) * m * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_B_d, h_B_d.data(), static_cast<size_t>(n) * m * sizeof(double), hipMemcpyHostToDevice));

            double alpha_d = 1.0;
            ROCBLAS_CHECK(rocblas_dtrsm(handle, rocblas_side_right, rocblas_fill_lower,
                                        rocblas_operation_transpose, rocblas_diagonal_non_unit,
                                        n, m, &alpha_d, d_A_d, lda, d_B_d, ldb));

            HIP_CHECK(hipMemcpy(h_X_gold.data(), d_B_d, static_cast<size_t>(n) * m * sizeof(double), hipMemcpyDeviceToHost));
            has_fp64_ref = true;
        }
        // FP32 Standard Reference Generation
        else {
            HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
            ROCBLAS_CHECK(rocblas_strsm(handle, rocblas_side_right, rocblas_fill_lower,
                                        rocblas_operation_transpose, rocblas_diagonal_non_unit,
                                        n, m, &alpha, d_A, lda, d_B, ldb));

            HIP_CHECK(hipMemcpy(h_X_ref_f32.data(), d_B, size_B, hipMemcpyDeviceToHost));
            has_f32_ref = true;
        }
    }

    // Benchmark Execution Setup
    vector<BenchmarkResult> results;
    double baseline_time = -1.0;
    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_b = [&]() {
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
    };

    auto calc_gflops_trsm = [&](double ms) {
        return calculate_gflops(n, m, ms);
    };

    auto verify_trsm = [&](double& f_err, double& b_err) {
        if (error_mode != ErrorMode::None) {
            HIP_CHECK(hipMemcpy(h_X.data(), d_B, size_B, hipMemcpyDeviceToHost));

            if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
                if (has_fp64_ref) f_err = verify_relative_error(h_X_gold, h_X);
                else if (has_f32_ref) f_err = verify_relative_error(h_X_ref_f32, h_X);
            }
            if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both) {
                b_err = verify_backward_error_sampled(h_A, h_B, h_X, n, m, lda, ldb, 4096);
            }
        }
    };

    cout << ">>> Running Benchmarks..." << endl;

    // Execute rocBLAS (FP32) Baseline
    if (test_rocblas) {
        baseline_time = runner.execute(
            "rocBLAS (FP32)", "N/A", reset_b,
            [&]() {
                ROCBLAS_CHECK(rocblas_strsm(handle, rocblas_side_right, rocblas_fill_lower,
                                            rocblas_operation_transpose, rocblas_diagonal_non_unit,
                                            n, m, &alpha, d_A, lda, d_B, ldb));
            },
            verify_trsm, calc_gflops_trsm);
        results.back().speedup = 1.0;
    }

    // Execute Recursive (FP32) Solver Sweep
    if (test_recursive) {
        for (int nb : nbs) {
            runner.execute(
                "Rec (FP32)", to_string(nb), reset_b,
                [&]() { recursive_trsm(handle, &pool, d_A, lda, d_B, ldb, n, m, nb); },
                verify_trsm, calc_gflops_trsm);
        }
    }

    // Execute Recursive Mixed-Precision (FP16/FP32) Solver Sweep
    if (test_half) {
        for (int nb : nbs) {
            runner.execute(
                "Rec (FP16)", to_string(nb), reset_b,
                [&]() { recursive_trsm_mixed(handle, &pool, d_A, lda, d_B, ldb, n, m, nb); },
                verify_trsm, calc_gflops_trsm);
        }
    }

    // Execute Iterative Right-Looking (FP32) Solver Sweep
    if (test_iterative) {
        for (int nb : nbs) {
            runner.execute(
                "Iterative RL", to_string(nb), reset_b,
                [&]() { iterative_trsm_right_looking(handle, d_A, lda, d_B, ldb, n, m, nb); },
                verify_trsm, calc_gflops_trsm);
        }
    }

    print_benchmark_results("TRSM Benchmark Summary (N = " + std::to_string(n) + ", M = " + std::to_string(m) + ")",
                            results, error_mode, 120);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

int main(int argc, char** argv)
{
    int n = 4096;
    int m = 4096;
    int error_input     = 0; 
    bool test_recursive = true;
    bool test_rocblas   = true;
    bool test_iterative = true;
    bool test_half      = true;
    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) m = atoi(argv[2]);
    if (argc >= 4) error_input = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);
    if (argc >= 6) test_rocblas = atoi(argv[5]);
    if (argc >= 7) test_iterative = atoi(argv[6]);

    if (argc >= 8) {
        test_half = atoi(argv[7]);
        for (int i = 8; i < argc; ++i) nbs.push_back(atoi(argv[i]));
    } else {
        for (int i = 7; i < argc; ++i) nbs.push_back(atoi(argv[i]));
    }

    if (nbs.empty()) nbs.push_back(256);

    ErrorMode error_mode = ErrorMode::None;
    if (error_input == 1) error_mode = ErrorMode::Forward;
    else if (error_input == 2) error_mode = ErrorMode::Backward;
    else if (error_input == 3) error_mode = ErrorMode::Both;

    run_all_benchmarks(n, m, nbs, error_mode, test_recursive, test_rocblas, test_iterative, test_half);
    return 0;
}