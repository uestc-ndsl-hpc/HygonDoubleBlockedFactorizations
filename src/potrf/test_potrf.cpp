#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <string>
#include <random>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "potrf/recursive_potrf.h"

#include "utils/error_check.h"
#include "utils/benchmark_result.h"
#include "utils/benchmark_printer.h"
#include "utils/gpu_timer.h"
#include "utils/device_buffer.h"
#include "utils/benchmark_runner.h"
#include "utils/matrix_generator.h"

using namespace std;

// ============================================================================
// UTILITIES
// ============================================================================

double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double flops = (1.0 / 3.0) * (double)n * (double)n * (double)n;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// ============================================================================
// FORWARD ERROR
// ============================================================================

template <typename T>
double verify_potrf_against_double_ref(
    const T* L_res,
    const double* L_ref_double,
    int n,
    int lda)
{
    double diff_norm_sq = 0.0;
    double ref_norm_sq  = 0.0;

    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            size_t idx = (size_t)j * lda + i;

            double val_res = static_cast<double>(L_res[idx]);
            double val_ref = L_ref_double[idx];

            double diff = val_res - val_ref;

            diff_norm_sq += diff * diff;
            ref_norm_sq  += val_ref * val_ref;
        }
    }

    return sqrt(diff_norm_sq) / (sqrt(ref_norm_sq) + 1e-12);
}

// ============================================================================
// BACKWARD ERROR (Sampled)
// ============================================================================

template <typename T>
double verify_backward_error_sampled(
    const T* L_res,
    const T* A_orig,
    int n,
    int lda,
    int sample_count = 4096)
{
    long long total = static_cast<long long>(n) * (n + 1) / 2;
    sample_count = std::min(static_cast<long long>(sample_count), total);

    double diff_norm_sq = 0.0;
    double a_norm_sq    = 0.0;

    std::mt19937 gen(2026);
    std::uniform_int_distribution<int> dist_row(0, n - 1);
    std::uniform_int_distribution<int> dist_col(0, n - 1);

    for (int s = 0; s < sample_count; ) {
        int row = dist_row(gen);
        int col = dist_col(gen);

        if (row < col) {
            continue; 
        }

        double reconstructed = 0.0;

        for (int k = 0; k <= col; ++k) {
            double Lik = static_cast<double>(L_res[k * lda + row]);
            double Ljk = static_cast<double>(L_res[k * lda + col]);
            reconstructed += Lik * Ljk;
        }

        double a_val = static_cast<double>(A_orig[col * lda + row]);
        double diff = a_val - reconstructed;

        double factor = (row == col) ? 1.0 : 2.0;

        diff_norm_sq += factor * diff * diff;
        a_norm_sq    += factor * a_val * a_val;

        ++s;
    }

    return sqrt(diff_norm_sq) / (sqrt(a_norm_sq) + 1e-12);
}

// ============================================================================
// TEMPLATE DISPATCHERS
// ============================================================================

template <typename T>
void call_rocsolver_potrf(
    rocblas_handle handle,
    rocblas_fill uplo,
    int n,
    T* A,
    int lda,
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, uplo, n, A, lda, info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, uplo, n, A, lda, info));
    }
}

template <typename T>
void call_recursive_potrf(
    rocblas_handle handle, StreamPool* pool,
    T* A,
    int n,
    int lda,
    int nb,
    int* info,
    int precision_type)
{
    if constexpr (std::is_same_v<T, float>) {
        if (precision_type == 2) {
            recursive_potrf_mixed_driver(handle, pool, A, lda, n, nb, info);
        } else {
            // 调用标准单精度算法
            recursive_potrf(handle, pool, A, lda, n, nb, info);
        }
    } else {
        // 调用双精度算法
        recursive_potrf_double(handle, pool, A, lda, n, nb, info);
    }
}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

template <typename T>
void run_all_benchmarks(
    int n,
    const vector<int>& nbs,
    bool test_rocsolver,
    bool test_recursive,
    ErrorMode error_mode,
    int precision_type) 
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t main_stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &main_stream));

    StreamPool pool(16);

    size_t num_elements = (size_t)n * (size_t)n;
    size_t bytes = num_elements * sizeof(T);

    DeviceBuffer<int> d_info(1);
    DeviceBuffer<T> d_A_orig(num_elements);

    generate_spd_matrix<T>(handle, d_A_orig, n, n, 2026);
    vector<T> h_A(num_elements);
    HIP_CHECK(hipMemcpy(h_A.data(), d_A_orig, bytes, hipMemcpyDeviceToHost));

    DeviceBuffer<T> d_A(num_elements);
    vector<double> h_L_ref_double;

    if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
        cout << ">>> Generating FP64 Golden Reference...\n";

        h_L_ref_double.resize(num_elements);
        vector<double> h_A_double(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
            h_A_double[i] = static_cast<double>(h_A[i]);

        DeviceBuffer<double> d_A_double(num_elements);

        HIP_CHECK(hipMemcpy(
            d_A_double,
            h_A_double.data(),
            num_elements * sizeof(double),
            hipMemcpyHostToDevice));

        ROCBLAS_CHECK(rocsolver_dpotrf(
            handle,
            rocblas_fill_lower,
            n,
            d_A_double,
            n,
            d_info));

        HIP_CHECK(hipMemcpy(
            h_L_ref_double.data(),
            d_A_double,
            num_elements * sizeof(double),
            hipMemcpyDeviceToHost));
    }

    // ========================================================================
    // Benchmark Preparation
    // ========================================================================
    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    BenchmarkRunner runner(main_stream, baseline_time, results);

    auto reset_a = [&]() {
        HIP_CHECK(hipMemcpy(d_A, d_A_orig, bytes, hipMemcpyDeviceToDevice));
    };

    auto calc_gflops_potrf = [&](double ms) {
        return calculate_gflops(n, ms);
    };

    auto verify_potrf = [&](double& f_err, double& b_err) {
        if (error_mode != ErrorMode::None) {
            vector<T> h_res(num_elements);
            HIP_CHECK(hipMemcpy(h_res.data(), d_A, bytes, hipMemcpyDeviceToHost));

            // 计算前向误差
            if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both) {
                f_err = verify_potrf_against_double_ref(
                    h_res.data(),
                    h_L_ref_double.data(),
                    n,
                    n);
            }

            // 计算后向误差
            if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both) {
                b_err = verify_backward_error_sampled(
                    h_res.data(),
                    h_A.data(),
                    n,
                    n,
                    4096);
            }
        }
    };

    // ========================================================================
    // Run Benchmarks
    // ========================================================================
    cout << ">>> Running Benchmarks...\n";

    if (test_rocsolver) {
        string rocsolver_name = (precision_type == 2) ? "rocSOLVER (FP32)" : "rocSOLVER";
        baseline_time = runner.execute(
            rocsolver_name, 
            "N/A", 
            reset_a,
            [&]() {
                call_rocsolver_potrf<T>(handle, rocblas_fill_lower, n, d_A, n, d_info);
            },
            verify_potrf, 
            calc_gflops_potrf);

        results.back().speedup = 1.0;
    }

    if (test_recursive) {
        for (int nb : nbs) {
            string method_name = (precision_type == 2) ? "Mixed-Recursive" : "Recursive";
            runner.execute(
                method_name, 
                to_string(nb), 
                reset_a,
                [&]() {
                    call_recursive_potrf<T>(handle, &pool, d_A, n, n, nb, d_info, precision_type);
                },
                verify_potrf, 
                calc_gflops_potrf);
        }
    }

    // ============================================================
    // Print
    // ============================================================
    print_benchmark_results(
        "POTRF Benchmark Summary (N = " + std::to_string(n) + ")",
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

    bool test_rocsolver = true;
    bool test_recursive = true;
    int error_input     = 0; // 对应传参：0=None, 1=Forward, 2=Backward, 3=Both
    int precision_type  = 0; // 0=float, 1=double, 2=mixed

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) test_rocsolver = atoi(argv[2]);
    if (argc >= 4) test_recursive = atoi(argv[3]);
    if (argc >= 5) error_input = atoi(argv[4]);
    if (argc >= 6) precision_type = atoi(argv[5]);

    for (int i = 6; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }

    if (nbs.empty()) {
        nbs.push_back(256);
    }

    ErrorMode error_mode = ErrorMode::None;
    if (error_input == 1) error_mode = ErrorMode::Forward;
    else if (error_input == 2) error_mode = ErrorMode::Backward;
    else if (error_input == 3) error_mode = ErrorMode::Both;

    if (precision_type == 1) {
        cout << "Running benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(
            n,
            nbs,
            test_rocsolver,
            test_recursive,
            error_mode,
            precision_type);
    }
    else if (precision_type == 2) {
        cout << "Running benchmark in MIXED (FP16/FP32 Tensor Cores) precision\n";
        run_all_benchmarks<float>(
            n,
            nbs,
            test_rocsolver,
            test_recursive,
            error_mode,
            precision_type);
    }
    else {
        cout << "Running benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(
            n,
            nbs,
            test_rocsolver,
            test_recursive,
            error_mode,
            precision_type);
    }

    return 0;
}