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

using namespace std;

// Calculate GFLOPS for SYRK
double calculate_gflops(int n, int k, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double flops = static_cast<double>(n) * (n + 1) * k;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// Host-side matrix initialization
void generate_random_matrix(float* A, int rows, int cols, int lda) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            A[j * lda + i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

void generate_symmetric_matrix(float* C, int n, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            C[j * ldc + i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
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
            // A is col-major. Compute dot product of A's rows.
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

// Struct to store benchmark results for the final table
struct BenchmarkResult {
    string method;
    string nb_str;
    double time_ms;
    double gflops;
    double speedup;
    double forward_error;
    double backward_error;
};

// ============================================================================
// Main Unified Benchmark Runner
// ============================================================================
void run_all_benchmarks(
    int n, int k, const vector<int>& nbs,
    int error_mode,
    bool test_recursive,
    bool test_rocblas,
    bool test_batched)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    int lda = n;
    int ldc = n;

    // ========================================================================
    // 1. Host Memory Allocation & Initialization
    // ========================================================================
    vector<float> h_A_f(n * k);
    vector<float> h_C_f(n * n);

    generate_random_matrix(h_A_f.data(), n, k, lda);
    generate_symmetric_matrix(h_C_f.data(), n, ldc);

    // ========================================================================
    // 2. Device Memory Allocation
    // ========================================================================
    float *d_A_f = nullptr;
    float *d_C_f = nullptr;
    
    size_t bytes_A = static_cast<size_t>(n) * k * sizeof(float);
    size_t bytes_C = static_cast<size_t>(n) * n * sizeof(float);

    HIP_CHECK(hipMalloc(&d_A_f, bytes_A));
    HIP_CHECK(hipMalloc(&d_C_f, bytes_C));
    HIP_CHECK(hipMemcpy(d_A_f, h_A_f.data(), bytes_A, hipMemcpyHostToDevice));

    // Scalar constants for SYRK (C = -A*A^T + C)
    float alpha_f = -1.0f;
    float beta_f  = 1.0f;

    // ========================================================================
    // 3. Golden Reference Generation
    // ========================================================================
    vector<double> h_C_gold(n * n);
    vector<float>  h_C_ref_f32(n * n);
    
    bool has_fp64_ref = false;
    bool has_f32_ref  = false;

    if (error_mode == 1 || error_mode == 3) {
        
        cout << ">>> Generating Golden Reference for Error Verification...\n";
        bool do_fp64_verify = (std::max(n, k) <= 16384);
        
        if (do_fp64_verify) {
            double *d_A_d = nullptr;
            double *d_C_d = nullptr;
            
            vector<double> h_A_d(n * k);
            vector<double> h_C_d(n * n);

            for (size_t i = 0; i < h_A_f.size(); ++i) h_A_d[i] = h_A_f[i];
            for (size_t i = 0; i < h_C_f.size(); ++i) h_C_d[i] = h_C_f[i];

            HIP_CHECK(hipMalloc(&d_A_d, static_cast<size_t>(n) * k * sizeof(double)));
            HIP_CHECK(hipMalloc(&d_C_d, static_cast<size_t>(n) * n * sizeof(double)));
            
            HIP_CHECK(hipMemcpy(d_A_d, h_A_d.data(), static_cast<size_t>(n) * k * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_C_d, h_C_d.data(), static_cast<size_t>(n) * n * sizeof(double), hipMemcpyHostToDevice));

            double alpha_d = -1.0;
            double beta_d  = 1.0;

            ROCBLAS_CHECK(
                rocblas_dsyrk(
                    handle, rocblas_fill_lower, rocblas_operation_none,
                    n, k, &alpha_d, d_A_d, lda, &beta_d, d_C_d, ldc));

            HIP_CHECK(hipMemcpy(h_C_gold.data(), d_C_d, static_cast<size_t>(n) * n * sizeof(double), hipMemcpyDeviceToHost));
            
            HIP_CHECK(hipFree(d_A_d));
            HIP_CHECK(hipFree(d_C_d));
            has_fp64_ref = true;
        } else {
            // Use standard rocBLAS FP32 as reference if N is too large for FP64
            HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
            
            ROCBLAS_CHECK(
                rocblas_ssyrk(
                    handle, rocblas_fill_lower, rocblas_operation_none,
                    n, k, &alpha_f, d_A_f, lda, &beta_f, d_C_f, ldc));
                    
            HIP_CHECK(hipMemcpy(h_C_ref_f32.data(), d_C_f, bytes_C, hipMemcpyDeviceToHost));
            has_f32_ref = true;
        }
    }

    // ========================================================================
    // 4. Benchmark Preparation
    // ========================================================================
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    // Helper lambda to run a single test and record its result
    auto execute_benchmark = [&](auto func, const string& name, const string& nb_str) {
        vector<float> h_C_out(n * n);

        // Warmup
        HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize()); // Sync device to avoid caching noise before running func
        func();

        // Actual Run
        HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        
        HIP_CHECK(hipEventRecord(start));
        func();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        double gflops = calculate_gflops(n, k, ms);
        
        double forward_error = -1.0;
        double backward_error = -1.0;

        // --------------------------------------------------------------------
        // Error Verification
        // --------------------------------------------------------------------
        if (error_mode != 0) {
            HIP_CHECK(hipMemcpy(h_C_out.data(), d_C_f, bytes_C, hipMemcpyDeviceToHost));

            // Forward Error (Full matrix exact comparison)
            if (error_mode == 1 || error_mode == 3) {
                if (has_fp64_ref) {
                    forward_error = verify_relative_error_lower(h_C_gold, h_C_out, n);
                } else if (has_f32_ref) {
                    forward_error = verify_relative_error_lower(h_C_ref_f32, h_C_out, n);
                }
            }

            // Backward Error (Sampled reconstructed verification)
            if (error_mode == 2 || error_mode == 3) {
                backward_error = verify_backward_error_sampled(
                    h_A_f, h_C_f, h_C_out, n, k, lda, ldc, alpha_f, beta_f, 4096);
            }
        }

        double speedup = (baseline_time > 0.0) ? (baseline_time / ms) : 0.0;
        
        results.push_back({
            name, 
            nb_str, 
            static_cast<double>(ms), 
            gflops, 
            speedup, 
            forward_error, 
            backward_error
        });
        
        return static_cast<double>(ms);
    };

    // ========================================================================
    // 5. Run Tests
    // ========================================================================
    cout << ">>> Running Benchmarks..." << endl;

    if (test_rocblas) {
        baseline_time = execute_benchmark([&]() {
            ROCBLAS_CHECK(
                rocblas_ssyrk(
                    handle, rocblas_fill_lower, rocblas_operation_none,
                    n, k, &alpha_f, d_A_f, lda, &beta_f, d_C_f, ldc));
        }, "rocBLAS", "N/A");
        
        // Fix speedup for baseline itself
        results.back().speedup = 1.0; 
    }

    if (test_recursive) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                recursive_syrk_parallel(handle, d_A_f, lda, d_C_f, ldc, n, k, nb);
            }, "Recursive", to_string(nb));
        }
    }

    if (test_batched) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                batched_syrk_bottom_up(handle, d_A_f, lda, d_C_f, ldc, n, k, nb);
            }, "Batched", to_string(nb));
        }
    }

    // ========================================================================
    // 6. Print Summary Table
    // ========================================================================
    cout << "\n";
    cout << string(85, '=') << "\n";
    cout << "SYRK Benchmark Summary (N = " << n << ", K = " << k << ")\n";
    cout << string(85, '=') << "\n";
    
    // Table Header
    cout << left 
         << setw(15) << "Method" 
         << " | " << setw(5) << "NB"
         << " | " << setw(11) << "Time (ms)"
         << " | " << setw(11) << "GFLOPS"
         << " | " << setw(9) << "Speedup"
         << " | " << "Forward Error | Backward Error\n";
    cout << string(85, '-') << "\n";

    // Table Rows
    for (const auto& res : results) {
        cout << left 
             << setw(15) << res.method
             << " | " << setw(5) << res.nb_str
             << " | " << right << setw(9) << fixed << setprecision(3) << res.time_ms << "  "
             << " | " << right << setw(9) << fixed << setprecision(2) << res.gflops << "  "
             << " | ";
             
        if (res.speedup > 0) {
            cout << right << setw(7) << fixed << setprecision(2) << res.speedup << "x ";
        } else {
            cout << right << setw(8) << "N/A ";
        }
        
        cout << " | ";
        if (error_mode == 1 || error_mode == 3) {
            cout << scientific << setprecision(2) << res.forward_error;
        } else {
            cout << "N/A          "; // pad for formatting alignment
        }
        cout << " | ";
        if (error_mode == 2 || error_mode == 3) {
            cout << scientific << setprecision(2) << res.backward_error;
        } else {
            cout << "N/A";
        }
        cout << "\n";
    }
    cout << string(85, '=') << "\n";

    // ========================================================================
    // 7. Cleanup
    // ========================================================================
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A_f));
    HIP_CHECK(hipFree(d_C_f));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    int n = 4096;
    int k = 4096;

    int error_mode      = 0;
    bool test_recursive = true;
    bool test_rocblas   = true;
    bool test_batched   = true;

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) error_mode = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);
    if (argc >= 6) test_rocblas = atoi(argv[5]);
    if (argc >= 7) test_batched = atoi(argv[6]);

    for (int i = 7; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }
    if (nbs.empty()) {
        nbs.push_back(256);
    }

    srand(2026);

    run_all_benchmarks(
        n, k, nbs, 
        error_mode, 
        test_recursive, 
        test_rocblas, 
        test_batched);

    return 0;
}