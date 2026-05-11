#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "potrf/recursive_potrf.h"

using namespace std;

// ============================================================================
// HIP & ROCBLAS CHECK MACROS
// ============================================================================

#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t error = (cmd);                                          \
        if (error != hipSuccess) {                                         \
            cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                 << " -> " << hipGetErrorString(error) << endl;            \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

#define ROCBLAS_CHECK(cmd)                                                 \
    do {                                                                   \
        rocblas_status status = (cmd);                                     \
        if (status != rocblas_status_success) {                            \
            cerr << "rocBLAS/rocSOLVER error at "                          \
                 << __FILE__ << ":" << __LINE__                            \
                 << " -> status = " << status << endl;                     \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// ============================================================================
// UTILITIES
// ============================================================================

// Calculate GFLOPS for POTRF
double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double flops = (1.0 / 3.0) * (double)n * (double)n * (double)n;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// Generate SPD Matrix
template <typename T>
void generate_spd_matrix(T* A, int n, int lda)
{
    for (size_t j = 0; j < (size_t)n; ++j) {
        for (size_t i = 0; i < (size_t)n; ++i) {
            A[j * (size_t)lda + i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }
    // Make it diagonally dominant to ensure positive definiteness
    for (size_t i = 0; i < (size_t)n; ++i) {
        A[i * (size_t)lda + i] += static_cast<T>(n);
    }
}

// FP64 Reference Error
template <typename T>
double verify_potrf_against_double_ref(
    const T* L_res,
    const double* L_ref_double,
    int n, int lda)
{
    double diff_norm_sq = 0.0;
    double ref_norm_sq  = 0.0;

    for (size_t j = 0; j < (size_t)n; ++j) {
        for (size_t i = j; i < (size_t)n; ++i) {
            size_t idx = j * (size_t)lda + i;
            
            double val_res = static_cast<double>(L_res[idx]);
            double val_ref = L_ref_double[idx];
            
            double diff = val_res - val_ref;
            diff_norm_sq += diff * diff;
            ref_norm_sq  += val_ref * val_ref;
        }
    }
    return std::sqrt(diff_norm_sq) / (std::sqrt(ref_norm_sq) + 1e-12);
}

// Struct to store benchmark results for the final table
struct BenchmarkResult {
    string method;
    string nb_str;
    double time_ms;
    double gflops;
    double speedup;
    double error;
};

// ============================================================================
// TEMPLATE DISPATCHERS
// ============================================================================

template <typename T>
void call_rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo, int n, T* A, int lda, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, uplo, n, A, lda, info));
    } else {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, uplo, n, A, lda, info));
    }
}

template <typename T>
void call_recursive_potrf(rocblas_handle handle, T* A, int n, int lda, int nb, int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        recursive_potrf(handle, A, lda, n, nb, info);
    } else {
        recursive_potrf_double(handle, A, lda, n, nb, info);
    }
}

// ============================================================================
// MAIN UNIFIED BENCHMARK RUNNER
// ============================================================================

template <typename T>
void run_all_benchmarks(
    int n, 
    const vector<int>& nbs,
    bool enable_error,
    bool test_rocsolver,
    bool test_recursive)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    size_t num_elements = (size_t)n * (size_t)n;
    size_t bytes = num_elements * sizeof(T);

    int* d_info = nullptr;
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    // ========================================================================
    // 1. Host Memory Allocation & Initialization
    // ========================================================================
    vector<T> h_A(num_elements);
    generate_spd_matrix(h_A.data(), n, n);

    // ========================================================================
    // 2. Device Memory Allocation
    // ========================================================================
    T* d_A = nullptr;
    HIP_CHECK(hipMalloc(&d_A, bytes));

    // ========================================================================
    // 3. Golden Reference Generation (for Error Verification)
    // ========================================================================
    vector<double> h_L_ref_double;

    if (enable_error) {
        cout << ">>> Generating FP64 Golden Reference for Error Verification...\n";
        h_L_ref_double.resize(num_elements);
        vector<double> h_A_double(num_elements);
        
        for (size_t i = 0; i < num_elements; ++i) {
            h_A_double[i] = static_cast<double>(h_A[i]);
        }

        double* d_A_double = nullptr;
        HIP_CHECK(hipMalloc(&d_A_double, num_elements * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_A_double, h_A_double.data(), num_elements * sizeof(double), hipMemcpyHostToDevice));

        ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, d_A_double, n, d_info));
        HIP_CHECK(hipMemcpy(h_L_ref_double.data(), d_A_double, num_elements * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_A_double));
    }

    // ========================================================================
    // 4. Benchmark Preparation
    // ========================================================================
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    auto execute_benchmark = [&](auto func, const string& name, const string& nb_str) {
        
        // Reset Input for Warmup
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        func(); // Warmup

        // Reset Input for Actual Run
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Timing
        HIP_CHECK(hipEventRecord(start, 0));
        func();
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        double gflops = calculate_gflops(n, ms);
        double error = -1.0;

        if (enable_error) {
            vector<T> h_res(num_elements);
            HIP_CHECK(hipMemcpy(h_res.data(), d_A, bytes, hipMemcpyDeviceToHost));
            error = verify_potrf_against_double_ref(h_res.data(), h_L_ref_double.data(), n, n);
        }

        double speedup = (baseline_time > 0.0) ? (baseline_time / ms) : 0.0;
        results.push_back({name, nb_str, static_cast<double>(ms), gflops, speedup, error});

        return static_cast<double>(ms);
    };

    // ========================================================================
    // 5. Run Tests
    // ========================================================================
    cout << ">>> Running Benchmarks..." << endl;

    if (test_rocsolver) {
        baseline_time = execute_benchmark([&]() {
            call_rocsolver_potrf<T>(handle, rocblas_fill_lower, n, d_A, n, d_info);
        }, "rocSOLVER", "N/A");

        // Fix speedup for baseline itself
        results.back().speedup = 1.0;
    }

    if (test_recursive) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                call_recursive_potrf<T>(handle, d_A, n, n, nb, d_info);
            }, "Recursive", to_string(nb));
        }
    }

    // ========================================================================
    // 6. Print Summary Table
    // ========================================================================
    cout << "\n";
    cout << string(85, '=') << "\n";
    cout << "POTRF Benchmark Summary (N = " << n 
         << ", Precision = " << (std::is_same_v<T, float> ? "FP32" : "FP64") << ")\n";
    cout << string(85, '=') << "\n";

    // Table Header
    cout << left 
         << setw(15) << "Method" 
         << " | " << setw(5) << "NB"
         << " | " << setw(11) << "Time (ms)"
         << " | " << setw(11) << "GFLOPS"
         << " | " << setw(9) << "Speedup"
         << " | " << "Error (vs FP64)\n";
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
        if (enable_error) {
            cout << scientific << setprecision(2) << res.error;
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
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_info));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    int n = 4096;
    
    bool enable_error   = false;
    bool test_rocsolver = true;
    bool test_recursive = true;

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) enable_error = atoi(argv[2]);
    if (argc >= 4) test_rocsolver = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);

    for (int i = 5; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }
    if (nbs.empty()) {
        nbs.push_back(256);
    }

    srand(2026);

    run_all_benchmarks<float>(
        n, 
        nbs, 
        enable_error,
        test_rocsolver,
        test_recursive);

    return 0;
}