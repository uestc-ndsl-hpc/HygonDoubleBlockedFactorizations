#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "trsm/recursive_trsm.h"
#include "trsm/iterative_trsm.h"

using namespace std;

// ============================================================================
// HIP & ROCBLAS CHECK MACROS
// ============================================================================

#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t error = cmd;                                            \
        if (error != hipSuccess) {                                         \
            cerr << "HIP error at " << __FILE__ << ":" << __LINE__         \
                 << ": " << hipGetErrorString(error) << endl;              \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while(0)

#define ROCBLAS_CHECK(cmd)                                                 \
    do {                                                                   \
        rocblas_status status = cmd;                                       \
        if (status != rocblas_status_success) {                            \
            cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__     \
                 << ": " << rocblas_status_to_string(status) << endl;      \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while(0)

// ============================================================================
// UTILITIES
// ============================================================================

// Calculate GFLOPS for TRSM
double calculate_gflops(int n, int m, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double total_flops = static_cast<double>(n) * n * m;
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

// Generate Stable Lower Triangular Matrix
void generate_stable_lower_triangular(float* A, int n, int lda)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i < j) {
                A[j * lda + i] = 0.0f;
            } else if (i == j) {
                A[j * lda + i] = n + static_cast<float>(rand()) / RAND_MAX;
            } else {
                A[j * lda + i] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
}

// Generate Random Matrix
void generate_random_matrix(float* B, int n, int m, int ldb)
{
    for (int i = 0; i < n * m; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
}

// Verify Result (AX = B)
double verify_result(
    float* h_A,
    float* h_X,
    float* h_B,
    int n, int m,
    int lda, int ldb)
{
    double max_err = 0.0;
    double max_b   = 0.0;

    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) {
                sum += static_cast<double>(h_A[k * lda + i]) *
                       static_cast<double>(h_X[j * ldb + k]);
            }
            double diff = std::abs(sum - h_B[j * ldb + i]);
            max_err = std::max(max_err, diff);
            max_b   = std::max(max_b, std::abs(static_cast<double>(h_B[j * ldb + i])));
        }
    }
    return max_err / (max_b + 1e-10);
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
// MAIN UNIFIED BENCHMARK RUNNER
// ============================================================================
void run_all_benchmarks(
    int n, int m, const vector<int>& nbs,
    bool enable_error,
    bool test_recursive,
    bool test_rocblas,
    bool test_iterative)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    size_t size_A = static_cast<size_t>(n) * n * sizeof(float);
    size_t size_B = static_cast<size_t>(n) * m * sizeof(float);

    int lda = n;
    int ldb = n;

    // ========================================================================
    // 1. Host Memory Allocation & Initialization
    // ========================================================================
    vector<float> h_A(n * n);
    vector<float> h_B(n * m);   // Original B matrix
    vector<float> h_X(n * m);   // To store X after TRSM (in-place computation)

    generate_stable_lower_triangular(h_A.data(), n, lda);
    generate_random_matrix(h_B.data(), n, m, ldb);

    // ========================================================================
    // 2. Device Memory Allocation
    // ========================================================================
    float* d_A = nullptr;
    float* d_B = nullptr;

    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));

    float alpha = 1.0f;

    // ========================================================================
    // 3. Benchmark Preparation
    // ========================================================================
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    // Helper lambda to run a single test
    auto execute_benchmark = [&](auto func, const string& name, const string& nb_str) {
        
        // Reset B before warmup
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        
        // Warmup
        func();

        // Reset B before actual run
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Timing
        HIP_CHECK(hipEventRecord(start_event, 0));
        func();
        HIP_CHECK(hipEventRecord(stop_event, 0));
        HIP_CHECK(hipEventSynchronize(stop_event));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));

        double gflops = calculate_gflops(n, m, ms);
        double error = -1.0;

        if (enable_error) {
            HIP_CHECK(hipMemcpy(h_X.data(), d_B, size_B, hipMemcpyDeviceToHost));
            error = verify_result(h_A.data(), h_X.data(), h_B.data(), n, m, lda, ldb);
        }

        double speedup = (baseline_time > 0.0) ? (baseline_time / ms) : 0.0;
        results.push_back({name, nb_str, static_cast<double>(ms), gflops, speedup, error});

        return static_cast<double>(ms);
    };

    // ========================================================================
    // 4. Run Tests
    // ========================================================================
    cout << ">>> Running Benchmarks..." << endl;

    if (test_rocblas) {
        baseline_time = execute_benchmark([&]() {
            ROCBLAS_CHECK(rocblas_strsm(
                handle, rocblas_side_left, rocblas_fill_lower, rocblas_operation_none,
                rocblas_diagonal_non_unit, n, m, &alpha, d_A, lda, d_B, ldb));
        }, "rocBLAS", "N/A");
        
        // Fix speedup for baseline itself
        results.back().speedup = 1.0; 
    }

    if (test_recursive) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                recursive_trsm(handle, d_A, lda, d_B, ldb, n, m, nb);
            }, "Recursive", to_string(nb));
        }
    }

    if (test_iterative) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                iterative_trsm_right_looking(handle, d_A, lda, d_B, ldb, n, m, nb);
            }, "Iterative RL", to_string(nb));
        }
    }

    // ========================================================================
    // 5. Print Summary Table
    // ========================================================================
    cout << "\n";
    cout << string(85, '=') << "\n";
    cout << "TRSM Benchmark Summary (N = " << n << ", M = " << m << ")\n";
    cout << string(85, '=') << "\n";

    // Table Header
    cout << left 
         << setw(15) << "Method" 
         << " | " << setw(5) << "NB"
         << " | " << setw(11) << "Time (ms)"
         << " | " << setw(11) << "GFLOPS"
         << " | " << setw(9) << "Speedup"
         << " | " << "Error (AX=B)\n";
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
    // 6. Cleanup
    // ========================================================================
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ============================================================================
// MAIN
// ============================================================================
int main(int argc, char** argv)
{
    int n  = 4096;
    int m  = 4096;
    
    bool enable_error   = false;
    bool test_recursive = true;
    bool test_rocblas   = true;
    bool test_iterative = true;

    vector<int> nbs;

    // 为了与 SYRK 保持完全一致的体验，将解析方式统一，末尾参数全部视为 NB。
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) m = atoi(argv[2]);
    if (argc >= 4) enable_error = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);
    if (argc >= 6) test_rocblas = atoi(argv[5]);
    if (argc >= 7) test_iterative = atoi(argv[6]);

    for (int i = 7; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }
    if (nbs.empty()) {
        nbs.push_back(256);
    }

    srand(2026);

    run_all_benchmarks(
        n, m, nbs, 
        enable_error, 
        test_recursive, 
        test_rocblas, 
        test_iterative);

    return 0;
}