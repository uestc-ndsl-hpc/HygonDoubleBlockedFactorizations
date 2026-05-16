#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "trsm/recursive_trsm.h"
#include "trsm/iterative_trsm.h"

#include "utils/error_check.h"
#include "utils/stream_pool.h"

using namespace std;

// ============================================================================
// UTILITIES
// ============================================================================

// Calculate GFLOPS for Right-TRSM: X(n x m) * L^T(m x m) = B(n x m)
double calculate_gflops(int n, int m, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;

    // Right-TRSM 运算量为 n * m^2
    double total_flops =
        static_cast<double>(n) * m * m;

    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

// Generate Stable Lower Triangular Matrix
void generate_stable_lower_triangular(
    float* A,
    int dim,
    int lda)
{
    for (int j = 0; j < dim; j++) {
        for (int i = 0; i < dim; i++) {

            if (i < j) {
                A[j * lda + i] = 0.0f;
            }
            else if (i == j) {
                A[j * lda + i] =
                    dim + static_cast<float>(rand()) / RAND_MAX;
            }
            else {
                A[j * lda + i] =
                    static_cast<float>(rand()) / RAND_MAX;
            }
        }
    }
}

// Generate Random Matrix
void generate_random_matrix(
    float* B,
    int n,
    int m,
    int ldb)
{
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            B[j * ldb + i] =
                static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

// Relative Frobenius Error
template<typename T_REF>
double verify_relative_error(
    const vector<T_REF>& ref,
    const vector<float>& res)
{
    double diff_norm = 0.0;
    double ref_norm  = 0.0;

    size_t size = ref.size();

    for (size_t i = 0; i < size; i++) {

        double diff =
            static_cast<double>(ref[i]) -
            static_cast<double>(res[i]);

        diff_norm += diff * diff;

        ref_norm +=
            static_cast<double>(ref[i]) *
            static_cast<double>(ref[i]);
    }

    return sqrt(diff_norm) /
           (sqrt(ref_norm) + 1e-20);
}

// Struct to store benchmark results
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
    int n,
    int m,
    const vector<int>& nbs,
    bool enable_error,
    bool test_recursive,
    bool test_rocblas,
    bool test_iterative)
{
    rocblas_handle handle;

    ROCBLAS_CHECK(
        rocblas_create_handle(&handle));

    // Right-TRSM 中，矩阵 A (即 L) 的维度为 m x m
    size_t size_A =
        static_cast<size_t>(m) *
        m *
        sizeof(float);

    size_t size_B =
        static_cast<size_t>(n) *
        m *
        sizeof(float);

    int lda = m; // L 的 leading dimension
    int ldb = n; // B 的 leading dimension

    StreamPool pool(4);

    // ========================================================================
    // 1. Host Memory
    // ========================================================================

    vector<float> h_A(m * m);
    vector<float> h_B(n * m);
    vector<float> h_X(n * m);

    generate_stable_lower_triangular(
        h_A.data(),
        m,
        lda);

    generate_random_matrix(
        h_B.data(),
        n,
        m,
        ldb);

    // ========================================================================
    // 2. Device Memory
    // ========================================================================

    float* d_A = nullptr;
    float* d_B = nullptr;

    HIP_CHECK(
        hipMalloc(&d_A, size_A));

    HIP_CHECK(
        hipMalloc(&d_B, size_B));

    HIP_CHECK(
        hipMemcpy(
            d_A,
            h_A.data(),
            size_A,
            hipMemcpyHostToDevice));

    float alpha = 1.0f;

    // ========================================================================
    // 3. Golden Reference Generation
    // ========================================================================

    vector<double> h_X_gold(n * m);
    vector<float>  h_X_ref_f32(n * m);

    bool has_fp64_ref = false;
    bool has_f32_ref  = false;

    if (enable_error) {

        cout << ">>> Generating Golden Reference...\n";

        // 防止矩阵过大致显存溢出
        bool do_fp64_verify = (std::max(n, m) <= 16384);

        // --------------------------------------------------------------------
        // FP64 Reference
        // --------------------------------------------------------------------

        if (do_fp64_verify) {

            double* d_A_d = nullptr;
            double* d_B_d = nullptr;

            vector<double> h_A_d(m * m);
            vector<double> h_B_d(n * m);

            for (size_t i = 0; i < h_A.size(); ++i)
                h_A_d[i] = static_cast<double>(h_A[i]);

            for (size_t i = 0; i < h_B.size(); ++i)
                h_B_d[i] = static_cast<double>(h_B[i]);

            HIP_CHECK(
                hipMalloc(
                    &d_A_d,
                    static_cast<size_t>(m) *
                    m *
                    sizeof(double)));

            HIP_CHECK(
                hipMalloc(
                    &d_B_d,
                    static_cast<size_t>(n) *
                    m *
                    sizeof(double)));

            HIP_CHECK(
                hipMemcpy(
                    d_A_d,
                    h_A_d.data(),
                    static_cast<size_t>(m) *
                    m *
                    sizeof(double),
                    hipMemcpyHostToDevice));

            HIP_CHECK(
                hipMemcpy(
                    d_B_d,
                    h_B_d.data(),
                    static_cast<size_t>(n) *
                    m *
                    sizeof(double),
                    hipMemcpyHostToDevice));

            double alpha_d = 1.0;

            // 调用 FP64 Right-TRSM Baseline 作为标答参考
            ROCBLAS_CHECK(
                rocblas_dtrsm(
                    handle,
                    rocblas_side_right,          // 右乘
                    rocblas_fill_lower,
                    rocblas_operation_transpose, // 转置 (L^T)
                    rocblas_diagonal_non_unit,
                    n,
                    m,
                    &alpha_d,
                    d_A_d,
                    lda,
                    d_B_d,
                    ldb));

            HIP_CHECK(
                hipMemcpy(
                    h_X_gold.data(),
                    d_B_d,
                    static_cast<size_t>(n) *
                    m *
                    sizeof(double),
                    hipMemcpyDeviceToHost));

            HIP_CHECK(hipFree(d_A_d));
            HIP_CHECK(hipFree(d_B_d));

            has_fp64_ref = true;
        }

        // --------------------------------------------------------------------
        // FP32 Reference
        // --------------------------------------------------------------------

        else {

            HIP_CHECK(
                hipMemcpy(
                    d_B,
                    h_B.data(),
                    size_B,
                    hipMemcpyHostToDevice));

            // 调用 FP32 Right-TRSM Baseline 作为标答参考
            ROCBLAS_CHECK(
                rocblas_strsm(
                    handle,
                    rocblas_side_right,          // 右乘
                    rocblas_fill_lower,
                    rocblas_operation_transpose, // 转置 (L^T)
                    rocblas_diagonal_non_unit,
                    n,
                    m,
                    &alpha,
                    d_A,
                    lda,
                    d_B,
                    ldb));

            HIP_CHECK(
                hipMemcpy(
                    h_X_ref_f32.data(),
                    d_B,
                    size_B,
                    hipMemcpyDeviceToHost));

            has_f32_ref = true;
        }
    }

    // ========================================================================
    // 4. Benchmark Preparation
    // ========================================================================

    hipEvent_t start_event;
    hipEvent_t stop_event;

    HIP_CHECK(
        hipEventCreate(&start_event));

    HIP_CHECK(
        hipEventCreate(&stop_event));

    vector<BenchmarkResult> results;

    double baseline_time = -1.0;

    // ========================================================================
    // Helper Lambda
    // ========================================================================

    auto execute_benchmark =
    [&](auto func,
        const string& name,
        const string& nb_str)
    {
        // Reset B before warmup
        HIP_CHECK(
            hipMemcpy(
                d_B,
                h_B.data(),
                size_B,
                hipMemcpyHostToDevice));

        HIP_CHECK(
            hipDeviceSynchronize());

        // Warmup
        func();

        // Reset B before actual run
        HIP_CHECK(
            hipMemcpy(
                d_B,
                h_B.data(),
                size_B,
                hipMemcpyHostToDevice));

        HIP_CHECK(
            hipDeviceSynchronize());

        // Timing
        HIP_CHECK(
            hipEventRecord(start_event, 0));

        func();

        HIP_CHECK(
            hipEventRecord(stop_event, 0));

        HIP_CHECK(
            hipEventSynchronize(stop_event));

        float ms = 0.0f;

        HIP_CHECK(
            hipEventElapsedTime(
                &ms,
                start_event,
                stop_event));

        double gflops =
            calculate_gflops(n, m, ms);

        double error = -1.0;

        // --------------------------------------------------------------------
        // Error Verification
        // --------------------------------------------------------------------

        if (enable_error) {

            HIP_CHECK(
                hipMemcpy(
                    h_X.data(),
                    d_B,
                    size_B,
                    hipMemcpyDeviceToHost));

            if (has_fp64_ref) {

                error =
                    verify_relative_error(
                        h_X_gold,
                        h_X);
            }
            else if (has_f32_ref) {

                error =
                    verify_relative_error(
                        h_X_ref_f32,
                        h_X);
            }
        }

        double speedup =
            (baseline_time > 0.0)
            ? (baseline_time / ms)
            : 0.0;

        results.push_back({
            name,
            nb_str,
            static_cast<double>(ms),
            gflops,
            speedup,
            error
        });

        return static_cast<double>(ms);
    };

    // ========================================================================
    // 5. Run Benchmarks
    // ========================================================================

    cout << ">>> Running Benchmarks..." << endl;

    // ------------------------------------------------------------------------
    // rocBLAS Baseline
    // ------------------------------------------------------------------------

    if (test_rocblas) {

        baseline_time =
            execute_benchmark(
                [&]() {

                    // 右乘、转置
                    ROCBLAS_CHECK(
                        rocblas_strsm(
                            handle,
                            rocblas_side_right,
                            rocblas_fill_lower,
                            rocblas_operation_transpose,
                            rocblas_diagonal_non_unit,
                            n,
                            m,
                            &alpha,
                            d_A,
                            lda,
                            d_B,
                            ldb));
                },
                "rocBLAS",
                "N/A");

        results.back().speedup = 1.0;
    }

    // ------------------------------------------------------------------------
    // Recursive
    // ------------------------------------------------------------------------

    if (test_recursive) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                recursive_trsm(handle,
                            &pool,
                            d_A, lda,
                            d_B, ldb,
                            n, m, nb);
            }, "Recursive", to_string(nb));
        }
    }

    // ------------------------------------------------------------------------
    // Iterative
    // ------------------------------------------------------------------------

    if (test_iterative) {

        for (int nb : nbs) {

            execute_benchmark(
                [&]() {

                    iterative_trsm_right_looking(
                        handle,
                        d_A,
                        lda,
                        d_B,
                        ldb,
                        n,
                        m,
                        nb);
                },
                "Iterative RL",
                to_string(nb));
        }
    }

    // ========================================================================
    // 6. Print Summary Table
    // ========================================================================

    cout << "\n";

    cout << string(85, '=') << "\n";

    cout
        << "TRSM Benchmark Summary (Right-TRSM: X * L^T = B) "
        << "\n(N = " << n
        << ", M = " << m
        << ")\n";

    cout << string(85, '=') << "\n";

    // Header
    cout << left
         << setw(15) << "Method"
         << " | "
         << setw(5)  << "NB"
         << " | "
         << setw(11) << "Time (ms)"
         << " | "
         << setw(11) << "GFLOPS"
         << " | "
         << setw(9)  << "Speedup"
         << " | "
         << "Relative Error\n";

    cout << string(85, '-') << "\n";

    // Rows
    for (const auto& res : results) {

        cout << left
             << setw(15) << res.method
             << " | "
             << setw(5) << res.nb_str
             << " | "
             << right
             << setw(9)
             << fixed
             << setprecision(3)
             << res.time_ms
             << "  | "
             << setw(9)
             << fixed
             << setprecision(2)
             << res.gflops
             << "  | ";

        if (res.speedup > 0) {

            cout
                << setw(7)
                << fixed
                << setprecision(2)
                << res.speedup
                << "x ";
        }
        else {

            cout << setw(8) << "N/A ";
        }

        cout << " | ";

        if (enable_error) {

            cout
                << scientific
                << setprecision(2)
                << res.error;
        }
        else {

            cout << "N/A";
        }

        cout << "\n";
    }

    cout << string(85, '=') << "\n";

    // ========================================================================
    // 7. Cleanup
    // ========================================================================

    HIP_CHECK(
        hipEventDestroy(start_event));

    HIP_CHECK(
        hipEventDestroy(stop_event));

    HIP_CHECK(
        hipFree(d_A));

    HIP_CHECK(
        hipFree(d_B));

    ROCBLAS_CHECK(
        rocblas_destroy_handle(handle));
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv)
{
    int n = 4096;
    int m = 4096;

    bool enable_error   = false;
    bool test_recursive = true;
    bool test_rocblas   = true;
    bool test_iterative = true;

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) m = atoi(argv[2]);

    if (argc >= 4)
        enable_error = atoi(argv[3]);

    if (argc >= 5)
        test_recursive = atoi(argv[4]);

    if (argc >= 6)
        test_rocblas = atoi(argv[5]);

    if (argc >= 7)
        test_iterative = atoi(argv[6]);

    for (int i = 7; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }

    if (nbs.empty()) {
        nbs.push_back(256);
    }

    srand(2026);

    run_all_benchmarks(
        n,
        m,
        nbs,
        enable_error,
        test_recursive,
        test_rocblas,
        test_iterative);

    return 0;
}