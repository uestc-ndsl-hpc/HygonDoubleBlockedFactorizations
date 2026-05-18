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

// Generate SPD Matrix
template <typename T>
void generate_spd_matrix(T* A, int n, int lda)
{
    for (size_t j = 0; j < (size_t)n; ++j) {
        for (size_t i = 0; i < (size_t)n; ++i) {
            A[j * (size_t)lda + i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
    }

    for (size_t i = 0; i < (size_t)n; ++i) {
        A[i * (size_t)lda + i] += static_cast<T>(n);
    }
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
    // 最多采样下三角元素总数
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

        // 仅采样并验证下三角元素
        if (row < col) {
            continue; 
        }

        double reconstructed = 0.0;

        // A = L * L^T  =>  A_{row, col} = \sum_{k=0}^{col} L_{row, k} * L_{col, k}
        for (int k = 0; k <= col; ++k) {
            double Lik = static_cast<double>(L_res[k * lda + row]);
            double Ljk = static_cast<double>(L_res[k * lda + col]);
            reconstructed += Lik * Ljk;
        }

        double a_val = static_cast<double>(A_orig[col * lda + row]);
        double diff = a_val - reconstructed;

        // 对非对角线元素权重乘以2，为了等效估计完整对称矩阵的 Frobenius 范数
        double factor = (row == col) ? 1.0 : 2.0;

        diff_norm_sq += factor * diff * diff;
        a_norm_sq    += factor * a_val * a_val;

        ++s;
    }

    return sqrt(diff_norm_sq) / (sqrt(a_norm_sq) + 1e-12);
}

// ============================================================================
// RESULT STRUCT
// ============================================================================

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
    int* info)
{
    if constexpr (std::is_same_v<T, float>) {
        recursive_potrf(handle, pool, A, lda, n, nb, info);
    } else {
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
    bool enable_error,
    bool test_rocsolver,
    bool test_recursive,
    int error_type)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    StreamPool pool(4);

    size_t num_elements = (size_t)n * (size_t)n;
    size_t bytes = num_elements * sizeof(T);

    int* d_info = nullptr;
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    vector<T> h_A(num_elements);
    generate_spd_matrix(h_A.data(), n, n);

    T* d_A = nullptr;
    HIP_CHECK(hipMalloc(&d_A, bytes));

    vector<double> h_L_ref_double;

    if (enable_error && (error_type == 0 || error_type == 2)) {
        cout << ">>> Generating FP64 Golden Reference...\n";

        h_L_ref_double.resize(num_elements);
        vector<double> h_A_double(num_elements);

        for (size_t i = 0; i < num_elements; ++i)
            h_A_double[i] = static_cast<double>(h_A[i]);

        double* d_A_double = nullptr;
        HIP_CHECK(hipMalloc(&d_A_double, num_elements * sizeof(double)));

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

        HIP_CHECK(hipFree(d_A_double));
    }

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    vector<BenchmarkResult> results;
    double baseline_time = -1.0;

    auto execute_benchmark = [&](auto func, const string& name, const string& nb_str)
    {
        // Warmup
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());
        func();

        // Actual execution
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipEventRecord(start, 0));
        func();
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        double gflops = calculate_gflops(n, ms);

        double forward_error  = -1.0;
        double backward_error = -1.0;

        if (enable_error) {
            vector<T> h_res(num_elements);
            HIP_CHECK(hipMemcpy(h_res.data(), d_A, bytes, hipMemcpyDeviceToHost));

            if (error_type == 0 || error_type == 2) {
                forward_error = verify_potrf_against_double_ref(
                    h_res.data(),
                    h_L_ref_double.data(),
                    n,
                    n);
            }

            if (error_type == 1 || error_type == 2) {
                backward_error = verify_backward_error_sampled(
                    h_res.data(),
                    h_A.data(),
                    n,
                    n,
                    4096);
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

    cout << ">>> Running Benchmarks...\n";

    if (test_rocsolver) {
        baseline_time = execute_benchmark([&]() {
            call_rocsolver_potrf<T>(handle, rocblas_fill_lower, n, d_A, n, d_info);
        }, "rocSOLVER", "N/A");

        results.back().speedup = 1.0;
    }

    if (test_recursive) {
        for (int nb : nbs) {
            execute_benchmark([&]() {
                call_recursive_potrf<T>(handle, &pool, d_A, n, n, nb, d_info);
            }, "Recursive", to_string(nb));
        }
    }

    // =========================================================================
    // PRINT
    // =========================================================================

    cout << "\n";
    cout << string(110, '=') << "\n";
    cout << "POTRF Benchmark Summary (N = " << n << ")\n";
    cout << string(110, '=') << "\n";

    cout << left
         << setw(15) << "Method"
         << " | " << setw(6) << "NB"
         << " | " << setw(12) << "Time(ms)"
         << " | " << setw(12) << "GFLOPS"
         << " | " << setw(10) << "Speedup"
         << " | ";

    if (error_type == 0)
        cout << "Forward Error\n";
    else if (error_type == 1)
        cout << "Backward Error\n";
    else
        cout << "Forward / Backward\n";

    cout << string(110, '-') << "\n";

    for (const auto& res : results) {
        cout << left << setw(15) << res.method
             << " | " << setw(6) << res.nb_str
             << " | " << right << setw(10) << fixed << setprecision(3) << res.time_ms << "  "
             << " | " << right << setw(10) << fixed << setprecision(2) << res.gflops << "  "
             << " | ";

        if (res.speedup > 0)
            cout << right << setw(8) << fixed << setprecision(2) << res.speedup << "x ";
        else
            cout << setw(9) << "N/A ";

        cout << " | ";

        if (!enable_error) {
            cout << "N/A";
        }
        else {
            if (error_type == 0) {
                cout << scientific << setprecision(2) << res.forward_error;
            }
            else if (error_type == 1) {
                cout << scientific << setprecision(2) << res.backward_error;
            }
            else {
                cout << scientific << setprecision(2)
                     << res.forward_error << " / "
                     << res.backward_error;
            }
        }

        cout << "\n";
    }

    cout << string(110, '=') << "\n";

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
    int error_type      = 0;

    // 0=float, 1=double
    int precision_type = 0;

    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) enable_error = atoi(argv[2]);
    if (argc >= 4) test_rocsolver = atoi(argv[3]);
    if (argc >= 5) test_recursive = atoi(argv[4]);
    if (argc >= 6) error_type = atoi(argv[5]);
    if (argc >= 7) precision_type = atoi(argv[6]);

    // nb 从第 7 个参数后开始
    for (int i = 7; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }

    if (nbs.empty()) {
        nbs.push_back(256);
    }

    srand(2026);

    if (precision_type == 1) {
        cout << "Running benchmark in DOUBLE precision\n";
        run_all_benchmarks<double>(
            n,
            nbs,
            enable_error,
            test_rocsolver,
            test_recursive,
            error_type);
    }
    else {
        cout << "Running benchmark in FLOAT precision\n";
        run_all_benchmarks<float>(
            n,
            nbs,
            enable_error,
            test_rocsolver,
            test_recursive,
            error_type);
    }

    return 0;
}