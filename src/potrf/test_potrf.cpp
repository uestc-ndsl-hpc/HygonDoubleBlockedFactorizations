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

// ======================================================
// HIP CHECK & ROCBLAS CHECK
// ======================================================
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

// ======================================================
// GFLOPS
// ======================================================
double calculate_gflops(int n, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double flops = (1.0 / 3.0) * n * n * n;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// ======================================================
// Generate SPD Matrix
// ======================================================
template <typename T>
void generate_spd_matrix(T* A, int n, int lda) {
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[j * lda + i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    for (int i = 0; i < n; ++i)
        A[i * lda + i] += static_cast<T>(n);
}

// ======================================================
// FP64 Reference Error
// ======================================================
template <typename T>
double verify_potrf_against_double_ref(const T* L_res, const double* L_ref_double, int n, int lda) {
    double diff_norm_sq = 0.0, ref_norm_sq = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) { // 只比较下三角部分
            double val_res = static_cast<double>(L_res[j * lda + i]);
            double val_ref = L_ref_double[j * lda + i];
            double diff = val_res - val_ref;
            diff_norm_sq += diff * diff;
            ref_norm_sq  += val_ref * val_ref;
        }
    }
    return std::sqrt(diff_norm_sq) / (std::sqrt(ref_norm_sq) + 1e-12);
}

// ======================================================
// 模板接口自动分发
// ======================================================
template <typename T>
void call_rocsolver_potrf(rocblas_handle handle, rocblas_fill uplo, int n, T* A, int lda, int* info) {
    if constexpr (std::is_same_v<T, double>) {
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, uplo, n, A, lda, info));
    } else if constexpr (std::is_same_v<T, float>) {
        ROCBLAS_CHECK(rocsolver_spotrf(handle, uplo, n, A, lda, info));
    }
}

template <typename T>
void call_recursive_potrf(rocblas_handle handle, T* A, int n, int lda, int nb, int* info) {
    if constexpr (std::is_same_v<T, double>) {
        recursive_potrf_double(handle, A, lda, n, nb, info); 
    } else if constexpr (std::is_same_v<T, float>) {
        recursive_potrf(handle, A, lda, n, nb, info); 
    }
}

// ======================================================
// Benchmark 核心逻辑
// ======================================================
template <typename T>
void run_potrf_comparison(int n, const vector<int>& nbs, bool enable_error = false)
{
    size_t num_elements = (size_t)n * (size_t)n;
    size_t bytes = num_elements * sizeof(T);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    int* d_info = nullptr;
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    vector<T> h_A(num_elements);
    generate_spd_matrix(h_A.data(), n, n);

    // ==================================================
    // FP64 Reference (仅在开启 error_check 时才花费时间计算)
    // ==================================================
    vector<double> h_L_ref_double;
    if (enable_error) {
        h_L_ref_double.resize(num_elements);
        vector<double> h_A_double(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            h_A_double[i] = static_cast<double>(h_A[i]);
        }

        double* d_A_double = nullptr;
        HIP_CHECK(hipMalloc(&d_A_double, num_elements * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_A_double, h_A_double.data(),
                             num_elements * sizeof(double), hipMemcpyHostToDevice));

        ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, d_A_double, n, d_info));

        HIP_CHECK(hipMemcpy(h_L_ref_double.data(), d_A_double,
                            num_elements * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_A_double));
    }

    T* d_A = nullptr;
    T* d_A_ref = nullptr;
    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_A_ref, bytes));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    auto benchmark = [&](auto func, T* d_ptr, const string& label) -> double {
        HIP_CHECK(hipMemcpy(d_ptr, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipEventRecord(start, 0));
        func();
        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        double gflops = calculate_gflops(n, ms);

        double rel_error = 0.0;
        if (enable_error) {
            vector<T> h_res(num_elements);
            HIP_CHECK(hipMemcpy(h_res.data(), d_ptr, bytes, hipMemcpyDeviceToHost));
            rel_error = verify_potrf_against_double_ref(h_res.data(), h_L_ref_double.data(), n, n);
        }

        cout << left
             << setw(26) << label << " | "
             << setw(8) << fixed << setprecision(2) << ms << " ms | "
             << setw(8) << fixed << setprecision(2) << gflops << " GFLOPS | "
             << "Rel.Err: " << scientific << setprecision(3) << (enable_error ? rel_error : 0.0);

        return (double)ms;
    };

    cout << endl;
    cout << "=========================================================================" << endl;
    cout << "POTRF Benchmark (N = " << n << ", Precision = " 
         << (std::is_same_v<T, float> ? "FP32 (Single)" : "FP64 (Double)") << ")" << endl;
    cout << "Enable Error Check: " << (enable_error ? "ON (vs FP64 Result)" : "OFF") << endl;
    cout << "=========================================================================" << endl;

    double t_std = benchmark(
        [&]() { call_rocsolver_potrf<T>(handle, rocblas_fill_lower, n, d_A_ref, n, d_info); },
        d_A_ref,
        "rocSOLVER POTRF"
    );
    cout << " | Speedup: Ref" << endl;
    cout << "-------------------------------------------------------------------------" << endl;

    for (int nb : nbs) {
        string label = "Recursive POTRF (NB=" + to_string(nb) + ")";
        double t_rec = benchmark(
            [&]() { call_recursive_potrf<T>(handle, d_A, n, n, nb, d_info); },
            d_A,
            label
        );
        cout << " | Speedup: " << fixed << setprecision(2) << t_std / (t_rec + 1e-12) << "x" << endl;
    }
    
    cout << "=========================================================================" << endl;

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_A_ref));
    HIP_CHECK(hipFree(d_info));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ======================================================
// MAIN
// ======================================================
int main(int argc, char** argv)
{
    int n = 4096;
    bool enable_error = false;
    vector<int> nbs;

    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) enable_error = atoi(argv[2]);
    for (int i = 3; i < argc; ++i) {
        nbs.push_back(atoi(argv[i]));
    }

    if (nbs.empty()) nbs.push_back(256);

    srand(2026);

    run_potrf_comparison<float>(n, nbs, enable_error);

    return 0;
}