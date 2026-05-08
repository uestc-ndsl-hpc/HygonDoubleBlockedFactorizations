#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "potrf/recursive_potrf.h"

using namespace std;

// ======================================================
// HIP CHECK
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

// ======================================================
// ROCBLAS CHECK
// ======================================================

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

double calculate_gflops(int n, double time_ms)
{
    if (time_ms <= 0.0)
        return 0.0;

    double flops =
        (1.0 / 3.0) *
        static_cast<double>(n) *
        static_cast<double>(n) *
        static_cast<double>(n);

    return (flops / (time_ms / 1000.0)) / 1e9;
}

// ======================================================
// Generate SPD Matrix
// Column-major
// ======================================================

template <typename T>
void generate_spd_matrix(T* A, int n, int lda)
{
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            A[j * lda + i] =
                static_cast<T>(rand()) /
                static_cast<T>(RAND_MAX);
        }
    }

    // Make SPD
    for (int i = 0; i < n; ++i)
    {
        A[i * lda + i] += static_cast<T>(n);
    }
}

// ======================================================
// Verify against Double Precision Reference
// 计算相对 Frobenius 范数误差: ||L_res - L_ref_double||_F / ||L_ref_double||_F
// 无论输入是 float 还是 double，标准答案全是 double
// ======================================================

template <typename T>
double verify_potrf_against_double_ref(const T* L_res,
                                       const double* L_ref_double,
                                       int n,
                                       int lda)
{
    double diff_norm_sq = 0.0;
    double ref_norm_sq = 0.0;

    for (int j = 0; j < n; ++j)
    {
        // 仅对比下三角 (i >= j)
        for (int i = j; i < n; ++i)
        {
            double val_res = static_cast<double>(L_res[j * lda + i]);
            double val_ref = L_ref_double[j * lda + i];

            double diff = val_res - val_ref;

            diff_norm_sq += diff * diff;
            ref_norm_sq  += val_ref * val_ref;
        }
    }

    double diff_norm = std::sqrt(diff_norm_sq);
    double ref_norm = std::sqrt(ref_norm_sq);

    return diff_norm / (ref_norm + 1e-12);
}

// ======================================================
// Benchmark
// ======================================================

template <typename T>
void run_potrf_comparison(int n, int nb)
{
    size_t num_elements = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t bytes = num_elements * sizeof(T);

    // ==================================================
    // Create Handle
    // ==================================================

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    int* d_info = nullptr;
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    // ==================================================
    // Initialize Original Matrix
    // ==================================================

    vector<T> h_A(num_elements);
    generate_spd_matrix(h_A.data(), n, n);

    // ==================================================
    // Generate Golden Reference using rocsolver_dpotrf
    // 强制使用双精度求出绝对标准结果 (FP64 Reference)
    // ==================================================
    vector<double> h_L_ref_double(num_elements);
    {
        vector<double> h_A_double(num_elements);
        for (size_t i = 0; i < num_elements; ++i)
        {
            h_A_double[i] = static_cast<double>(h_A[i]); 
        }

        double* d_A_double = nullptr;
        HIP_CHECK(hipMalloc(&d_A_double, num_elements * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_A_double, h_A_double.data(), num_elements * sizeof(double), hipMemcpyHostToDevice));

        // 运行双精度的 dpotrf 获取绝对正确答案
        ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, d_A_double, n, d_info));

        HIP_CHECK(hipMemcpy(h_L_ref_double.data(), d_A_double, num_elements * sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_A_double));
    }

    // ==================================================
    // Allocate Device for Benchmark
    // ==================================================

    T* d_A = nullptr;
    T* d_A_ref = nullptr;

    HIP_CHECK(hipMalloc(&d_A, bytes));
    HIP_CHECK(hipMalloc(&d_A_ref, bytes));

    // ==================================================
    // Events
    // ==================================================

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // ==================================================
    // Benchmark Lambda
    // 运行传入的函数，并统一将结果和双精度答案比较
    // ==================================================

    auto benchmark = [&](auto func, T* d_ptr, const string& label)
    {
        HIP_CHECK(hipMemcpy(d_ptr, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipEventRecord(start, 0));

        func();

        HIP_CHECK(hipEventRecord(stop, 0));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        double gflops = calculate_gflops(n, ms);

        // 拷回结果，用于验证
        vector<T> h_res(num_elements);
        HIP_CHECK(hipMemcpy(h_res.data(), d_ptr, bytes, hipMemcpyDeviceToHost));

        // 任何函数跑出的结果，一律与 FP64 的 Golden Reference 计算范数误差
        double rel_error = verify_potrf_against_double_ref(
            h_res.data(),
            h_L_ref_double.data(),
            n,
            n);

        cout << left
             << setw(24) << label
             << " | "
             << setw(10) << fixed << setprecision(2)
             << ms << " ms"
             << " | "
             << setw(10) << fixed << setprecision(2)
             << gflops << " GFLOPS"
             << " | Rel.Err(vs FP64): "
             << scientific << setprecision(3)
             << rel_error
             << endl;

        return static_cast<double>(ms);
    };

    // ==================================================
    // Print Header
    // ==================================================

    string type_name = (sizeof(T) == 4) ? "Float32" : "Float64";

    cout << endl;
    cout << "==========================================================" << endl;
    cout << "POTRF Benchmark (" << type_name << ", N=" << n << ", NB=" << nb << ")" << endl;
    cout << "==========================================================" << endl;

    // ==================================================
    // 1. 测试 rocSOLVER POTRF 速度与误差
    // ==================================================

    double t_std = benchmark(
        [&]()
        {
            if constexpr (is_same_v<T, float>)
            {
                ROCBLAS_CHECK(rocsolver_spotrf(handle, rocblas_fill_lower, n, d_A_ref, n, d_info));
            }
            else
            {
                ROCBLAS_CHECK(rocsolver_dpotrf(handle, rocblas_fill_lower, n, d_A_ref, n, d_info));
            }
        },
        d_A_ref,
        "rocSOLVER POTRF");

    // ==================================================
    // 2. 测试 Recursive POTRF 速度与误差
    // ==================================================

    double t_rec = benchmark(
        [&]()
        {
            if constexpr (is_same_v<T, float>)
            {
                recursive_potrf(handle, d_A, n, n, nb, d_info);
            }
            else
            {
                recursive_potrf_double(handle, d_A, n, n, nb, d_info);
            }
        },
        d_A,
        "Recursive POTRF");

    // ==================================================
    // Speedup
    // ==================================================

    cout << "----------------------------------------------------------" << endl;
    cout << "Speedup = " << fixed << setprecision(2) << t_std / (t_rec + 1e-12) << "x" << endl;
    cout << "==========================================================" << endl;

    // ==================================================
    // Cleanup
    // ==================================================

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_A_ref));
    HIP_CHECK(hipFree(d_info));

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

// ======================================================
// Main
// ======================================================

int main(int argc, char** argv)
{
    int n = 4096;
    int nb = 256;

    if (argc >= 2)
    {
        n = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        nb = atoi(argv[2]);
    }

    srand(2026);

    run_potrf_comparison<float>(n, nb);

    return 0;
}