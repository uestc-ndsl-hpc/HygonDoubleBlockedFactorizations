#include <iostream>
#include <vector>
#include <iomanip>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "trsm/recursive_trsm.h"
#include "trsm/iterative_trsm.h"

using namespace std;

// HIP 错误检查宏
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << hipGetErrorString(error) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// rocBLAS 错误检查宏
#define ROCBLAS_CHECK(cmd) \
    do { \
        rocblas_status status = cmd; \
        if (status != rocblas_status_success) { \
            cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__ << ": " \
                 << rocblas_status_to_string(status) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * 计算吞吐量 (GFLOPS)
 * 计算公式: (n^2 * m) / (time_s * 1e9)
 */
double calculate_gflops(int n, int m, double time_ms) {
    if (time_ms <= 0) return 0;
    double total_flops = static_cast<double>(n) * n * m; 
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

// --- 辅助函数：矩阵生成与校验 ---
void generate_stable_lower_triangular(float* A, int n, int lda) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            if (i < j) A[j * lda + i] = 0.0f;
            else if (i == j) A[j * lda + i] = n + static_cast<float>(rand()) / (float)RAND_MAX;
            else A[j * lda + i] = static_cast<float>(rand()) / (float)RAND_MAX;
        }
    }
}

void generate_random_matrix(float* B, int n, int m, int ldb) {
    for (int i = 0; i < n * m; i++) B[i] = static_cast<float>(rand()) / (float)RAND_MAX - 0.5f;
}

double verify_result(float* h_A, float* h_X, float* h_B, int n, int m, int lda, int ldb) {
    double max_err = 0.0, max_b = 0.0;
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int k = 0; k <= i; k++) sum += (double)h_A[k * lda + i] * h_X[j * ldb + k];
            double diff = std::abs(sum - h_B[j * ldb + i]);
            if (diff > max_err) max_err = diff;
            if (std::abs(h_B[j * ldb + i]) > max_b) max_b = std::abs(h_B[j * ldb + i]);
        }
    }
    return max_err / (max_b + 1e-10);
}

// --- 测试主逻辑 ---
void run_comparison(int n, int m, int nb) {
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    size_t size_A = (size_t)n * n * sizeof(float);
    size_t size_B = (size_t)n * m * sizeof(float);
    vector<float> h_A(n * n), h_B(n * m), h_X(n * m);
    
    generate_stable_lower_triangular(h_A.data(), n, n);
    generate_random_matrix(h_B.data(), n, m, n);

    float *d_A, *d_B;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));

    // 创建 HIP 事件
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    auto benchmark = [&](auto func, const string& label) -> double {
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice)); // 重置 B
        HIP_CHECK(hipDeviceSynchronize());
        
        // 记录开始事件
        HIP_CHECK(hipEventRecord(start_event, 0));
        
        func();
        
        // 记录结束事件
        HIP_CHECK(hipEventRecord(stop_event, 0));
        HIP_CHECK(hipEventSynchronize(stop_event));
        
        // 计算耗时（毫秒）
        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event));
        
        double gflops = calculate_gflops(n, m, ms);
        
        HIP_CHECK(hipMemcpy(h_X.data(), d_B, size_B, hipMemcpyDeviceToHost));
        double error = verify_result(h_A.data(), h_X.data(), h_B.data(), n, m, n, n);

        cout << left << setw(15) << label 
             << " | " << setw(10) << fixed << setprecision(2) << ms << " ms"
             << " | " << setw(10) << gflops << " GFLOPS"
             << " | Res: " << scientific << setprecision(2) << error << endl;
        return static_cast<double>(ms);
    };

    cout << "\n对比测试结果 (N=" << n << ", M=" << m << ", nb=" << nb << "):" << endl;
    cout << "--------------------------------------------------------------------------" << endl;

    // 1. 标准库 rocBLAS
    float alpha = 1.0f;
    double t_std = benchmark([&]() {
        ROCBLAS_CHECK(rocblas_strsm(handle, rocblas_side_left, rocblas_fill_lower, 
                                     rocblas_operation_none, rocblas_diagonal_non_unit, 
                                     n, m, &alpha, d_A, n, d_B, n));
    }, "rocBLAS (Std)");

    // 2. 递归分块版
    double t_rec = benchmark([&]() {
        recursive_trsm(handle, d_A, n, d_B, n, n, m, nb);
    }, "Recursive");

    // 3. 迭代右视版
    double t_iter = benchmark([&]() {
        iterative_trsm_right_looking(handle, d_A, n, d_B, n, n, m, nb);
    }, "Iterative RL");

    cout << "--------------------------------------------------------------------------" << endl;
    cout << "加速比 (Recursive vs Std): " << fixed << setprecision(2) << t_std / t_rec << "x" << endl;
    cout << "加速比 (Iterative vs Std): " << fixed << setprecision(2) << t_std / t_iter << "x" << endl;

    // 销毁 HIP 事件
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
    
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main(int argc, char** argv) {
    int n = 4096, m = 4096, nb = 256;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) m = atoi(argv[2]);
    if (argc >= 4) nb = atoi(argv[3]);

    srand(2026);
    run_comparison(n, m, nb);
    return 0;
}