#include <iostream>
#include <cstdlib>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>
#include <random>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "utils/common/error_check.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/memory/device_buffer.h"
#include "utils/matrix/matrix_generator.h"

using namespace std;

namespace {

int read_iteration_count(const char* variable, int fallback, int minimum)
{
    const char* value = std::getenv(variable);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }

    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < minimum || parsed > 100000) {
        return fallback;
    }

    return static_cast<int>(parsed);
}

} // namespace

double calculate_gemm_gflops(int m, int n, int k, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double total_flops = 2.0 * static_cast<double>(m) * n * k;
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

void run_right_looking_gemm_benchmark(const vector<int>& k_steps, int num_warmup, int num_iters)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const int TOTAL_MATRIX_SIZE = 32768; 
    float alpha = -1.0f; 
    float beta  = 1.0f;

    cout << "=========================================================================================================\n";
    cout << "  Simulating GEMM Shapes in Right-Looking Cholesky Progress (Total Size = " << TOTAL_MATRIX_SIZE << ")\n";
    cout << "  Operation: C (M x N) = C - A (M x K) * B (K x N)\n";
    cout << "=========================================================================================================\n";
    cout << setw(10) << "Step (K)" 
         << setw(10) << "M" 
         << setw(10) << "N" 
         << setw(18) << "Matrix A Shape"
         << setw(18) << "Matrix C Shape"
         << setw(15) << "Avg Time (ms)" 
         << setw(15) << "GFLOPS" << endl;
    cout << "---------------------------------------------------------------------------------------------------------\n";

    GPUTimer timer;

    for (int k : k_steps) {
        int m = TOTAL_MATRIX_SIZE - k;
        int n = TOTAL_MATRIX_SIZE - k;

        if (m <= 0 || n <= 0) continue;

        int lda = m;
        int ldb = k;
        int ldc = m;

        size_t size_A = static_cast<size_t>(m) * k;
        size_t size_B = static_cast<size_t>(k) * n;
        size_t size_C = static_cast<size_t>(m) * n;

        DeviceBuffer<float> d_A(size_A);
        DeviceBuffer<float> d_B(size_B);
        DeviceBuffer<float> d_C(size_C);

        (void)hipMemsetAsync(d_A, 1, size_A * sizeof(float), stream);
        (void)hipMemsetAsync(d_B, 2, size_B * sizeof(float), stream);
        (void)hipMemsetAsync(d_C, 3, size_C * sizeof(float), stream);
        (void)hipStreamSynchronize(stream);

        for (int w = 0; w < num_warmup; ++w) {
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, 
                rocblas_operation_none, 
                m, n, k,
                &alpha,
                d_A, lda,
                d_B, ldb,
                &beta,
                d_C, ldc));
        }
        (void)hipStreamSynchronize(stream);

        float total_time_ms = timer.measure(stream, [&]() {
            for (int i = 0; i < num_iters; ++i) {
                (void)rocblas_sgemm(
                    handle,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    m, n, k,
                    &alpha,
                    d_A, lda,
                    d_B, ldb,
                    &beta,
                    d_C, ldc);
            }
        });
        
        double avg_time_ms = static_cast<double>(total_time_ms) / num_iters;
        double gflops = calculate_gemm_gflops(m, n, k, avg_time_ms);

        string a_shape = to_string(m) + "x" + to_string(k);
        string c_shape = to_string(m) + "x" + to_string(n);

        cout << setw(10) << k 
             << setw(10) << m 
             << setw(10) << n 
             << setw(18) << a_shape
             << setw(18) << c_shape
             << setw(15) << fixed << setprecision(3) << avg_time_ms 
             << setw(15) << fixed << setprecision(1) << gflops << endl;
    }
    cout << "=========================================================================================================\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

int main(int argc, char** argv)
{
    int num_warmup = read_iteration_count("BENCHMARK_WARMUP_ITERS", 2, 0);
    int num_iters = read_iteration_count("BENCHMARK_PROFILE_ITERS", 5, 1);

    vector<int> k_steps = {
        64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384
    };

    if (argc >= 2) {
        k_steps.clear();
        k_steps.push_back(atoi(argv[1]));
    }

    try {
        run_right_looking_gemm_benchmark(k_steps, num_warmup, num_iters);
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
        return 1;
    }

    return 0;
}
