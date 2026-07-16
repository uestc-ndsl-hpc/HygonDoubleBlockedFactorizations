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

double calculate_gemm_gflops(int m, int n, int k, double time_ms);

namespace {

struct GemmShapeOptions {
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int m_tile;
    int n_tile;
    int k_tile;
};

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

int ceil_div(int value, int divisor)
{
    return (value + divisor - 1) / divisor;
}

void run_tiled_gemm(rocblas_handle handle,
                    const GemmShapeOptions& options,
                    float* A,
                    float* B,
                    float* C)
{
    const float alpha = -1.0f;
    const float beta = 1.0f;

    for (int column = 0; column < options.n; column += options.n_tile) {
        const int n = std::min(options.n_tile, options.n - column);
        for (int row = 0; row < options.m; row += options.m_tile) {
            const int m = std::min(options.m_tile, options.m - row);
            for (int inner = 0; inner < options.k; inner += options.k_tile) {
                const int k = std::min(options.k_tile, options.k - inner);
                ROCBLAS_CHECK(rocblas_sgemm(
                    handle,
                    rocblas_operation_none,
                    rocblas_operation_none,
                    m,
                    n,
                    k,
                    &alpha,
                    A + row + static_cast<size_t>(inner) * options.lda,
                    options.lda,
                    B + inner + static_cast<size_t>(column) * options.ldb,
                    options.ldb,
                    &beta,
                    C + row + static_cast<size_t>(column) * options.ldc,
                    options.ldc));
            }
        }
    }

    return;
}

void run_explicit_shape_benchmark(const GemmShapeOptions& options,
                                  int num_warmup,
                                  int num_iters)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t size_A = static_cast<size_t>(options.lda) * options.k;
    const size_t size_B = static_cast<size_t>(options.ldb) * options.n;
    const size_t size_C = static_cast<size_t>(options.ldc) * options.n;
    DeviceBuffer<float> d_A(size_A);
    DeviceBuffer<float> d_B(size_B);
    DeviceBuffer<float> d_C(size_C);

    HIP_CHECK(hipMemsetAsync(d_A, 1, size_A * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(d_B, 2, size_B * sizeof(float), stream));
    HIP_CHECK(hipMemsetAsync(d_C, 3, size_C * sizeof(float), stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int iteration = 0; iteration < num_warmup; ++iteration) {
        run_tiled_gemm(handle, options, d_A, d_B, d_C);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    GPUTimer timer;
    const float total_time_ms = timer.measure(stream, [&]() {
        for (int iteration = 0; iteration < num_iters; ++iteration) {
            run_tiled_gemm(handle, options, d_A, d_B, d_C);
        }
    });
    const double average_time_ms = total_time_ms / num_iters;
    const double gflops = calculate_gemm_gflops(
        options.m, options.n, options.k, average_time_ms);
    const int calls_per_iteration =
        ceil_div(options.m, options.m_tile)
        * ceil_div(options.n, options.n_tile)
        * ceil_div(options.k, options.k_tile);

    cout << "==================================================\n";
    cout << "Explicit GEMM shape benchmark\n";
    cout << "M/N/K              : " << options.m << "/"
         << options.n << "/" << options.k << "\n";
    cout << "LDA/LDB/LDC         : " << options.lda << "/"
         << options.ldb << "/" << options.ldc << "\n";
    cout << "M/N/K tiles         : " << options.m_tile << "/"
         << options.n_tile << "/" << options.k_tile << "\n";
    cout << "Calls per iteration : " << calls_per_iteration << "\n";
    cout << "Warmup/iterations   : " << num_warmup << "/"
         << num_iters << "\n";
    cout << "Average time (ms)   : " << fixed << setprecision(3)
         << average_time_ms << "\n";
    cout << "Throughput (GFLOPS) : " << fixed << setprecision(2)
         << gflops << "\n";
    cout << "==================================================\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));

    return;
}

bool parse_explicit_shape(int argc, char** argv, GemmShapeOptions& options)
{
    if (argc < 5 || string(argv[1]) != "--shape") {
        return false;
    }

    options.m = atoi(argv[2]);
    options.n = atoi(argv[3]);
    options.k = atoi(argv[4]);
    options.lda = options.m;
    options.ldb = options.k;
    options.ldc = options.m;
    options.m_tile = options.m;
    options.n_tile = options.n;
    options.k_tile = options.k;

    for (int index = 5; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }

        const string argument = argv[index];
        const int value = atoi(argv[index + 1]);
        if (argument == "--lda") {
            options.lda = value;
        } else if (argument == "--ldb") {
            options.ldb = value;
        } else if (argument == "--ldc") {
            options.ldc = value;
        } else if (argument == "--m-tile") {
            options.m_tile = value;
        } else if (argument == "--n-tile") {
            options.n_tile = value;
        } else if (argument == "--k-tile") {
            options.k_tile = value;
        } else {
            return false;
        }
    }

    return options.m > 0 && options.n > 0 && options.k > 0
        && options.lda >= options.m
        && options.ldb >= options.k
        && options.ldc >= options.m
        && options.m_tile > 0
        && options.n_tile > 0
        && options.k_tile > 0;
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

    GemmShapeOptions shape_options{};
    if (argc >= 2 && string(argv[1]) == "--shape") {
        if (!parse_explicit_shape(argc, argv, shape_options)) {
            cerr << "Usage: benchmark_gemm --shape M N K "
                 << "[--lda LDA] [--ldb LDB] [--ldc LDC] "
                 << "[--m-tile TM] [--n-tile TN] [--k-tile TK]\n";
            return 1;
        }

        shape_options.m_tile = std::min(
            shape_options.m_tile, shape_options.m);
        shape_options.n_tile = std::min(
            shape_options.n_tile, shape_options.n);
        shape_options.k_tile = std::min(
            shape_options.k_tile, shape_options.k);
        run_explicit_shape_benchmark(
            shape_options, num_warmup, num_iters);
        return 0;
    }

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
