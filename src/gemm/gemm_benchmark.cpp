#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "gemm/mixed_gemm.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

using namespace std;

namespace {

constexpr int kTotalMatrixSize = 32768;
constexpr size_t kDefaultVerifyElementLimit = 32ull * 1024ull * 1024ull;

struct BenchmarkOptions {
    vector<int> k_steps = {
        64, 128, 256, 384, 512, 768, 1024, 1536,
        2048, 3072, 4096, 6144, 8192, 12288, 16384
    };
    int total_matrix_size = kTotalMatrixSize;
    int num_warmup = 2;
    int num_iters = 5;
    bool verify = false;
    size_t verify_element_limit = kDefaultVerifyElementLimit;
};

struct ErrorStats {
    bool skipped = true;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double rel_l2 = 0.0;
};

double calculate_gemm_gflops(int m, int n, int k, double time_ms)
{
    if (time_ms <= 0.0) return 0.0;
    double total_flops = 2.0 * static_cast<double>(m) * n * k;
    return (total_flops / (time_ms / 1000.0)) / 1e9;
}

bool parse_bool_env(const char* name, bool default_value)
{
    const char* value = getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    string text(value);
    return text == "1" || text == "ON" || text == "on" ||
           text == "TRUE" || text == "true" || text == "YES" || text == "yes";
}

size_t parse_size_env(const char* name, size_t default_value)
{
    const char* value = getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    char* end = nullptr;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (end == value || parsed == 0) {
        return default_value;
    }

    return static_cast<size_t>(parsed);
}

int parse_int_env(const char* name, int default_value)
{
    const char* value = getenv(name);
    if (value == nullptr) {
        return default_value;
    }

    int parsed = atoi(value);
    return parsed > 0 ? parsed : default_value;
}

void fill_matrix(vector<float>& matrix, int rows, int cols, unsigned int seed)
{
    mt19937 gen(seed);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            matrix[row + static_cast<size_t>(col) * rows] = dist(gen);
        }
    }
}

void copy_to_device(float* dst, const vector<float>& src)
{
    HIP_CHECK(hipMemcpy(dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice));
}

void reset_c(
    hipStream_t stream,
    float* d_C,
    size_t size_C,
    const vector<float>* h_C_initial)
{
    if (h_C_initial != nullptr) {
        copy_to_device(d_C, *h_C_initial);
    } else {
        HIP_CHECK(hipMemsetAsync(d_C, 0, size_C * sizeof(float), stream));
    }
}

double run_rocblas_sgemm(
    rocblas_handle handle,
    hipStream_t stream,
    int m,
    int n,
    int k,
    const float* d_A,
    const float* d_B,
    float* d_C,
    size_t size_C,
    const vector<float>* h_C_initial,
    int num_warmup,
    int num_iters)
{
    float alpha = -1.0f;
    float beta = 1.0f;
    int lda = m;
    int ldb = n;
    int ldc = m;

    for (int w = 0; w < num_warmup; ++w) {
        ROCBLAS_CHECK(rocblas_sgemm(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            k,
            &alpha,
            d_A,
            lda,
            d_B,
            ldb,
            &beta,
            d_C,
            ldc));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    reset_c(stream, d_C, size_C, h_C_initial);
    HIP_CHECK(hipStreamSynchronize(stream));

    GPUTimer timer;
    float total_time_ms = timer.measure(stream, [&]() {
        for (int i = 0; i < num_iters; ++i) {
            ROCBLAS_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none,
                rocblas_operation_transpose,
                m,
                n,
                k,
                &alpha,
                d_A,
                lda,
                d_B,
                ldb,
                &beta,
                d_C,
                ldc));
        }
    });

    return static_cast<double>(total_time_ms) / num_iters;
}

double run_hss_gemm(
    rocblas_handle handle,
    hipStream_t stream,
    int m,
    int n,
    int k,
    const float* d_A,
    const float* d_B,
    float* d_C,
    size_t size_C,
    const vector<float>* h_C_initial,
    int num_warmup,
    int num_iters)
{
    float alpha = -1.0f;
    float beta = 1.0f;
    int lda = m;
    int ldb = n;
    int ldc = m;

    for (int w = 0; w < num_warmup; ++w) {
        launch_hss_gemm(
            handle,
            stream,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            k,
            alpha,
            d_A,
            lda,
            d_B,
            ldb,
            beta,
            d_C,
            ldc);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    reset_c(stream, d_C, size_C, h_C_initial);
    HIP_CHECK(hipStreamSynchronize(stream));

    GPUTimer timer;
    float total_time_ms = timer.measure(stream, [&]() {
        for (int i = 0; i < num_iters; ++i) {
            launch_hss_gemm(
                handle,
                stream,
                rocblas_operation_none,
                rocblas_operation_transpose,
                m,
                n,
                k,
                alpha,
                d_A,
                lda,
                d_B,
                ldb,
                beta,
                d_C,
                ldc);
        }
    });

    HIP_CHECK(hipGetLastError());
    return static_cast<double>(total_time_ms) / num_iters;
}

ErrorStats compare_device_matrices(
    hipStream_t stream,
    const float* d_ref,
    const float* d_test,
    size_t elements,
    size_t element_limit)
{
    ErrorStats stats;
    if (elements == 0 || elements > element_limit) {
        return stats;
    }

    vector<float> ref(elements);
    vector<float> test(elements);
    HIP_CHECK(hipMemcpyAsync(ref.data(), d_ref, elements * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(test.data(), d_test, elements * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    double diff_l2 = 0.0;
    double ref_l2 = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;

    for (size_t i = 0; i < elements; ++i) {
        double r = static_cast<double>(ref[i]);
        double t = static_cast<double>(test[i]);
        double diff = fabs(t - r);
        double denom = max(fabs(r), 1.0e-20);

        max_abs = max(max_abs, diff);
        max_rel = max(max_rel, diff / denom);
        diff_l2 += diff * diff;
        ref_l2 += r * r;
    }

    stats.skipped = false;
    stats.max_abs = max_abs;
    stats.max_rel = max_rel;
    stats.rel_l2 = sqrt(diff_l2) / max(sqrt(ref_l2), numeric_limits<double>::min());
    return stats;
}

void print_usage(const char* program)
{
    cerr << "Usage: " << program << " [k_step]\n"
         << "Environment:\n"
         << "  GEMM_VERIFY=1                 Enable HSS vs SGEMM correctness check.\n"
         << "  GEMM_VERIFY_ELEMENT_LIMIT=N   Max C elements copied for verification.\n"
         << "  GEMM_TOTAL_SIZE=N             Simulated parent matrix size.\n"
         << "  GEMM_WARMUP=N                 Warmup iterations.\n"
         << "  GEMM_ITERS=N                  Timed iterations.\n";
    cerr << "  HYGON_HSS_A_MODE=preconvert|tile\n"
         << "  HYGON_HSS_KERNEL=auto|tuned|legacy|square|wide128|square4\n";
}

BenchmarkOptions parse_options(int argc, char** argv)
{
    BenchmarkOptions options;
    options.verify = parse_bool_env("GEMM_VERIFY", false);
    options.verify_element_limit =
        parse_size_env("GEMM_VERIFY_ELEMENT_LIMIT", kDefaultVerifyElementLimit);
    options.total_matrix_size = parse_int_env("GEMM_TOTAL_SIZE", options.total_matrix_size);
    options.num_warmup = parse_int_env("GEMM_WARMUP", options.num_warmup);
    options.num_iters = parse_int_env("GEMM_ITERS", options.num_iters);

    if (argc >= 2) {
        string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }

        options.k_steps.clear();
        options.k_steps.push_back(atoi(argv[1]));
    }

    return options;
}

void run_right_looking_gemm_benchmark(const BenchmarkOptions& options)
{
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    cout << "======================================================================================================================\n";
    cout << "  GEMM update benchmark for right-looking Cholesky on DCU K100_AI\n";
    cout << "  Operation: C(M x N) = C - A(M x K) * B(N x K)^T\n";
    cout << "  HSS path: launch_hss_gemm(N, T), FP32 inputs, FP32 output, mixed-precision kernel internally\n";
    cout << "  HSS backend: " << hygon_hss_gemm_backend_name()
         << " (true HSS: half A, single B, single C)\n";
    cout << "  Total matrix size: " << options.total_matrix_size
         << ", warmup: " << options.num_warmup
         << ", iterations: " << options.num_iters
         << ", verify: " << (options.verify ? "on" : "off")
         << "\n";
    cout << "======================================================================================================================\n";
    cout << setw(8) << "K"
         << setw(9) << "M"
         << setw(9) << "N"
         << setw(14) << "SGEMM ms"
         << setw(14) << "HSS ms"
         << setw(13) << "speedup"
         << setw(15) << "SGEMM GF/s"
         << setw(15) << "HSS GF/s"
         << setw(14) << "max abs"
         << setw(14) << "rel L2"
         << setw(12) << "verify"
         << "\n";
    cout << "----------------------------------------------------------------------------------------------------------------------\n";

    for (int k : options.k_steps) {
        int m = options.total_matrix_size - k;
        int n = options.total_matrix_size - k;

        if (m <= 0 || n <= 0 || k <= 0) {
            continue;
        }

        size_t size_A = static_cast<size_t>(m) * k;
        size_t size_B = static_cast<size_t>(n) * k;
        size_t size_C = static_cast<size_t>(m) * n;

        DeviceBuffer<float> d_A(size_A);
        DeviceBuffer<float> d_B(size_B);
        bool verify_this_shape = options.verify && size_C <= options.verify_element_limit;
        DeviceBuffer<float> d_C_sgemm(size_C);
        DeviceBuffer<float> d_C_hss(verify_this_shape ? size_C : 0);
        float* d_C_hss_target = verify_this_shape ? d_C_hss.get() : d_C_sgemm.get();

        vector<float> h_A;
        vector<float> h_B;
        vector<float> h_C;
        const vector<float>* h_C_initial = nullptr;

        if (verify_this_shape) {
            h_A.resize(size_A);
            h_B.resize(size_B);
            h_C.resize(size_C);
            fill_matrix(h_A, m, k, 1009u + static_cast<unsigned int>(k));
            fill_matrix(h_B, n, k, 2003u + static_cast<unsigned int>(k));
            fill_matrix(h_C, m, n, 3001u + static_cast<unsigned int>(k));
            copy_to_device(d_A, h_A);
            copy_to_device(d_B, h_B);
            h_C_initial = &h_C;
        } else {
            HIP_CHECK(hipMemsetAsync(d_A, 0, size_A * sizeof(float), stream));
            HIP_CHECK(hipMemsetAsync(d_B, 0, size_B * sizeof(float), stream));
        }

        reset_c(stream, d_C_sgemm, size_C, h_C_initial);
        if (verify_this_shape) {
            reset_c(stream, d_C_hss_target, size_C, h_C_initial);
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        double sgemm_ms = run_rocblas_sgemm(
            handle,
            stream,
            m,
            n,
            k,
            d_A,
            d_B,
            d_C_sgemm,
            size_C,
            h_C_initial,
            options.num_warmup,
            options.num_iters);

        reset_c(stream, d_C_hss_target, size_C, h_C_initial);
        HIP_CHECK(hipStreamSynchronize(stream));

        double hss_ms = run_hss_gemm(
            handle,
            stream,
            m,
            n,
            k,
            d_A,
            d_B,
            d_C_hss_target,
            size_C,
            h_C_initial,
            options.num_warmup,
            options.num_iters);

        ErrorStats error;
        if (options.verify) {
            error = compare_device_matrices(
                stream,
                d_C_sgemm,
                d_C_hss_target,
                size_C,
                options.verify_element_limit);
        }

        double sgemm_gflops = calculate_gemm_gflops(m, n, k, sgemm_ms);
        double hss_gflops = calculate_gemm_gflops(m, n, k, hss_ms);
        double speedup = hss_ms > 0.0 ? sgemm_ms / hss_ms : 0.0;

        cout << setw(8) << k
             << setw(9) << m
             << setw(9) << n
             << setw(14) << fixed << setprecision(3) << sgemm_ms
             << setw(14) << fixed << setprecision(3) << hss_ms
             << setw(13) << fixed << setprecision(3) << speedup
             << setw(15) << fixed << setprecision(1) << sgemm_gflops
             << setw(15) << fixed << setprecision(1) << hss_gflops;

        if (options.verify && !error.skipped) {
            cout << setw(14) << scientific << setprecision(2) << error.max_abs
                 << setw(14) << scientific << setprecision(2) << error.rel_l2
                 << setw(12) << "ok";
        } else if (options.verify) {
            cout << setw(14) << "skipped"
                 << setw(14) << "skipped"
                 << setw(12) << "too large";
        } else {
            cout << setw(14) << "-"
                 << setw(14) << "-"
                 << setw(12) << "off";
        }

        cout << defaultfloat << "\n";
    }

    cout << "======================================================================================================================\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        BenchmarkOptions options = parse_options(argc, argv);
        run_right_looking_gemm_benchmark(options);
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
        return 1;
    }

    return 0;
}
