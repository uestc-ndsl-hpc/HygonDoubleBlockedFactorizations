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
#include "kernels/fp16_convert.h"
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
    bool run_sgemm = true;
    bool run_half_a = false;
    bool run_rocblas_hss = false;
    bool run_core_hss = false;
    bool run_hss = true;
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

string to_lower_ascii(string text)
{
    for (char& ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }

    return text;
}

vector<string> split_env_list(const char* value)
{
    vector<string> tokens;
    if (value == nullptr) {
        return tokens;
    }

    string current;
    for (const char* p = value; *p != '\0'; ++p) {
        char ch = *p;
        if (ch == ',' || ch == ';' || ch == ':' || ch == ' ' || ch == '\t') {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }

    if (!current.empty()) {
        tokens.push_back(current);
    }

    return tokens;
}

vector<int> parse_int_list_env(const char* name, const vector<int>& default_value)
{
    vector<string> tokens = split_env_list(getenv(name));
    if (tokens.empty()) {
        return default_value;
    }

    vector<int> parsed_values;
    for (const string& token : tokens) {
        char* end = nullptr;
        long parsed = strtol(token.c_str(), &end, 10);
        if (end != token.c_str() && parsed > 0) {
            parsed_values.push_back(static_cast<int>(parsed));
        }
    }

    return parsed_values.empty() ? default_value : parsed_values;
}

void parse_cases_env(BenchmarkOptions& options)
{
    vector<string> tokens = split_env_list(getenv("GEMM_CASES"));
    if (tokens.empty()) {
        return;
    }

    options.run_sgemm = false;
    options.run_half_a = false;
    options.run_rocblas_hss = false;
    options.run_core_hss = false;
    options.run_hss = false;

    for (string token : tokens) {
        token = to_lower_ascii(token);
        if (token == "all" || token == "diagnostic" || token == "diagnostics") {
            options.run_sgemm = true;
            options.run_half_a = true;
            options.run_rocblas_hss = true;
            options.run_core_hss = true;
            options.run_hss = true;
        } else if (token == "default" || token == "fast") {
            options.run_sgemm = true;
            options.run_hss = true;
        } else if (token == "sgemm" || token == "s") {
            options.run_sgemm = true;
        } else if (token == "halfa" || token == "half_a" ||
                   token == "half-a" || token == "legacy") {
            options.run_half_a = true;
        } else if (token == "rochss" || token == "roc_hss" ||
                   token == "rocblas_hss" || token == "rocblas") {
            options.run_rocblas_hss = true;
        } else if (token == "core" || token == "core_hss" || token == "hss_core") {
            options.run_core_hss = true;
        } else if (token == "hss" || token == "custom" || token == "total") {
            options.run_hss = true;
        }
    }

    if (!options.run_sgemm && !options.run_half_a && !options.run_rocblas_hss &&
        !options.run_core_hss && !options.run_hss) {
        options.run_sgemm = true;
        options.run_hss = true;
    }
}

string enabled_cases_description(const BenchmarkOptions& options)
{
    vector<string> names;
    if (options.run_sgemm) names.push_back("sgemm");
    if (options.run_half_a) names.push_back("halfa");
    if (options.run_rocblas_hss) names.push_back("rochss");
    if (options.run_core_hss) names.push_back("core");
    if (options.run_hss) names.push_back("hss");

    string text;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            text += ",";
        }
        text += names[i];
    }

    return text;
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

double run_hss_gemm_core_preconverted(
    rocblas_handle handle,
    hipStream_t stream,
    int m,
    int n,
    int k,
    const __half* d_A_half,
    const __half* d_B_half,
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
        launch_hss_gemm_preconverted(
            handle,
            stream,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            k,
            alpha,
            d_A_half,
            lda,
            d_B_half,
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
            launch_hss_gemm_preconverted(
                handle,
                stream,
                rocblas_operation_none,
                rocblas_operation_transpose,
                m,
                n,
                k,
                alpha,
                d_A_half,
                lda,
                d_B_half,
                ldb,
                beta,
                d_C,
                ldc);
        }
    });

    HIP_CHECK(hipGetLastError());
    return static_cast<double>(total_time_ms) / num_iters;
}

double run_rocblas_hss_gemm_preconverted(
    rocblas_handle handle,
    hipStream_t stream,
    int m,
    int n,
    int k,
    const rocblas_half* d_A_half,
    const rocblas_half* d_B_half,
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

    auto call_rocblas_hss = [&]() {
        ROCBLAS_CHECK(rocblas_gemm_ex(
            handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            m,
            n,
            k,
            &alpha,
            d_A_half,
            rocblas_datatype_f16_r,
            lda,
            d_B_half,
            rocblas_datatype_f16_r,
            ldb,
            &beta,
            d_C,
            rocblas_datatype_f32_r,
            ldc,
            d_C,
            rocblas_datatype_f32_r,
            ldc,
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_standard,
            0,
            0));
    };

    for (int w = 0; w < num_warmup; ++w) {
        call_rocblas_hss();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    reset_c(stream, d_C, size_C, h_C_initial);
    HIP_CHECK(hipStreamSynchronize(stream));

    GPUTimer timer;
    float total_time_ms = timer.measure(stream, [&]() {
        for (int i = 0; i < num_iters; ++i) {
            call_rocblas_hss();
        }
    });

    return static_cast<double>(total_time_ms) / num_iters;
}

double run_half_a_single_b_gemm(
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
        launch_half_a_single_b_gemm(
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
            launch_half_a_single_b_gemm(
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
         << "  GEMM_K_STEPS=K1,K2,...        Comma-separated K sweep list.\n"
         << "  GEMM_CASES=sgemm,hss          Cases: sgemm,halfa,rochss,core,hss,all.\n"
         << "  GEMM_WARMUP=N                 Warmup iterations.\n"
         << "  GEMM_ITERS=N                  Timed iterations.\n";
    cerr << "  HYGON_HSS_A_MODE=preconvert|tile\n"
         << "  HYGON_HSS_INPUT_MODE=true|half_a_single_b\n"
         << "  HYGON_HSS_KERNEL=auto|tuned|legacy|square|wide128|square4\n";
}

BenchmarkOptions parse_options(int argc, char** argv)
{
    BenchmarkOptions options;
    options.verify = parse_bool_env("GEMM_VERIFY", false);
    options.verify_element_limit =
        parse_size_env("GEMM_VERIFY_ELEMENT_LIMIT", kDefaultVerifyElementLimit);
    options.total_matrix_size = parse_int_env("GEMM_TOTAL_SIZE", options.total_matrix_size);
    options.k_steps = parse_int_list_env("GEMM_K_STEPS", options.k_steps);
    options.num_warmup = parse_int_env("GEMM_WARMUP", options.num_warmup);
    options.num_iters = parse_int_env("GEMM_ITERS", options.num_iters);
    parse_cases_env(options);

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
    cout << "  HSS path: launch_hss_gemm(N, T), FP32 inputs, FP32 output, A/B converted to FP16 internally\n";
    cout << "  HSS backend: " << hygon_hss_gemm_backend_name()
         << " (true HSS: half A, half B, single C)\n";
    cout << "  HSS core-only: A/B converted once, then custom true-HSS kernel only\n";
    cout << "  rocBLAS HSS: rocblas_gemm_ex with half A/B, float C/D, float compute\n";
    cout << "  Half-A baseline: " << hygon_half_a_single_b_backend_name()
         << " (legacy comparison: half A, single B, single C)\n";
    cout << "  Total matrix size: " << options.total_matrix_size
         << ", warmup: " << options.num_warmup
         << ", iterations: " << options.num_iters
         << ", cases: " << enabled_cases_description(options)
         << ", verify: " << (options.verify ? "on" : "off")
         << "\n";
    cout << "======================================================================================================================\n";
    cout << setw(8) << "K"
         << setw(9) << "M"
         << setw(9) << "N";
    if (options.run_sgemm) cout << setw(14) << "SGEMM ms";
    if (options.run_half_a) cout << setw(14) << "HalfA ms";
    if (options.run_rocblas_hss) cout << setw(14) << "rocHSS ms";
    if (options.run_core_hss) cout << setw(14) << "Core ms";
    if (options.run_hss) cout << setw(14) << "HSS ms";
    if (options.run_sgemm && options.run_half_a) cout << setw(13) << "HalfA spd";
    if (options.run_sgemm && options.run_rocblas_hss) cout << setw(13) << "rocHSS spd";
    if (options.run_sgemm && options.run_core_hss) cout << setw(13) << "Core spd";
    if (options.run_sgemm && options.run_hss) cout << setw(13) << "HSS spd";
    if (options.run_sgemm) cout << setw(15) << "SGEMM GF/s";
    if (options.run_half_a) cout << setw(15) << "HalfA GF/s";
    if (options.run_rocblas_hss) cout << setw(15) << "rocHSS GF/s";
    if (options.run_core_hss) cout << setw(15) << "Core GF/s";
    if (options.run_hss) cout << setw(15) << "HSS GF/s";
    cout << setw(14) << "max abs"
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
        bool needs_preconverted_hss = options.run_rocblas_hss || options.run_core_hss;

        DeviceBuffer<float> d_A(size_A);
        DeviceBuffer<float> d_B(size_B);
        DeviceBuffer<rocblas_half> d_A_half(needs_preconverted_hss ? size_A : 0);
        DeviceBuffer<rocblas_half> d_B_half(needs_preconverted_hss ? size_B : 0);
        bool verify_this_shape = options.verify && size_C <= options.verify_element_limit;
        DeviceBuffer<float> d_C_sgemm(size_C);
        DeviceBuffer<float> d_C_half_a((verify_this_shape && options.run_half_a) ? size_C : 0);
        DeviceBuffer<float> d_C_roc_hss((verify_this_shape && options.run_rocblas_hss) ? size_C : 0);
        DeviceBuffer<float> d_C_core_hss((verify_this_shape && options.run_core_hss) ? size_C : 0);
        DeviceBuffer<float> d_C_hss((verify_this_shape && options.run_hss) ? size_C : 0);
        float* d_C_half_a_target =
            (verify_this_shape && options.run_half_a) ? d_C_half_a.get() : d_C_sgemm.get();
        float* d_C_roc_hss_target =
            (verify_this_shape && options.run_rocblas_hss) ? d_C_roc_hss.get() : d_C_sgemm.get();
        float* d_C_core_hss_target =
            (verify_this_shape && options.run_core_hss) ? d_C_core_hss.get() : d_C_sgemm.get();
        float* d_C_hss_target =
            (verify_this_shape && options.run_hss) ? d_C_hss.get() : d_C_sgemm.get();

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
            if (options.run_half_a) reset_c(stream, d_C_half_a_target, size_C, h_C_initial);
            if (options.run_rocblas_hss) reset_c(stream, d_C_roc_hss_target, size_C, h_C_initial);
            if (options.run_core_hss) reset_c(stream, d_C_core_hss_target, size_C, h_C_initial);
            if (options.run_hss) reset_c(stream, d_C_hss_target, size_C, h_C_initial);
        }
        if (needs_preconverted_hss) {
            launch_fp32_to_fp16(
                d_A,
                reinterpret_cast<__half*>(d_A_half.get()),
                m,
                k,
                m,
                m,
                stream);
            launch_fp32_to_fp16(
                d_B,
                reinterpret_cast<__half*>(d_B_half.get()),
                n,
                k,
                n,
                n,
                stream);
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        double sgemm_ms = 0.0;
        double half_a_ms = 0.0;
        double roc_hss_ms = 0.0;
        double core_hss_ms = 0.0;
        double hss_ms = 0.0;

        if (options.run_sgemm) {
            sgemm_ms = run_rocblas_sgemm(
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
        }

        if (options.run_half_a) {
            reset_c(stream, d_C_half_a_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            half_a_ms = run_half_a_single_b_gemm(
                handle,
                stream,
                m,
                n,
                k,
                d_A,
                d_B,
                d_C_half_a_target,
                size_C,
                h_C_initial,
                options.num_warmup,
                options.num_iters);
        }

        if (options.run_rocblas_hss) {
            reset_c(stream, d_C_roc_hss_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            roc_hss_ms = run_rocblas_hss_gemm_preconverted(
                handle,
                stream,
                m,
                n,
                k,
                d_A_half,
                d_B_half,
                d_C_roc_hss_target,
                size_C,
                h_C_initial,
                options.num_warmup,
                options.num_iters);
        }

        if (options.run_core_hss) {
            reset_c(stream, d_C_core_hss_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            core_hss_ms = run_hss_gemm_core_preconverted(
                handle,
                stream,
                m,
                n,
                k,
                reinterpret_cast<const __half*>(d_A_half.get()),
                reinterpret_cast<const __half*>(d_B_half.get()),
                d_C_core_hss_target,
                size_C,
                h_C_initial,
                options.num_warmup,
                options.num_iters);
        }

        if (options.run_hss) {
            reset_c(stream, d_C_hss_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            hss_ms = run_hss_gemm(
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
        }

        ErrorStats error;
        if (options.verify && options.run_sgemm && options.run_hss) {
            error = compare_device_matrices(
                stream,
                d_C_sgemm,
                d_C_hss_target,
                size_C,
                options.verify_element_limit);
        }

        double sgemm_gflops = calculate_gemm_gflops(m, n, k, sgemm_ms);
        double half_a_gflops = calculate_gemm_gflops(m, n, k, half_a_ms);
        double roc_hss_gflops = calculate_gemm_gflops(m, n, k, roc_hss_ms);
        double core_hss_gflops = calculate_gemm_gflops(m, n, k, core_hss_ms);
        double hss_gflops = calculate_gemm_gflops(m, n, k, hss_ms);
        double half_a_speedup = half_a_ms > 0.0 ? sgemm_ms / half_a_ms : 0.0;
        double roc_hss_speedup = roc_hss_ms > 0.0 ? sgemm_ms / roc_hss_ms : 0.0;
        double core_hss_speedup = core_hss_ms > 0.0 ? sgemm_ms / core_hss_ms : 0.0;
        double hss_speedup = hss_ms > 0.0 ? sgemm_ms / hss_ms : 0.0;

        cout << setw(8) << k
             << setw(9) << m
             << setw(9) << n;
        if (options.run_sgemm) cout << setw(14) << fixed << setprecision(3) << sgemm_ms;
        if (options.run_half_a) cout << setw(14) << fixed << setprecision(3) << half_a_ms;
        if (options.run_rocblas_hss) cout << setw(14) << fixed << setprecision(3) << roc_hss_ms;
        if (options.run_core_hss) cout << setw(14) << fixed << setprecision(3) << core_hss_ms;
        if (options.run_hss) cout << setw(14) << fixed << setprecision(3) << hss_ms;
        if (options.run_sgemm && options.run_half_a) {
            cout << setw(13) << fixed << setprecision(3) << half_a_speedup;
        }
        if (options.run_sgemm && options.run_rocblas_hss) {
            cout << setw(13) << fixed << setprecision(3) << roc_hss_speedup;
        }
        if (options.run_sgemm && options.run_core_hss) {
            cout << setw(13) << fixed << setprecision(3) << core_hss_speedup;
        }
        if (options.run_sgemm && options.run_hss) {
            cout << setw(13) << fixed << setprecision(3) << hss_speedup;
        }
        if (options.run_sgemm) cout << setw(15) << fixed << setprecision(1) << sgemm_gflops;
        if (options.run_half_a) cout << setw(15) << fixed << setprecision(1) << half_a_gflops;
        if (options.run_rocblas_hss) cout << setw(15) << fixed << setprecision(1) << roc_hss_gflops;
        if (options.run_core_hss) cout << setw(15) << fixed << setprecision(1) << core_hss_gflops;
        if (options.run_hss) cout << setw(15) << fixed << setprecision(1) << hss_gflops;

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
