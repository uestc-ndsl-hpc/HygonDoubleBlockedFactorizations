#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
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

struct GemmShape {
    int m = 0;
    int n = 0;
    int k = 0;
    string tag;
    int launches = 1;
    double workload_flops = 0.0;
};

struct BenchmarkOptions {
    vector<int> k_steps = {
        64, 128, 256, 384, 512, 768, 1024, 1536,
        2048, 3072, 4096, 6144, 8192, 12288, 16384
    };
    vector<GemmShape> shapes;
    string shape_mode = "sweep";
    double workload_total_flops = 0.0;
    int total_matrix_size = kTotalMatrixSize;
    int potrf_n = kTotalMatrixSize;
    int potrf_nb = 256;
    int shape_limit = 0;
    int num_warmup = 2;
    int num_iters = 5;
    bool verify = false;
    bool run_sgemm = true;
    bool run_rocblas_hss = true;
    bool run_custom_hss = true;
    bool list_shapes = false;
    size_t verify_element_limit = kDefaultVerifyElementLimit;
};

struct ErrorStats {
    bool skipped = true;
    double max_abs = 0.0;
    double rel_l2 = 0.0;
};

double calculate_gemm_gflops(int m, int n, int k, double time_ms)
{
    if (time_ms <= 0.0) {
        return 0.0;
    }

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

void add_workload_shape(
    vector<GemmShape>& shapes,
    const string& tag,
    int m,
    int n,
    int k,
    int launches = 1)
{
    if (m <= 0 || n <= 0 || k <= 0 || launches <= 0) {
        return;
    }

    double flops = 2.0 * static_cast<double>(m) * n * k * launches;
    for (GemmShape& shape : shapes) {
        if (shape.tag == tag && shape.m == m && shape.n == n && shape.k == k) {
            shape.launches += launches;
            shape.workload_flops += flops;
            return;
        }
    }

    shapes.push_back({m, n, k, tag, launches, flops});
}

void enumerate_trsm_internal_shapes(
    vector<GemmShape>& shapes,
    int rows,
    int cols,
    int nb)
{
    if (rows <= 0 || cols <= nb) {
        return;
    }

    int m1 = cols / 2;
    int m2 = cols - m1;

    enumerate_trsm_internal_shapes(shapes, rows, m1, nb);
    add_workload_shape(shapes, "trsm", rows, m2, m1);
    enumerate_trsm_internal_shapes(shapes, rows, m2, nb);
}

void enumerate_trsm_shapes(
    vector<GemmShape>& shapes,
    int rows,
    int cols,
    int nb)
{
    constexpr int kTrsmRowLimit = 8192;
    if (rows <= kTrsmRowLimit) {
        enumerate_trsm_internal_shapes(shapes, rows, cols, nb);
        return;
    }

    int chunks = (rows + kTrsmRowLimit - 1) / kTrsmRowLimit;
    for (int i = 0; i < chunks; ++i) {
        int current_rows = min(kTrsmRowLimit, rows - i * kTrsmRowLimit);
        enumerate_trsm_internal_shapes(shapes, current_rows, cols, nb);
    }
}

void enumerate_syrk_shapes(
    vector<GemmShape>& shapes,
    int n,
    int k,
    int nb)
{
    if (n <= nb || k <= 0) {
        return;
    }

    int num_blocks = n / nb;
    for (int b = 0; b < num_blocks; ++b) {
        add_workload_shape(shapes, "syrk", nb, nb, k);
    }

    int tail = n % nb;
    if (tail > 0) {
        add_workload_shape(shapes, "syrk-tail", tail, tail, k);
    }

    for (int i = 1; (n / (2 * i * nb) > 0) || (n % (2 * i * nb) > i * nb); i *= 2) {
        int span = i * nb;
        int batch_count = n / (2 * span);
        for (int b = 0; b < batch_count; ++b) {
            add_workload_shape(shapes, "syrk", span, span, k);
        }

        int rem = n % (2 * span);
        if (rem > span) {
            add_workload_shape(shapes, "syrk-tail", rem - span, span, k);
        }
    }
}

void enumerate_potrf_shapes(
    vector<GemmShape>& shapes,
    int n,
    int nb)
{
    if (n <= nb) {
        return;
    }

    int n1 = n / 2;
    int n2 = n - n1;

    enumerate_potrf_shapes(shapes, n1, nb);
    enumerate_trsm_shapes(shapes, n2, n1, nb);
    enumerate_syrk_shapes(shapes, n2, n1, nb);
    enumerate_potrf_shapes(shapes, n2, nb);
}

bool parse_shape_token(const string& token, GemmShape& shape)
{
    vector<int> values;
    string current;
    for (char ch : token) {
        if (ch == 'x' || ch == 'X' || ch == '*') {
            if (current.empty()) {
                return false;
            }
            values.push_back(atoi(current.c_str()));
            current.clear();
        } else if (ch >= '0' && ch <= '9') {
            current.push_back(ch);
        } else {
            return false;
        }
    }

    if (!current.empty()) {
        values.push_back(atoi(current.c_str()));
    }

    if (values.size() != 3 || values[0] <= 0 || values[1] <= 0 || values[2] <= 0) {
        return false;
    }

    shape = {values[0], values[1], values[2], "manual", 1, 2.0 * values[0] * values[1] * values[2]};
    return true;
}

vector<GemmShape> parse_shape_list_env(const char* name)
{
    vector<GemmShape> shapes;
    vector<string> tokens = split_env_list(getenv(name));
    for (const string& token : tokens) {
        GemmShape shape;
        if (parse_shape_token(token, shape)) {
            shapes.push_back(shape);
        }
    }

    return shapes;
}

void sort_shapes_by_workload_weight(vector<GemmShape>& shapes)
{
    sort(shapes.begin(), shapes.end(), [](const GemmShape& a, const GemmShape& b) {
        if (a.workload_flops != b.workload_flops) {
            return a.workload_flops > b.workload_flops;
        }
        if (a.m != b.m) return a.m > b.m;
        if (a.n != b.n) return a.n > b.n;
        return a.k > b.k;
    });
}

double sum_workload_flops(const vector<GemmShape>& shapes)
{
    double total = 0.0;
    for (const GemmShape& shape : shapes) {
        total += shape.workload_flops;
    }

    return total;
}

vector<GemmShape> make_default_sweep_shapes(const BenchmarkOptions& options)
{
    vector<GemmShape> shapes;
    for (int k : options.k_steps) {
        int m = options.total_matrix_size - k;
        int n = options.total_matrix_size - k;
        if (m > 0 && n > 0 && k > 0) {
            add_workload_shape(shapes, "sweep", m, n, k);
        }
    }

    return shapes;
}

string benchmark_mode_description(const BenchmarkOptions& options)
{
    if (options.shape_mode == "potrf") {
        return "mixed POTRF HSS GEMM workload";
    }

    if (options.shape_mode == "manual") {
        return "explicit GEMM shapes";
    }

    return "right-looking sweep";
}

vector<GemmShape> build_benchmark_shapes(const BenchmarkOptions& options)
{
    if (options.shape_mode == "sweep") {
        return make_default_sweep_shapes(options);
    }

    return options.shapes;
}

void print_shape_plan(const BenchmarkOptions& options)
{
    vector<GemmShape> shapes = build_benchmark_shapes(options);
    double total_workload_flops = options.workload_total_flops > 0.0
        ? options.workload_total_flops
        : sum_workload_flops(shapes);

    cout << "GEMM shape plan\n";
    cout << "  Shape mode: " << benchmark_mode_description(options) << "\n";
    if (options.shape_mode == "potrf") {
        cout << "  POTRF_N=" << options.potrf_n
             << ", POTRF_NB=" << options.potrf_nb
             << ", selected shapes=" << shapes.size()
             << ", shape limit=" << (options.shape_limit > 0 ? options.shape_limit : 0)
             << "\n";
    } else {
        cout << "  Selected shapes: " << shapes.size() << "\n";
    }

    cout << setw(8) << "K"
         << setw(9) << "M"
         << setw(9) << "N"
         << setw(12) << "tag"
         << setw(10) << "launch"
         << setw(11) << "work %"
         << "\n";
    cout << "-----------------------------------------------------------\n";

    for (const GemmShape& shape : shapes) {
        double work_pct = total_workload_flops > 0.0
            ? (100.0 * shape.workload_flops / total_workload_flops)
            : 0.0;
        cout << setw(8) << shape.k
             << setw(9) << shape.m
             << setw(9) << shape.n
             << setw(12) << shape.tag
             << setw(10) << shape.launches
             << setw(11) << fixed << setprecision(2) << work_pct
             << defaultfloat << "\n";
    }
}

void parse_cases_env(BenchmarkOptions& options)
{
    vector<string> tokens = split_env_list(getenv("GEMM_CASES"));
    if (tokens.empty()) {
        return;
    }

    options.run_sgemm = false;
    options.run_rocblas_hss = false;
    options.run_custom_hss = false;

    for (string token : tokens) {
        token = to_lower_ascii(token);
        if (token == "all" || token == "default" || token == "fast") {
            options.run_sgemm = true;
            options.run_rocblas_hss = true;
            options.run_custom_hss = true;
        } else if (token == "sgemm" || token == "s") {
            options.run_sgemm = true;
        } else if (token == "rochss" || token == "roc_hss" ||
                   token == "rocblas_hss" || token == "rocblas") {
            options.run_rocblas_hss = true;
        } else if (token == "hss" || token == "custom" || token == "custom_hss") {
            options.run_custom_hss = true;
        }
    }

    if (!options.run_sgemm && !options.run_rocblas_hss && !options.run_custom_hss) {
        options.run_sgemm = true;
        options.run_rocblas_hss = true;
        options.run_custom_hss = true;
    }
}

string enabled_cases_description(const BenchmarkOptions& options)
{
    vector<string> names;
    if (options.run_sgemm) names.push_back("sgemm");
    if (options.run_rocblas_hss) names.push_back("rochss");
    if (options.run_custom_hss) names.push_back("hss");

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

void convert_inputs_to_half(
    hipStream_t stream,
    int m,
    int n,
    int k,
    const float* d_A,
    const float* d_B,
    rocblas_half* d_A_half,
    rocblas_half* d_B_half)
{
    launch_fp32_to_fp16(
        d_A,
        reinterpret_cast<__half*>(d_A_half),
        m,
        k,
        m,
        m,
        stream);
    launch_fp32_to_fp16(
        d_B,
        reinterpret_cast<__half*>(d_B_half),
        n,
        k,
        n,
        n,
        stream);
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

double run_rocblas_hss_gemm(
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

double run_custom_hss_gemm(
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

    for (size_t i = 0; i < elements; ++i) {
        double r = static_cast<double>(ref[i]);
        double t = static_cast<double>(test[i]);
        double diff = fabs(t - r);

        max_abs = max(max_abs, diff);
        diff_l2 += diff * diff;
        ref_l2 += r * r;
    }

    stats.skipped = false;
    stats.max_abs = max_abs;
    stats.rel_l2 = sqrt(diff_l2) / max(sqrt(ref_l2), numeric_limits<double>::min());
    return stats;
}

void print_usage(const char* program)
{
    cerr << "Usage: " << program << " [k_step]\n"
         << "Environment:\n"
         << "  GEMM_VERIFY=1                 Enable custom HSS vs SGEMM check.\n"
         << "  GEMM_VERIFY_ELEMENT_LIMIT=N   Max C elements copied for verification.\n"
         << "  GEMM_TOTAL_SIZE=N             Simulated parent matrix size.\n"
         << "  GEMM_K_STEPS=K1,K2,...        Comma-separated K sweep list.\n"
         << "  GEMM_SHAPES=MxNxK,...         Explicit GEMM shape list.\n"
         << "  GEMM_WORKLOAD=potrf           Use mixed POTRF TRSM/SYRK GEMM shapes.\n"
         << "  GEMM_SHAPE_LIMIT=N            Run only the first N selected shapes.\n"
         << "  GEMM_LIST_SHAPES=1            Print selected shapes without running GEMM.\n"
         << "  POTRF_N=N                     POTRF workload matrix size.\n"
         << "  POTRF_NB=NB                   POTRF workload leaf/block size.\n"
         << "  GEMM_CASES=sgemm,rochss,hss   Cases: sgemm,rochss,hss,all.\n"
         << "  GEMM_WARMUP=N                 Warmup iterations.\n"
         << "  GEMM_ITERS=N                  Timed iterations.\n"
         << "  HYGON_HSS_KERNEL=auto|tuned|legacy|square|wide128|square4\n";
}

BenchmarkOptions parse_options(int argc, char** argv)
{
    BenchmarkOptions options;
    options.verify = parse_bool_env("GEMM_VERIFY", false);
    options.verify_element_limit =
        parse_size_env("GEMM_VERIFY_ELEMENT_LIMIT", kDefaultVerifyElementLimit);
    options.total_matrix_size = parse_int_env("GEMM_TOTAL_SIZE", options.total_matrix_size);
    options.potrf_n = parse_int_env("POTRF_N", options.potrf_n);
    options.potrf_nb = parse_int_env("POTRF_NB", options.potrf_nb);
    options.shape_limit = parse_int_env("GEMM_SHAPE_LIMIT", 0);
    options.k_steps = parse_int_list_env("GEMM_K_STEPS", options.k_steps);
    options.num_warmup = parse_int_env("GEMM_WARMUP", options.num_warmup);
    options.num_iters = parse_int_env("GEMM_ITERS", options.num_iters);
    options.list_shapes = parse_bool_env("GEMM_LIST_SHAPES", false);
    parse_cases_env(options);

    options.shapes = parse_shape_list_env("GEMM_SHAPES");
    if (!options.shapes.empty()) {
        options.shape_mode = "manual";
        options.workload_total_flops = sum_workload_flops(options.shapes);
    }

    string workload = to_lower_ascii(getenv("GEMM_WORKLOAD") ? getenv("GEMM_WORKLOAD") : "");
    if (options.shapes.empty() &&
        (workload == "potrf" || workload == "cholesky" || workload == "mixed_potrf")) {
        enumerate_potrf_shapes(options.shapes, options.potrf_n, options.potrf_nb);
        sort_shapes_by_workload_weight(options.shapes);
        options.shape_mode = "potrf";
        options.workload_total_flops = sum_workload_flops(options.shapes);
    }
    if (options.shape_limit > 0 &&
        static_cast<size_t>(options.shape_limit) < options.shapes.size()) {
        options.shapes.resize(static_cast<size_t>(options.shape_limit));
    }

    if (argc >= 2) {
        string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }

        options.k_steps.clear();
        options.k_steps.push_back(atoi(argv[1]));
        options.shapes.clear();
        options.shape_mode = "sweep";
        options.workload_total_flops = 0.0;
    }

    return options;
}

void run_right_looking_gemm_benchmark(const BenchmarkOptions& options)
{
    vector<GemmShape> shapes = build_benchmark_shapes(options);
    double total_workload_flops = options.workload_total_flops > 0.0
        ? options.workload_total_flops
        : sum_workload_flops(shapes);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    cout << "======================================================================================================================\n";
    cout << "  GEMM update benchmark for right-looking Cholesky on DCU K100_AI\n";
    cout << "  Operation: C(M x N) = C - A(M x K) * B(N x K)^T\n";
    cout << "  SGEMM: rocblas_sgemm with FP32 inputs/output\n";
    cout << "  rocBLAS HSS: rocblas_gemm_ex with half A/B, float C/D, float compute\n";
    cout << "  Custom HSS: " << hygon_hss_gemm_backend_name()
         << " (half A, half B, single C)\n";
    cout << "  HSS inputs are converted to FP16 once before timing each shape.\n";
    cout << "  Shape mode: " << benchmark_mode_description(options) << "\n";
    if (options.shape_mode == "potrf") {
        cout << "  POTRF workload controls: N=" << options.potrf_n
             << ", NB=" << options.potrf_nb
             << ", shape limit=" << (options.shape_limit > 0 ? options.shape_limit : 0)
             << "\n";
    }
    cout << "  Total matrix size: " << options.total_matrix_size
         << ", warmup: " << options.num_warmup
         << ", iterations: " << options.num_iters
         << ", cases: " << enabled_cases_description(options)
         << ", verify: " << (options.verify ? "on" : "off")
         << "\n";
    cout << "======================================================================================================================\n";

    cout << setw(8) << "K"
         << setw(9) << "M"
         << setw(9) << "N"
         << setw(12) << "tag"
         << setw(10) << "launch";
    if (options.run_sgemm) cout << setw(14) << "SGEMM ms";
    if (options.run_rocblas_hss) cout << setw(14) << "rocHSS ms";
    if (options.run_custom_hss) cout << setw(14) << "Custom ms";
    if (options.run_sgemm && options.run_rocblas_hss) cout << setw(13) << "rocHSS spd";
    if (options.run_sgemm && options.run_custom_hss) cout << setw(13) << "Custom spd";
    if (options.run_sgemm) cout << setw(15) << "SGEMM GF/s";
    if (options.run_rocblas_hss) cout << setw(15) << "rocHSS GF/s";
    if (options.run_custom_hss) cout << setw(15) << "Custom GF/s";
    cout << setw(11) << "work %";
    cout << setw(14) << "max abs"
         << setw(14) << "rel L2"
         << setw(12) << "verify"
         << "\n";
    cout << "----------------------------------------------------------------------------------------------------------------------\n";

    for (const GemmShape& shape : shapes) {
        int m = shape.m;
        int n = shape.n;
        int k = shape.k;
        if (m <= 0 || n <= 0 || k <= 0) {
            continue;
        }

        size_t size_A = static_cast<size_t>(m) * k;
        size_t size_B = static_cast<size_t>(n) * k;
        size_t size_C = static_cast<size_t>(m) * n;
        bool needs_hss_inputs = options.run_rocblas_hss || options.run_custom_hss;
        bool verify_this_shape = options.verify && size_C <= options.verify_element_limit;

        DeviceBuffer<float> d_A(size_A);
        DeviceBuffer<float> d_B(size_B);
        DeviceBuffer<rocblas_half> d_A_half(needs_hss_inputs ? size_A : 0);
        DeviceBuffer<rocblas_half> d_B_half(needs_hss_inputs ? size_B : 0);
        DeviceBuffer<float> d_C_sgemm(size_C);
        DeviceBuffer<float> d_C_roc_hss((verify_this_shape && options.run_rocblas_hss) ? size_C : 0);
        DeviceBuffer<float> d_C_custom_hss((verify_this_shape && options.run_custom_hss) ? size_C : 0);

        float* d_C_roc_hss_target =
            (verify_this_shape && options.run_rocblas_hss) ? d_C_roc_hss.get() : d_C_sgemm.get();
        float* d_C_custom_hss_target =
            (verify_this_shape && options.run_custom_hss) ? d_C_custom_hss.get() : d_C_sgemm.get();

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
            if (options.run_rocblas_hss) reset_c(stream, d_C_roc_hss_target, size_C, h_C_initial);
            if (options.run_custom_hss) reset_c(stream, d_C_custom_hss_target, size_C, h_C_initial);
        }
        if (needs_hss_inputs) {
            convert_inputs_to_half(
                stream,
                m,
                n,
                k,
                d_A,
                d_B,
                d_A_half,
                d_B_half);
        }
        HIP_CHECK(hipStreamSynchronize(stream));

        double sgemm_ms = 0.0;
        double roc_hss_ms = 0.0;
        double custom_hss_ms = 0.0;

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

        if (options.run_rocblas_hss) {
            reset_c(stream, d_C_roc_hss_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            roc_hss_ms = run_rocblas_hss_gemm(
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

        if (options.run_custom_hss) {
            reset_c(stream, d_C_custom_hss_target, size_C, h_C_initial);
            HIP_CHECK(hipStreamSynchronize(stream));

            custom_hss_ms = run_custom_hss_gemm(
                handle,
                stream,
                m,
                n,
                k,
                reinterpret_cast<const __half*>(d_A_half.get()),
                reinterpret_cast<const __half*>(d_B_half.get()),
                d_C_custom_hss_target,
                size_C,
                h_C_initial,
                options.num_warmup,
                options.num_iters);
        }

        ErrorStats error;
        if (options.verify && options.run_sgemm && options.run_custom_hss) {
            error = compare_device_matrices(
                stream,
                d_C_sgemm,
                d_C_custom_hss_target,
                size_C,
                options.verify_element_limit);
        }

        double sgemm_gflops = calculate_gemm_gflops(m, n, k, sgemm_ms);
        double roc_hss_gflops = calculate_gemm_gflops(m, n, k, roc_hss_ms);
        double custom_hss_gflops = calculate_gemm_gflops(m, n, k, custom_hss_ms);
        double roc_hss_speedup = roc_hss_ms > 0.0 ? sgemm_ms / roc_hss_ms : 0.0;
        double custom_hss_speedup = custom_hss_ms > 0.0 ? sgemm_ms / custom_hss_ms : 0.0;
        double work_pct = total_workload_flops > 0.0
            ? (100.0 * shape.workload_flops / total_workload_flops)
            : 0.0;

        cout << setw(8) << k
             << setw(9) << m
             << setw(9) << n
             << setw(12) << shape.tag
             << setw(10) << shape.launches;
        if (options.run_sgemm) cout << setw(14) << fixed << setprecision(3) << sgemm_ms;
        if (options.run_rocblas_hss) cout << setw(14) << fixed << setprecision(3) << roc_hss_ms;
        if (options.run_custom_hss) cout << setw(14) << fixed << setprecision(3) << custom_hss_ms;
        if (options.run_sgemm && options.run_rocblas_hss) {
            cout << setw(13) << fixed << setprecision(3) << roc_hss_speedup;
        }
        if (options.run_sgemm && options.run_custom_hss) {
            cout << setw(13) << fixed << setprecision(3) << custom_hss_speedup;
        }
        if (options.run_sgemm) cout << setw(15) << fixed << setprecision(1) << sgemm_gflops;
        if (options.run_rocblas_hss) cout << setw(15) << fixed << setprecision(1) << roc_hss_gflops;
        if (options.run_custom_hss) cout << setw(15) << fixed << setprecision(1) << custom_hss_gflops;
        cout << setw(11) << fixed << setprecision(2) << work_pct;

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
        if (options.list_shapes) {
            print_shape_plan(options);
            return 0;
        }

        run_right_looking_gemm_benchmark(options);
    } catch (const exception& e) {
        cerr << "Exception caught: " << e.what() << endl;
        return 1;
    }

    return 0;
}
