#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "syrk/recursive_syrk.h"
#include "syrk/batched_syrk.h"

using namespace std;

// Error checking macros for HIP and rocBLAS
#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            cerr << "HIP error at " << __FILE__ << ":" << __LINE__ \
                 << " : " << hipGetErrorString(error) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define ROCBLAS_CHECK(cmd) \
    do { \
        rocblas_status status = (cmd); \
        if (status != rocblas_status_success) { \
            cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__ \
                 << " : " << rocblas_status_to_string(status) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Calculate GFLOPS for SYRK (Lower triangle: n*(n+1)*k operations)
double calculate_gflops(int n, int k, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double flops = static_cast<double>(n) * (n + 1) * k;
    return (flops / (time_ms / 1000.0)) / 1e9;
}

// Host-side matrix initialization
void generate_random_matrix(float* A, int rows, int cols, int lda) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            A[j * lda + i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

void generate_symmetric_matrix(float* C, int n, int ldc) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            C[j * ldc + i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
        }
    }
}

// Relative Frobenius norm error calculation
template<typename T_REF>
double verify_relative_error_lower(const vector<T_REF>& ref, const vector<float>& res, int n) {
    double diff_norm = 0.0;
    double ref_norm  = 0.0;
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            int idx = j * n + i;
            double diff = static_cast<double>(ref[idx]) - static_cast<double>(res[idx]);
            diff_norm += diff * diff;
            ref_norm += static_cast<double>(ref[idx]) * static_cast<double>(ref[idx]);
        }
    }
    return sqrt(diff_norm) / (sqrt(ref_norm) + 1e-20);
}

void run_comparison(int n, int k, int nb) {
    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));

    int lda = n;
    int ldc = n;
    bool do_fp64_verify = (n <= 4096); 

    // Host memory allocation
    vector<float> h_A_f(n * k);
    vector<float> h_C_f(n * n);
    vector<float> h_C_rocblas(n * n);
    vector<float> h_C_rec(n * n);
    vector<float> h_C_batch(n * n);

    generate_random_matrix(h_A_f.data(), n, k, lda);
    generate_symmetric_matrix(h_C_f.data(), n, ldc);

    // Device memory allocation (FP32)
    float *d_A_f, *d_C_f;
    size_t bytes_A = (size_t)n * k * sizeof(float);
    size_t bytes_C = (size_t)n * n * sizeof(float);
    HIP_CHECK(hipMalloc(&d_A_f, bytes_A));
    HIP_CHECK(hipMalloc(&d_C_f, bytes_C));
    HIP_CHECK(hipMemcpy(d_A_f, h_A_f.data(), bytes_A, hipMemcpyHostToDevice));

    // Optional FP64 golden reference generation
    vector<double> h_C_gold;
    if (do_fp64_verify) {
        cout << ">>> Generating FP64 Golden Reference..." << endl;
        double *d_A_d, *d_C_d;
        vector<double> h_A_d(n * k), h_C_d(n * n);
        h_C_gold.resize(n * n);

        for(size_t i=0; i<h_A_f.size(); ++i) h_A_d[i] = h_A_f[i];
        for(size_t i=0; i<h_C_f.size(); ++i) h_C_d[i] = h_C_f[i];

        HIP_CHECK(hipMalloc(&d_A_d, (size_t)n * k * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_C_d, (size_t)n * n * sizeof(double)));
        HIP_CHECK(hipMemcpy(d_A_d, h_A_d.data(), (size_t)n * k * sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_C_d, h_C_d.data(), (size_t)n * n * sizeof(double), hipMemcpyHostToDevice));

        double alpha_d = -1.0, beta_d = 1.0;
        ROCBLAS_CHECK(rocblas_dsyrk(handle, rocblas_fill_lower, rocblas_operation_none, n, k, &alpha_d, d_A_d, lda, &beta_d, d_C_d, ldc));
        
        HIP_CHECK(hipMemcpy(h_C_gold.data(), d_C_d, (size_t)n * n * sizeof(double), hipMemcpyDeviceToHost));
        
        HIP_CHECK(hipFree(d_A_d));
        HIP_CHECK(hipFree(d_C_d));
    }

    // Benchmark timing utility
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    auto benchmark = [&](auto func, const string& label, vector<float>& output) {
        HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
        
        func(); // Warmup
        
        HIP_CHECK(hipMemcpy(d_C_f, h_C_f.data(), bytes_C, hipMemcpyHostToDevice));
        HIP_CHECK(hipEventRecord(start));
        func();
        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        HIP_CHECK(hipMemcpy(output.data(), d_C_f, bytes_C, hipMemcpyDeviceToHost));

        double gflops = calculate_gflops(n, k, ms);
        cout << left << setw(20) << label << " | " << setw(10) << fixed << setprecision(2) << ms << " ms | " << gflops << " GFLOPS" << endl;
        return static_cast<double>(ms);
    };

    cout << "\n" << string(52, '=') << "\n";
    cout << "SYRK Benchmark (N=" << n << ", K=" << k << ", NB=" << nb << ")\n";
    cout << string(52, '=') << "\n";

    float alpha_f = -1.0f, beta_f = 1.0f;
    
    double t_roc = benchmark([&](){ 
        ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_lower, rocblas_operation_none, n, k, &alpha_f, d_A_f, lda, &beta_f, d_C_f, ldc)); 
    }, "rocBLAS SYRK", h_C_rocblas);

    double t_rec = benchmark([&](){ 
        recursive_syrk_parallel(handle, d_A_f, lda, d_C_f, ldc, n, k, nb); 
    }, "Recursive SYRK", h_C_rec);

    double t_bat = benchmark([&](){ 
        batched_syrk_bottom_up(handle, d_A_f, lda, d_C_f, ldc, n, k, nb); 
    }, "Batched SYRK", h_C_batch);

    // Error and speedup analysis
    cout << string(52, '-') << "\n";
    if (do_fp64_verify) {
        cout << "Relative Error (vs FP64 Golden):\n";
        cout << "  Recursive: " << scientific << verify_relative_error_lower(h_C_gold, h_C_rec, n) << endl;
        cout << "  Batched:   " << scientific << verify_relative_error_lower(h_C_gold, h_C_batch, n) << endl;
    } else {
        cout << "Relative Error (vs FP32 rocBLAS Reference):\n";
        cout << "  Recursive: " << scientific << verify_relative_error_lower(h_C_rocblas, h_C_rec, n) << endl;
        cout << "  Batched:   " << scientific << verify_relative_error_lower(h_C_rocblas, h_C_batch, n) << endl;
    }

    cout << fixed << setprecision(2);
    cout << string(52, '-') << "\n";
    cout << "Recursive Speedup : " << t_roc / t_rec << "x" << endl;
    cout << "Batched Speedup   : " << t_roc / t_bat << "x" << endl;
    cout << string(52, '=') << "\n";

    // Resource cleanup
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A_f));
    HIP_CHECK(hipFree(d_C_f));
    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
}

int main(int argc, char** argv) {
    int n = 4096, k = 4096, nb = 256;
    if (argc >= 2) n = atoi(argv[1]);
    if (argc >= 3) k = atoi(argv[2]);
    if (argc >= 4) nb = atoi(argv[3]);

    srand(2026);
    run_comparison(n, k, nb);
    return 0;
}