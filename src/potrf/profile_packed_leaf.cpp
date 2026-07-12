#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "potrf/packed_leaf.h"
#include "trsm/right_lower_trsm.h"
#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

namespace {

struct StageTimes {
    double pack_ms = 0.0;
    double potrf_ms = 0.0;
    double trsm_ms = 0.0;
    double unpack_ms = 0.0;
    double total_ms = 0.0;
};

int read_int_arg(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }

    return fallback;
}

StageTimes measure_leaf(rocblas_handle handle,
                        hipStream_t stream,
                        float* matrix,
                        const float* matrix_template,
                        int lda,
                        int n,
                        int b,
                        float* workspace,
                        int* info,
                        bool packed,
                        int warmup,
                        int iterations)
{
    const size_t matrix_elements = static_cast<size_t>(lda) * b;
    const size_t matrix_bytes = matrix_elements * sizeof(float);
    const int trailing_rows = n - b;

    hipEvent_t start;
    hipEvent_t after_pack;
    hipEvent_t after_potrf;
    hipEvent_t after_trsm;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&after_pack));
    HIP_CHECK(hipEventCreate(&after_potrf));
    HIP_CHECK(hipEventCreate(&after_trsm));
    HIP_CHECK(hipEventCreate(&stop));

    StageTimes total;
    for (int iteration = -warmup; iteration < iterations; ++iteration) {
        HIP_CHECK(hipMemcpyAsync(
            matrix,
            matrix_template,
            matrix_bytes,
            hipMemcpyDeviceToDevice,
            stream));
        HIP_CHECK(hipMemsetAsync(info, 0, sizeof(int), stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        HIP_CHECK(hipEventRecord(start, stream));
        if (packed) {
            pack_potrf_leaf(matrix, lda, workspace, b, b, stream);
        }
        HIP_CHECK(hipEventRecord(after_pack, stream));

        float* factor = packed ? workspace : matrix;
        const int factor_ld = packed ? b : lda;
        ROCBLAS_CHECK(rocsolver_spotrf(
            handle,
            rocblas_fill_lower,
            b,
            factor,
            factor_ld,
            info));
        HIP_CHECK(hipEventRecord(after_potrf, stream));

        trsm_right_lower_transpose(
            handle,
            trailing_rows,
            b,
            factor,
            factor_ld,
            matrix + b,
            lda);
        HIP_CHECK(hipEventRecord(after_trsm, stream));

        if (packed) {
            unpack_potrf_leaf(workspace, b, matrix, lda, b, stream);
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        if (iteration >= 0) {
            float pack_ms = 0.0f;
            float potrf_ms = 0.0f;
            float trsm_ms = 0.0f;
            float unpack_ms = 0.0f;
            float total_ms = 0.0f;
            HIP_CHECK(
                hipEventElapsedTime(&pack_ms, start, after_pack));
            HIP_CHECK(hipEventElapsedTime(
                &potrf_ms, after_pack, after_potrf));
            HIP_CHECK(
                hipEventElapsedTime(&trsm_ms, after_potrf, after_trsm));
            HIP_CHECK(
                hipEventElapsedTime(&unpack_ms, after_trsm, stop));
            HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));

            total.pack_ms += pack_ms;
            total.potrf_ms += potrf_ms;
            total.trsm_ms += trsm_ms;
            total.unpack_ms += unpack_ms;
            total.total_ms += total_ms;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(after_pack));
    HIP_CHECK(hipEventDestroy(after_potrf));
    HIP_CHECK(hipEventDestroy(after_trsm));
    HIP_CHECK(hipEventDestroy(stop));

    total.pack_ms /= iterations;
    total.potrf_ms /= iterations;
    total.trsm_ms /= iterations;
    total.unpack_ms /= iterations;
    total.total_ms /= iterations;
    return total;
}

void print_stage(const std::string& method,
                 const std::string& stage,
                 double milliseconds,
                 double work)
{
    const double throughput = milliseconds > 0.0
        ? work / (milliseconds / 1000.0) / 1e9
        : 0.0;

    std::cout << std::left << std::setw(12) << method
              << " | " << std::setw(10) << stage
              << " | " << std::right << std::setw(10) << std::fixed
              << std::setprecision(4) << milliseconds
              << " | " << std::setw(12) << std::setprecision(2)
              << throughput << "\n";

    return;
}

} // namespace

int main(int argc, char** argv)
{
    const int device = read_int_arg(argc, argv, "--device", 0);
    const int n = read_int_arg(argc, argv, "--n", 40960);
    const int b = read_int_arg(argc, argv, "--b", 512);
    const int warmup = read_int_arg(argc, argv, "--warmup", 5);
    const int iterations = read_int_arg(argc, argv, "--iters", 30);

    if (device < 0 || n <= 0 || b <= 0 || b > n || warmup < 0 ||
        iterations <= 0) {
        std::cerr << "Usage: profile_packed_leaf "
                  << "[--device D] [--n N] [--b B] "
                  << "[--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipSetDevice(device));

    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, device));

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    const size_t matrix_elements = static_cast<size_t>(n) * b;
    const size_t leaf_elements = static_cast<size_t>(b) * b;
    DeviceBuffer<float> matrix_template(matrix_elements);
    DeviceBuffer<float> matrix(matrix_elements);
    DeviceBuffer<float> workspace(leaf_elements);
    DeviceBuffer<int> info(1);

    HIP_CHECK(hipMemset(
        matrix_template.get(), 0, matrix_elements * sizeof(float)));

    std::vector<float> host_leaf(leaf_elements, 0.0f);
    for (int index = 0; index < b; ++index) {
        host_leaf[static_cast<size_t>(index) * b + index] = 2.0f;
    }
    HIP_CHECK(hipMemcpy2D(
        matrix_template.get(),
        static_cast<size_t>(n) * sizeof(float),
        host_leaf.data(),
        static_cast<size_t>(b) * sizeof(float),
        static_cast<size_t>(b) * sizeof(float),
        b,
        hipMemcpyHostToDevice));

    const StageTimes direct = measure_leaf(
        handle,
        stream,
        matrix.get(),
        matrix_template.get(),
        n,
        n,
        b,
        workspace.get(),
        info.get(),
        false,
        warmup,
        iterations);
    const StageTimes packed = measure_leaf(
        handle,
        stream,
        matrix.get(),
        matrix_template.get(),
        n,
        n,
        b,
        workspace.get(),
        info.get(),
        true,
        warmup,
        iterations);

    int host_info = 0;
    HIP_CHECK(hipMemcpy(
        &host_info, info.get(), sizeof(int), hipMemcpyDeviceToHost));
    if (host_info != 0) {
        std::cerr << "POTRF failed with info=" << host_info << "\n";
        ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        return EXIT_FAILURE;
    }

    const double potrf_flops =
        static_cast<double>(b) * b * b / 3.0;
    const double trsm_flops =
        static_cast<double>(n - b) * b * b;

    std::cout << "Device            : " << device << " "
              << properties.name << "\n"
              << "Panel N/B         : " << n << "/" << b << "\n"
              << "Direct factor lda : " << n << "\n"
              << "Packed factor lda : " << b << "\n"
              << "Warmup/iterations : " << warmup << "/" << iterations
              << "\n\n";

    std::cout << std::left << std::setw(12) << "Method"
              << " | " << std::setw(10) << "Stage"
              << " | " << std::right << std::setw(10) << "Time(ms)"
              << " | " << std::setw(12) << "GFLOPS"
              << "\n";
    std::cout << std::string(53, '-') << "\n";
    print_stage("Direct", "POTRF", direct.potrf_ms, potrf_flops);
    print_stage("Direct", "TRSM", direct.trsm_ms, trsm_flops);
    print_stage("Direct", "Total", direct.total_ms, 0.0);
    print_stage("Packed", "Pack", packed.pack_ms, 0.0);
    print_stage("Packed", "POTRF", packed.potrf_ms, potrf_flops);
    print_stage("Packed", "TRSM", packed.trsm_ms, trsm_flops);
    print_stage("Packed", "Unpack", packed.unpack_ms, 0.0);
    print_stage("Packed", "Total", packed.total_ms, 0.0);

    const double saved_ms = direct.total_ms - packed.total_ms;
    const double speedup = packed.total_ms > 0.0
        ? direct.total_ms / packed.total_ms
        : 0.0;
    const int leaf_count = (n + b - 1) / b;
    std::cout << "\nNet speedup       : " << std::fixed << std::setprecision(4)
              << speedup << "x\n"
              << "Saved per leaf    : " << saved_ms << " ms\n"
              << "Projected N total : " << saved_ms * leaf_count
              << " ms over " << leaf_count << " leaves\n";

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
