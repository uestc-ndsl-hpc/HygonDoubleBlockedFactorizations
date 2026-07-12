#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include "utils/common/error_check.h"
#include "utils/memory/device_buffer.h"

namespace {

struct Options {
    int device = 0;
    int block_size = 512;
    int far_tile_rows = 8192;
    int max_repeats = 5;
    int warmup = 5;
    int iterations = 30;
};

struct ConcurrentTiming {
    double wall_ms = 0.0;
    double critical_done_ms = 0.0;
    double far_done_ms = 0.0;
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

Options read_options(int argc, char** argv)
{
    Options options;
    options.device = read_int_arg(argc, argv, "--device", options.device);
    options.block_size =
        read_int_arg(argc, argv, "--b", options.block_size);
    options.far_tile_rows =
        read_int_arg(argc, argv, "--far-m", options.far_tile_rows);
    options.max_repeats =
        read_int_arg(argc, argv, "--max-repeats", options.max_repeats);
    options.warmup =
        read_int_arg(argc, argv, "--warmup", options.warmup);
    options.iterations =
        read_int_arg(argc, argv, "--iters", options.iterations);

    return options;
}

template <typename Prepare, typename Enqueue>
double measure_single(hipStream_t stream,
                      int warmup,
                      int iterations,
                      Prepare&& prepare,
                      Enqueue&& enqueue)
{
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    double total_ms = 0.0;
    for (int iteration = -warmup; iteration < iterations; ++iteration) {
        prepare();
        HIP_CHECK(hipStreamSynchronize(stream));

        HIP_CHECK(hipEventRecord(start, stream));
        enqueue();
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        if (iteration >= 0) {
            float elapsed_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
            total_ms += elapsed_ms;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / iterations;
}

template <typename PrepareCritical,
          typename EnqueueCritical,
          typename PrepareFar,
          typename EnqueueFar>
ConcurrentTiming measure_concurrent(hipStream_t control_stream,
                                    hipStream_t critical_stream,
                                    hipStream_t far_stream,
                                    int warmup,
                                    int iterations,
                                    PrepareCritical&& prepare_critical,
                                    EnqueueCritical&& enqueue_critical,
                                    PrepareFar&& prepare_far,
                                    EnqueueFar&& enqueue_far)
{
    hipEvent_t start;
    hipEvent_t gate;
    hipEvent_t critical_done;
    hipEvent_t far_done;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&gate));
    HIP_CHECK(hipEventCreate(&critical_done));
    HIP_CHECK(hipEventCreate(&far_done));
    HIP_CHECK(hipEventCreate(&stop));

    ConcurrentTiming total;
    for (int iteration = -warmup; iteration < iterations; ++iteration) {
        prepare_critical();
        prepare_far();
        HIP_CHECK(hipStreamSynchronize(critical_stream));
        HIP_CHECK(hipStreamSynchronize(far_stream));

        HIP_CHECK(hipEventRecord(start, control_stream));
        HIP_CHECK(hipEventRecord(gate, control_stream));
        HIP_CHECK(hipStreamWaitEvent(critical_stream, gate, 0));
        HIP_CHECK(hipStreamWaitEvent(far_stream, gate, 0));

        enqueue_far();
        HIP_CHECK(hipEventRecord(far_done, far_stream));
        enqueue_critical();
        HIP_CHECK(hipEventRecord(critical_done, critical_stream));

        HIP_CHECK(hipStreamWaitEvent(control_stream, critical_done, 0));
        HIP_CHECK(hipStreamWaitEvent(control_stream, far_done, 0));
        HIP_CHECK(hipEventRecord(stop, control_stream));
        HIP_CHECK(hipEventSynchronize(stop));

        if (iteration >= 0) {
            float wall_ms = 0.0f;
            float critical_ms = 0.0f;
            float far_ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&wall_ms, start, stop));
            HIP_CHECK(
                hipEventElapsedTime(&critical_ms, start, critical_done));
            HIP_CHECK(hipEventElapsedTime(&far_ms, start, far_done));
            total.wall_ms += wall_ms;
            total.critical_done_ms += critical_ms;
            total.far_done_ms += far_ms;
        }
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(gate));
    HIP_CHECK(hipEventDestroy(critical_done));
    HIP_CHECK(hipEventDestroy(far_done));
    HIP_CHECK(hipEventDestroy(stop));

    total.wall_ms /= iterations;
    total.critical_done_ms /= iterations;
    total.far_done_ms /= iterations;
    return total;
}

void print_result(const std::string& name,
                  double critical_ms,
                  double far_ms,
                  const ConcurrentTiming& concurrent)
{
    const double sequential_ms = critical_ms + far_ms;
    const double hidden_ms = sequential_ms - concurrent.wall_ms;
    const double overlap_limit = std::min(critical_ms, far_ms);
    const double overlap_percent = overlap_limit > 0.0
        ? 100.0 * hidden_ms / overlap_limit
        : 0.0;
    const double critical_slowdown = critical_ms > 0.0
        ? concurrent.critical_done_ms / critical_ms
        : 0.0;
    const double far_slowdown = far_ms > 0.0
        ? concurrent.far_done_ms / far_ms
        : 0.0;

    std::cout << std::left << std::setw(24) << name
              << " | " << std::right << std::setw(9) << std::fixed
              << std::setprecision(3) << critical_ms
              << " | " << std::setw(9) << far_ms
              << " | " << std::setw(9) << sequential_ms
              << " | " << std::setw(9) << concurrent.wall_ms
              << " | " << std::setw(9) << hidden_ms
              << " | " << std::setw(8) << std::setprecision(1)
              << overlap_percent << "%"
              << " | " << std::setw(9) << std::setprecision(3)
              << critical_slowdown
              << " | " << std::setw(9) << far_slowdown
              << "\n";

    return;
}

} // namespace

int main(int argc, char** argv)
{
    const Options options = read_options(argc, argv);
    if (options.device < 0 || options.block_size <= 0 ||
        options.far_tile_rows <= 0 || options.max_repeats <= 0 ||
        options.warmup < 0 || options.iterations <= 0) {
        std::cerr << "Usage: profile_wavefront_overlap "
                  << "[--device D] [--b B] [--far-m M] "
                  << "[--max-repeats R] [--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    const long long max_rows_long =
        static_cast<long long>(options.far_tile_rows) *
        options.max_repeats;
    if (max_rows_long > std::numeric_limits<int>::max()) {
        std::cerr << "far-m * max-repeats exceeds the rocBLAS integer range\n";
        return EXIT_FAILURE;
    }

    HIP_CHECK(hipSetDevice(options.device));

    hipDeviceProp_t properties;
    HIP_CHECK(hipGetDeviceProperties(&properties, options.device));

    int least_priority = 0;
    int greatest_priority = 0;
    HIP_CHECK(
        hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

    hipStream_t control_stream;
    hipStream_t critical_stream;
    hipStream_t far_stream;
    HIP_CHECK(hipStreamCreateWithFlags(
        &control_stream, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithPriority(
        &critical_stream, hipStreamNonBlocking, greatest_priority));
    HIP_CHECK(hipStreamCreateWithPriority(
        &far_stream, hipStreamNonBlocking, least_priority));

    rocblas_handle critical_handle;
    rocblas_handle far_handle;
    ROCBLAS_CHECK(rocblas_create_handle(&critical_handle));
    ROCBLAS_CHECK(rocblas_create_handle(&far_handle));
    ROCBLAS_CHECK(
        rocblas_set_pointer_mode(critical_handle, rocblas_pointer_mode_host));
    ROCBLAS_CHECK(
        rocblas_set_pointer_mode(far_handle, rocblas_pointer_mode_host));
    ROCBLAS_CHECK(rocblas_set_stream(critical_handle, critical_stream));
    ROCBLAS_CHECK(rocblas_set_stream(far_handle, far_stream));

    const int b = options.block_size;
    const int max_rows = static_cast<int>(max_rows_long);
    const size_t diagonal_elements = static_cast<size_t>(b) * b;
    const size_t panel_elements = static_cast<size_t>(max_rows) * b;

    DeviceBuffer<float> diagonal_template(diagonal_elements);
    DeviceBuffer<float> diagonal_work(diagonal_elements);
    DeviceBuffer<float> triangular(diagonal_elements);
    DeviceBuffer<float> panel_template(panel_elements);
    DeviceBuffer<float> panel_work(panel_elements);
    DeviceBuffer<float> gemm_output(panel_elements);
    DeviceBuffer<int> info(1);

    std::vector<float> host_diagonal(diagonal_elements, 0.0f);
    for (int index = 0; index < b; ++index) {
        host_diagonal[static_cast<size_t>(index) * b + index] = 2.0f;
    }

    HIP_CHECK(hipMemcpy(
        diagonal_template.get(),
        host_diagonal.data(),
        diagonal_elements * sizeof(float),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        triangular.get(),
        host_diagonal.data(),
        diagonal_elements * sizeof(float),
        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(
        panel_template.get(), 0, panel_elements * sizeof(float)));

    const float one = 1.0f;
    const float zero = 0.0f;

    auto prepare_critical = [&]() {
        HIP_CHECK(hipMemcpyAsync(
            diagonal_work.get(),
            diagonal_template.get(),
            diagonal_elements * sizeof(float),
            hipMemcpyDeviceToDevice,
            critical_stream));
        HIP_CHECK(hipMemsetAsync(
            info.get(), 0, sizeof(int), critical_stream));
    };

    auto enqueue_critical = [&]() {
        ROCBLAS_CHECK(rocsolver_spotrf(
            critical_handle,
            rocblas_fill_lower,
            b,
            diagonal_work.get(),
            b,
            info.get()));
    };

    const double critical_ms = measure_single(
        critical_stream,
        options.warmup,
        options.iterations,
        prepare_critical,
        enqueue_critical);

    std::cout << "Device            : " << options.device << " "
              << properties.name << "\n"
              << "Leaf block        : " << b << "\n"
              << "Far GEMM tile     : " << options.far_tile_rows
              << " x " << b << " x " << b << "\n"
              << "Maximum repeats   : " << options.max_repeats << "\n"
              << "Warmup/iterations : " << options.warmup << "/"
              << options.iterations << "\n"
              << "Stream priorities : critical=" << greatest_priority
              << ", far=" << least_priority << "\n"
              << "Enqueue order     : far, then critical\n"
              << "Critical POTRF    : " << std::fixed << std::setprecision(3)
              << critical_ms << " ms\n\n";

    std::cout << std::left << std::setw(24) << "Case"
              << " | " << std::right << std::setw(9) << "Crit(ms)"
              << " | " << std::setw(9) << "Far(ms)"
              << " | " << std::setw(9) << "Seq(ms)"
              << " | " << std::setw(9) << "Conc(ms)"
              << " | " << std::setw(9) << "Hidden"
              << " | " << std::setw(9) << "Overlap"
              << " | " << std::setw(9) << "CritSlow"
              << " | " << std::setw(9) << "FarSlow"
              << "\n";
    std::cout << std::string(133, '-') << "\n";

    for (int repeats = 1; repeats <= options.max_repeats; ++repeats) {
        auto no_prepare = []() { return; };
        auto enqueue_gemm = [&]() {
            for (int tile = 0; tile < repeats; ++tile) {
                const int row_offset = tile * options.far_tile_rows;
                ROCBLAS_CHECK(rocblas_sgemm(
                    far_handle,
                    rocblas_operation_none,
                    rocblas_operation_transpose,
                    options.far_tile_rows,
                    b,
                    b,
                    &one,
                    panel_template.get() + row_offset,
                    max_rows,
                    triangular.get(),
                    b,
                    &zero,
                    gemm_output.get() + row_offset,
                    max_rows));
            }
        };

        const double gemm_ms = measure_single(
            far_stream,
            options.warmup,
            options.iterations,
            no_prepare,
            enqueue_gemm);
        const ConcurrentTiming gemm_concurrent = measure_concurrent(
            control_stream,
            critical_stream,
            far_stream,
            options.warmup,
            options.iterations,
            prepare_critical,
            enqueue_critical,
            no_prepare,
            enqueue_gemm);
        print_result(
            "GEMM tile x" + std::to_string(repeats),
            critical_ms,
            gemm_ms,
            gemm_concurrent);

        const int trsm_rows = options.far_tile_rows * repeats;
        auto prepare_trsm = [&]() {
            HIP_CHECK(hipMemcpy2DAsync(
                panel_work.get(),
                static_cast<size_t>(max_rows) * sizeof(float),
                panel_template.get(),
                static_cast<size_t>(max_rows) * sizeof(float),
                static_cast<size_t>(trsm_rows) * sizeof(float),
                b,
                hipMemcpyDeviceToDevice,
                far_stream));
        };
        auto enqueue_trsm = [&]() {
            ROCBLAS_CHECK(rocblas_strsm(
                far_handle,
                rocblas_side_right,
                rocblas_fill_lower,
                rocblas_operation_transpose,
                rocblas_diagonal_non_unit,
                trsm_rows,
                b,
                &one,
                triangular.get(),
                b,
                panel_work.get(),
                max_rows));
        };

        const double trsm_ms = measure_single(
            far_stream,
            options.warmup,
            options.iterations,
            prepare_trsm,
            enqueue_trsm);
        const ConcurrentTiming trsm_concurrent = measure_concurrent(
            control_stream,
            critical_stream,
            far_stream,
            options.warmup,
            options.iterations,
            prepare_critical,
            enqueue_critical,
            prepare_trsm,
            enqueue_trsm);
        print_result(
            "TRSM M=" + std::to_string(trsm_rows),
            critical_ms,
            trsm_ms,
            trsm_concurrent);

        auto enqueue_tiled_trsm = [&]() {
            for (int tile = 0; tile < repeats; ++tile) {
                const int row_offset = tile * options.far_tile_rows;
                ROCBLAS_CHECK(rocblas_strsm(
                    far_handle,
                    rocblas_side_right,
                    rocblas_fill_lower,
                    rocblas_operation_transpose,
                    rocblas_diagonal_non_unit,
                    options.far_tile_rows,
                    b,
                    &one,
                    triangular.get(),
                    b,
                    panel_work.get() + row_offset,
                    max_rows));
            }
        };

        const double tiled_trsm_ms = measure_single(
            far_stream,
            options.warmup,
            options.iterations,
            prepare_trsm,
            enqueue_tiled_trsm);
        const ConcurrentTiming tiled_trsm_concurrent = measure_concurrent(
            control_stream,
            critical_stream,
            far_stream,
            options.warmup,
            options.iterations,
            prepare_critical,
            enqueue_critical,
            prepare_trsm,
            enqueue_tiled_trsm);
        print_result(
            "TRSM tile x" + std::to_string(repeats),
            critical_ms,
            tiled_trsm_ms,
            tiled_trsm_concurrent);
    }

    HIP_CHECK(hipStreamSynchronize(critical_stream));
    int host_info = 0;
    HIP_CHECK(hipMemcpy(
        &host_info, info.get(), sizeof(int), hipMemcpyDeviceToHost));
    if (host_info != 0) {
        std::cerr << "POTRF failed with info=" << host_info << "\n";
    }

    ROCBLAS_CHECK(rocblas_destroy_handle(critical_handle));
    ROCBLAS_CHECK(rocblas_destroy_handle(far_handle));
    HIP_CHECK(hipStreamDestroy(control_stream));
    HIP_CHECK(hipStreamDestroy(critical_stream));
    HIP_CHECK(hipStreamDestroy(far_stream));

    return host_info == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
