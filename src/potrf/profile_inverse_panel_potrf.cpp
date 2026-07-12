#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include "potrf/double_blocked_potrf.h"
#include "potrf/profile_inverse_panel_potrf.h"
#include "utils/benchmark/stage_breakdown_profiler.h"
#include "utils/common/error_check.h"
#include "utils/matrix/matrix_generator.h"
#include "utils/memory/device_buffer.h"

namespace {

int read_int_arg(int argc, char** argv, const char* name, int fallback)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }

    return fallback;
}

void reset_matrix(float* destination,
                  const float* source,
                  size_t bytes,
                  int* info,
                  hipStream_t stream)
{
    HIP_CHECK(hipMemcpyAsync(
        destination,
        source,
        bytes,
        hipMemcpyDeviceToDevice,
        stream));
    HIP_CHECK(hipMemsetAsync(info, 0, sizeof(int), stream));

    return;
}

} // namespace

int main(int argc, char** argv)
{
    const int n = read_int_arg(argc, argv, "--n", 40960);
    const int nb = read_int_arg(argc, argv, "--nb", 8192);
    const int b = read_int_arg(argc, argv, "--b", 512);
    const int syrk_k_tile =
        read_int_arg(argc, argv, "--syrk-k-tile", nb);
    const int requested_panel_rows =
        read_int_arg(argc, argv, "--panel-rows", 8192);
    const int warmup = read_int_arg(argc, argv, "--warmup", 1);
    const int iters = read_int_arg(argc, argv, "--iters", 3);

    if (n <= 0 || nb <= 0 || b <= 0 || b > nb ||
        syrk_k_tile <= 0 || requested_panel_rows <= 0 || warmup < 0 ||
        iters <= 0) {
        std::cerr << "Usage: profile_inverse_panel_potrf "
                  << "--n N --nb NB --b B [--syrk-k-tile K] "
                  << "[--panel-rows M] [--warmup W] [--iters I]\n";
        return EXIT_FAILURE;
    }

    const int panel_rows = std::min(requested_panel_rows, n);
    const size_t num_elements = static_cast<size_t>(n) * n;
    const size_t bytes = num_elements * sizeof(float);

    rocblas_handle handle;
    ROCBLAS_CHECK(rocblas_create_handle(&handle));
    ROCBLAS_CHECK(
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    hipStream_t stream;
    ROCBLAS_CHECK(rocblas_get_stream(handle, &stream));

    DeviceBuffer<float> original(num_elements);
    DeviceBuffer<float> matrix(num_elements);
    DeviceBuffer<float> inverse_workspace(
        static_cast<size_t>(b) * b);
    DeviceBuffer<float> panel_workspace(
        static_cast<size_t>(panel_rows) * b);
    DeviceBuffer<int> local_info(1);
    DeviceBuffer<int> info(1);

    std::cout << "Generating SPD matrix: N=" << n << "\n";
    generate_spd_matrix<float>(handle, original, n, n);

    std::cout << "Profiling inverse-panel POTRF: N=" << n
              << ", NB=" << nb
              << ", b=" << b
              << ", syrk_k_tile=" << syrk_k_tile
              << ", panel_rows=" << panel_rows
              << ", warmup=" << warmup
              << ", iters=" << iters << "\n";

    for (int iteration = 0; iteration < warmup; ++iteration) {
        reset_matrix(matrix, original, bytes, info, stream);
        double_blocked_potrf_inverse_panel(
            handle,
            matrix,
            n,
            n,
            nb,
            b,
            syrk_k_tile,
            inverse_workspace,
            b,
            panel_workspace,
            panel_rows,
            local_info,
            info);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    StageBreakdownProfiler profiler("GFLOPS");
    for (int iteration = 0; iteration < iters; ++iteration) {
        reset_matrix(matrix, original, bytes, info, stream);
        double_blocked_potrf_inverse_panel_profiled(
            handle,
            matrix,
            n,
            n,
            nb,
            b,
            syrk_k_tile,
            inverse_workspace,
            b,
            panel_workspace,
            panel_rows,
            local_info,
            info,
            profiler);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    int host_info = 0;
    HIP_CHECK(hipMemcpy(
        &host_info, info, sizeof(int), hipMemcpyDeviceToHost));
    if (host_info != 0) {
        std::cerr << "POTRF failed with info=" << host_info << "\n";
        ROCBLAS_CHECK(rocblas_destroy_handle(handle));
        return EXIT_FAILURE;
    }

    profiler.print(
        "Inverse-Panel POTRF Stage Breakdown (N=" +
            std::to_string(n) + ", NB=" + std::to_string(nb) +
            ", b=" + std::to_string(b) + ", panel_rows=" +
            std::to_string(panel_rows) + ")",
        110);

    ROCBLAS_CHECK(rocblas_destroy_handle(handle));
    return EXIT_SUCCESS;
}
