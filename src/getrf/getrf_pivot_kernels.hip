#include <cstddef>

#include <hip/hip_runtime.h>

#include "getrf/getrf_pivot_kernels.h"
#include "utils/common/hip_error_check.h"

namespace {

template <typename T>
__global__ void apply_panel_pivots_kernel(T* A,
                                          int lda,
                                          int n,
                                          int panel,
                                          int panel_size,
                                          const int* pivots)
{
    const int external_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int external_columns = n - panel_size;
    if (external_index >= external_columns) {
        return;
    }

    const int column = external_index < panel
        ? external_index
        : external_index + panel_size;
    T* column_data = A + static_cast<size_t>(column) * lda;

    // A thread owns one column, so the LAPACK pivot sequence is race-free.
    for (int offset = 0; offset < panel_size; ++offset) {
        const int row = panel + offset;
        const int pivot_row = panel + pivots[offset] - 1;
        if (pivot_row != row) {
            const T value = column_data[row];
            column_data[row] = column_data[pivot_row];
            column_data[pivot_row] = value;
        }
    }

    return;
}

__global__ void globalize_panel_pivots_kernel(int* pivots,
                                               int panel,
                                               int panel_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < panel_size) {
        // rocSOLVER reports a 1-based row within A(panel:n, panel:*).
        pivots[index] += panel;
    }

    return;
}

template <typename T>
void launch_apply_panel_pivots(T* A,
                               int lda,
                               int n,
                               int panel,
                               int panel_size,
                               int* pivots,
                               hipStream_t stream)
{
    if (panel_size <= 0) {
        return;
    }

    constexpr int threads = 256;
    const int external_columns = n - panel_size;
    if (external_columns > 0) {
        const int blocks = (external_columns + threads - 1) / threads;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(apply_panel_pivots_kernel<T>),
            dim3(blocks),
            dim3(threads),
            0,
            stream,
            A,
            lda,
            n,
            panel,
            panel_size,
            pivots);
        HIP_CHECK(hipGetLastError());
    }

    const int pivot_blocks = (panel_size + threads - 1) / threads;
    hipLaunchKernelGGL(
        globalize_panel_pivots_kernel,
        dim3(pivot_blocks),
        dim3(threads),
        0,
        stream,
        pivots,
        panel,
        panel_size);
    HIP_CHECK(hipGetLastError());

    return;
}

} // namespace

extern "C" {

void launch_apply_getrf_panel_pivots(float* A,
                                     int lda,
                                     int n,
                                     int panel,
                                     int panel_size,
                                     int* pivots,
                                     hipStream_t stream)
{
    launch_apply_panel_pivots(
        A, lda, n, panel, panel_size, pivots, stream);

    return;
}

void launch_apply_getrf_panel_pivots_double(double* A,
                                            int lda,
                                            int n,
                                            int panel,
                                            int panel_size,
                                            int* pivots,
                                            hipStream_t stream)
{
    launch_apply_panel_pivots(
        A, lda, n, panel, panel_size, pivots, stream);

    return;
}

} // extern "C"
