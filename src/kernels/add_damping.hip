#include <hip/hip_runtime.h>

#include "kernels/add_damping.h"

extern "C" {

// ======================================================
// Add damping to SPD matrix (Float version)
// ======================================================
__global__ void add_damping_kernel(
    float* A,
    int n,
    int lda,
    float lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        A[idx + (size_t)idx * lda] += lambda;
    }

    return;
}

void launch_add_damping(
    float* A,
    int n,
    int lda,
    float lambda,
    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    int GRID = (n + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        add_damping_kernel,
        dim3(GRID),
        dim3(BLOCK),
        0,
        stream,
        A,
        n,
        lda,
        lambda);

    return;
}

// ======================================================
// Add damping to SPD matrix (Double version)
// ======================================================
__global__ void add_damping_kernel_double(
    double* A,
    int n,
    int lda,
    double lambda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        A[idx + (size_t)idx * lda] += lambda;
    }

    return;
}

void launch_add_damping_double(
    double* A,
    int n,
    int lda,
    double lambda,
    hipStream_t stream)
{
    constexpr int BLOCK = 256;
    int GRID = (n + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        add_damping_kernel_double,
        dim3(GRID),
        dim3(BLOCK),
        0,
        stream,
        A,
        n,
        lda,
        lambda);

    return;
}

} // extern "C"
