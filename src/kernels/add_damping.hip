#include <hip/hip_runtime.h>

#include "kernels/add_damping.h"

extern "C" {

// ======================================================
// Add damping to SPD matrix:
//
// A = A + lambda * I
//
// Only diagonal entries are modified.
// Used when FP16 update causes POTRF failure.
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
        A[idx + idx * lda] += lambda;
    }
}

// ======================================================
// launcher
// ======================================================

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
}

} // extern "C"