#pragma once
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <vector>

struct RocblasStreamContext {
    hipStream_t stream;
    rocblas_handle handle;
};

class StreamPool {
public:
    std::vector<RocblasStreamContext> contexts;

    StreamPool(int num_streams = 4);
    ~StreamPool();

    RocblasStreamContext& get(int idx);
    int size() const;
};