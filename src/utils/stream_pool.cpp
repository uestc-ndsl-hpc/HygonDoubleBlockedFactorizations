#include "utils/stream_pool.h"
#include "utils/error_check.h"

StreamPool::StreamPool(int num_streams) {
    contexts.resize(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        HIP_CHECK(hipStreamCreate(&contexts[i].stream));
        ROCBLAS_CHECK(rocblas_create_handle(&contexts[i].handle));
        ROCBLAS_CHECK(rocblas_set_stream(contexts[i].handle, contexts[i].stream));
    }
}

StreamPool::~StreamPool() {
    for (auto& ctx : contexts) {
        ROCBLAS_CHECK(rocblas_destroy_handle(ctx.handle));
        HIP_CHECK(hipStreamDestroy(ctx.stream));
    }
}

RocblasStreamContext& StreamPool::get(int idx) {
    return contexts[idx % contexts.size()];
}

int StreamPool::size() const {
    return contexts.size();
}