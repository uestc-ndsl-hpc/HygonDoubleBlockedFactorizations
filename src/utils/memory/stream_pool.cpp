#include "utils/memory/stream_pool.h"
#include "utils/common/error_check.h"

StreamPool::StreamPool(int num_streams) {
    contexts.resize(num_streams);

    for (int i = 0; i < num_streams; ++i) {
        HIP_CHECK(hipStreamCreateWithFlags(&contexts[i].stream, hipStreamNonBlocking));
        ROCBLAS_CHECK(rocblas_create_handle(&contexts[i].handle));
        ROCBLAS_CHECK(rocblas_set_stream(contexts[i].handle, contexts[i].stream));

        HIP_CHECK(hipEventCreateWithFlags(&contexts[i].event, hipEventDisableTiming));
    }

    HIP_CHECK(hipEventCreateWithFlags(&main_event, hipEventDisableTiming));
}

StreamPool::~StreamPool() {
    for (auto& ctx : contexts) {
        ROCBLAS_CHECK(rocblas_destroy_handle(ctx.handle));
        HIP_CHECK(hipEventDestroy(ctx.event));
        HIP_CHECK(hipStreamDestroy(ctx.stream));
    }
    HIP_CHECK(hipEventDestroy(main_event));
}

RocblasStreamContext& StreamPool::get(int idx) {
    return contexts[idx % contexts.size()];
}

int StreamPool::size() const {
    return contexts.size();
}