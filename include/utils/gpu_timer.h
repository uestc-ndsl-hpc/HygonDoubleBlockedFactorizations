#pragma once

#include <hip/hip_runtime.h>
#include "utils/error_check.h"

// ============================================================================
// GPU Timer
// ============================================================================

class GPUTimer {
public:
    GPUTimer() {
        HIP_CHECK(hipEventCreate(&start_));
        HIP_CHECK(hipEventCreate(&stop_));
    }

    ~GPUTimer() {
        HIP_CHECK(hipEventDestroy(start_));
        HIP_CHECK(hipEventDestroy(stop_));
    }

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    template <typename Func>
    float measure(hipStream_t stream, Func&& func) {
        HIP_CHECK(hipEventRecord(start_, stream));

        func();

        HIP_CHECK(hipEventRecord(stop_, stream));

        HIP_CHECK(hipEventSynchronize(stop_));

        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));

        return ms;
    }

private:
    hipEvent_t start_;
    hipEvent_t stop_;
};