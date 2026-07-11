#pragma once
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/common/error_check.h"

class BenchmarkRunner {
public:
    BenchmarkRunner(hipStream_t stream, double& baseline_time, std::vector<BenchmarkResult>& results)
        : stream_(stream), baseline_time_(baseline_time), results_(results) {}

    template <typename ResetFunc, typename TargetFunc, typename VerifyFunc, typename GflopsFunc>
    double execute(
        const std::string& name,
        const std::string& nb_str,
        ResetFunc&& reset,
        TargetFunc&& target,
        VerifyFunc&& verify,
        GflopsFunc&& calc_gflops)
    {
        const int warmup_iters = read_iteration_count(
            "BENCHMARK_WARMUP_ITERS", 3, 0);
        for (int i = 0; i < warmup_iters; ++i) {
            reset();
            HIP_CHECK(hipDeviceSynchronize());
            target();
        }
        HIP_CHECK(hipDeviceSynchronize());

        const int profile_iters = read_iteration_count(
            "BENCHMARK_PROFILE_ITERS", 10, 1);
        std::vector<float> iter_times(profile_iters);

        GPUTimer timer;
        for (int i = 0; i < profile_iters; ++i) {
            reset();
            HIP_CHECK(hipDeviceSynchronize());
            
            iter_times[i] = timer.measure(stream_, [&]() {
                target();
            });
        }
        HIP_CHECK(hipDeviceSynchronize());

        float total_ms = std::accumulate(iter_times.begin(), iter_times.end(), 0.0f);
        double ms = static_cast<double>(total_ms / profile_iters);

        double gflops = calc_gflops(ms);
        double forward_error = -1.0;
        double backward_error = -1.0;

        reset();
        HIP_CHECK(hipDeviceSynchronize());
        target();
        HIP_CHECK(hipDeviceSynchronize());
        verify(forward_error, backward_error);

        if (name.find("rocBLAS") != std::string::npos) {
            baseline_time_ = ms;
        }
        double speedup = (baseline_time_ > 0.0) ? (baseline_time_ / ms) : 0.0;

        results_.push_back({
            name,
            nb_str,
            ms,
            gflops,
            speedup,
            forward_error,
            backward_error
        });

        return ms;
    }

private:
    static int read_iteration_count(
        const char* variable,
        int fallback,
        int minimum)
    {
        const char* value = std::getenv(variable);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }

        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || parsed < minimum || parsed > 100000) {
            return fallback;
        }

        return static_cast<int>(parsed);
    }

    hipStream_t stream_;
    double& baseline_time_;
    std::vector<BenchmarkResult>& results_;
};
