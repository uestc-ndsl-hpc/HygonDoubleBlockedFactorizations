#pragma once
#include <hip/hip_runtime.h>
#include <string>
#include <vector>
#include "utils/benchmark/benchmark_result.h"
#include "utils/gpu_timer.h"

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
        // 1. Warmup
        reset();
        HIP_CHECK(hipDeviceSynchronize());
        target();

        // 2. Real Run
        reset();
        HIP_CHECK(hipDeviceSynchronize());

        GPUTimer timer;
        float ms = timer.measure(stream_, [&]() {
            target();
        });

        double gflops = calc_gflops(ms);
        double forward_error = -1.0;
        double backward_error = -1.0;

        // 3. Optional Verify (在测时循环外执行)
        verify(forward_error, backward_error);

        double speedup = (baseline_time_ > 0.0) ? (baseline_time_ / ms) : 0.0;

        results_.push_back({
            name,
            nb_str,
            static_cast<double>(ms),
            gflops,
            speedup,
            forward_error,
            backward_error
        });

        return static_cast<double>(ms);
    }

private:
    hipStream_t stream_;
    double& baseline_time_;
    std::vector<BenchmarkResult>& results_;
};