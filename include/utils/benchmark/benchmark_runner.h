#pragma once
#include <cmath>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include <numeric>
#include <algorithm>
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/gpu_timer.h"
#include "utils/common/error_check.h"

class BenchmarkRunner {
public:
    BenchmarkRunner(hipStream_t stream,
                    double& baseline_time,
                    std::vector<BenchmarkResult>& results,
                    ResidualMetric residual_metric)
        : stream_(stream),
          baseline_time_(baseline_time),
          results_(results),
          residual_metric_(residual_metric)
    {
    }

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
        std::vector<double> iter_times(profile_iters);

        GPUTimer timer;
        for (int i = 0; i < profile_iters; ++i) {
            reset();
            HIP_CHECK(hipDeviceSynchronize());
            
            iter_times[i] = static_cast<double>(timer.measure(stream_, [&]() {
                target();
            }));
        }
        HIP_CHECK(hipDeviceSynchronize());

        const double total_ms = std::accumulate(
            iter_times.begin(), iter_times.end(), 0.0);
        const double ms = total_ms / static_cast<double>(profile_iters);

        std::vector<double> sorted_times = iter_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        const size_t middle = sorted_times.size() / 2;
        const double median_ms = sorted_times.size() % 2 == 0
            ? (sorted_times[middle - 1] + sorted_times[middle]) / 2.0
            : sorted_times[middle];
        const double min_ms = sorted_times.front();
        const double max_ms = sorted_times.back();

        double variance = 0.0;
        for (double value : iter_times) {
            const double delta = value - ms;
            variance += delta * delta;
        }
        const double stddev_ms = iter_times.size() > 1
            ? std::sqrt(variance / static_cast<double>(iter_times.size() - 1))
            : 0.0;

        double gflops = calc_gflops(ms);
        double forward_error = -1.0;
        double residual_value = -1.0;

        reset();
        HIP_CHECK(hipDeviceSynchronize());
        target();
        HIP_CHECK(hipDeviceSynchronize());
        verify(forward_error, residual_value);

        const char* residual_limit_environment
            = residual_metric_limit_environment(residual_metric_);
        const double residual_limit = residual_limit_environment == nullptr
            ? -1.0
            : read_optional_double(residual_limit_environment, -1.0);
        if (residual_limit >= 0.0
            && (residual_value < 0.0
                || !std::isfinite(residual_value)
                || residual_value > residual_limit)) {
            throw std::runtime_error(
                name + " " + residual_metric_name(residual_metric_) + " "
                + std::to_string(residual_value)
                + " exceeds " + residual_limit_environment + "="
                + std::to_string(residual_limit));
        }

        if (name.find("rocBLAS") != std::string::npos) {
            baseline_time_ = ms;
        }
        double speedup = (baseline_time_ > 0.0) ? (baseline_time_ / ms) : 0.0;

        results_.push_back({
            name,
            nb_str,
            ms,
            median_ms,
            stddev_ms,
            min_ms,
            max_ms,
            gflops,
            speedup,
            forward_error,
            residual_metric_,
            residual_value,
            std::move(iter_times)
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

    static double read_optional_double(const char* variable, double fallback)
    {
        const char* value = std::getenv(variable);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }

        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || *end != '\0' || !std::isfinite(parsed)) {
            return fallback;
        }

        return parsed;
    }

    hipStream_t stream_;
    double& baseline_time_;
    std::vector<BenchmarkResult>& results_;
    ResidualMetric residual_metric_;
};
