#pragma once

#include <string>
#include <vector>

#include "utils/benchmark/residual_metric.h"

struct BenchmarkResult
{
    std::string method;
    std::string nb_str;

    double time_ms        = 0.0;
    double median_time_ms = 0.0;
    double stddev_time_ms = 0.0;
    double min_time_ms    = 0.0;
    double max_time_ms    = 0.0;
    double gflops         = 0.0;
    double speedup        = 0.0;

    double forward_error = -1.0;
    ResidualMetric residual_metric = ResidualMetric::None;
    double residual_value = -1.0;

    std::vector<double> iteration_times_ms;
};
