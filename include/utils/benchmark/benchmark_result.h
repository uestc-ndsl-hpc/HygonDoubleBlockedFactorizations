#pragma once

#include <string>

struct BenchmarkResult
{
    std::string method;
    std::string nb_str;

    double time_ms        = 0.0;
    double gflops         = 0.0;
    double speedup        = 0.0;

    double forward_error  = -1.0;
    double backward_error = -1.0;
};