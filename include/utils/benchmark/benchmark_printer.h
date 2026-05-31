#pragma once

#include <vector>
#include <string>
#include "utils/benchmark_result.h"

enum class ErrorMode
{
    None     = 0,
    Forward  = 1,
    Backward = 2,
    Both     = 3
};

void print_benchmark_results(
    const std::string& title,
    const std::vector<BenchmarkResult>& results,
    ErrorMode error_mode,
    int line_width = 110);