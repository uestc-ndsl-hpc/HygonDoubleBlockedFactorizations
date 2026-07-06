#pragma once

#include <vector>
#include <string>
#include "utils/benchmark/benchmark_result.h"
#include "utils/benchmark/error_mode.h"

void print_benchmark_results(
    const std::string& title,
    const std::vector<BenchmarkResult>& results,
    ErrorMode error_mode,
    int line_width = 110);
