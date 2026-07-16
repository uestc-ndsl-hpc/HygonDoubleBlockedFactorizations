#include "utils/benchmark/benchmark_printer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {

std::string csv_quote(const std::string& value)
{
    std::string quoted = "\"";
    for (char character : value) {
        if (character == '"') {
            quoted += "\"\"";
        } else {
            quoted += character;
        }
    }
    quoted += '"';

    return quoted;
}

void write_timing_csv(const std::string& title,
                      const std::vector<BenchmarkResult>& results)
{
    const char* path = std::getenv("BENCHMARK_CSV_PATH");
    if (path == nullptr || *path == '\0') {
        return;
    }

    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        std::cerr << "Unable to open BENCHMARK_CSV_PATH: " << path << "\n";
        return;
    }

    output << "benchmark,method,parameters,iteration,time_ms,mean_ms,"
              "median_ms,stddev_ms,min_ms,max_ms,gflops,speedup,"
              "forward_error,backward_error\n";
    output << std::setprecision(12);

    for (const auto& result : results) {
        for (size_t iteration = 0;
             iteration < result.iteration_times_ms.size();
             ++iteration) {
            output << csv_quote(title) << ','
                   << csv_quote(result.method) << ','
                   << csv_quote(result.nb_str) << ','
                   << iteration << ','
                   << result.iteration_times_ms[iteration] << ','
                   << result.time_ms << ','
                   << result.median_time_ms << ','
                   << result.stddev_time_ms << ','
                   << result.min_time_ms << ','
                   << result.max_time_ms << ','
                   << result.gflops << ','
                   << result.speedup << ','
                   << result.forward_error << ','
                   << result.backward_error << '\n';
        }
    }

    std::cout << "Raw timing CSV: " << path << "\n";
}

} // namespace

void print_benchmark_results(
    const std::string& title,
    const std::vector<BenchmarkResult>& results,
    ErrorMode error_mode,
    int line_width)
{
    using namespace std;

    int method_width = 15;
    int parameter_width = 10;
    for (const auto& result : results) {
        method_width = max(method_width, static_cast<int>(result.method.size()));
        parameter_width = max(
            parameter_width,
            static_cast<int>(result.nb_str.size()));
    }
    const int table_width = max(
        line_width,
        method_width + parameter_width + 80);

    cout << "\n";
    cout << string(table_width, '=') << "\n";
    cout << title << "\n";
    cout << string(table_width, '=') << "\n";

    cout << left
         << setw(method_width) << "Method"
         << " | " << setw(parameter_width) << "Parameters"
         << " | " << setw(12) << "Time(ms)"
         << " | " << setw(12) << "GFLOPS"
         << " | " << setw(10) << "Speedup"
         << " | " << setw(14) << "Forward Error"
         << " | " << setw(14) << "Backward Error"
         << "\n";

    cout << string(table_width, '-') << "\n";

    for (const auto& res : results) {
        cout << left
             << setw(method_width) << res.method
             << " | " << setw(parameter_width) << res.nb_str
             << " | " << right << setw(10) << fixed << setprecision(3) << res.time_ms << "  "
             << " | " << right << setw(10) << fixed << setprecision(2) << res.gflops << "  "
             << " | ";

        if (res.speedup > 0.0)
            cout << right << setw(8) << fixed << setprecision(2) << res.speedup << "x ";
        else
            cout << setw(9) << "N/A ";

        cout << " | ";

        if (error_mode == ErrorMode::Forward || error_mode == ErrorMode::Both)
            cout << scientific << setprecision(2) << setw(12) << res.forward_error;
        else
            cout << setw(12) << "N/A";

        cout << " | ";

        if (error_mode == ErrorMode::Backward || error_mode == ErrorMode::Both)
            cout << scientific << setprecision(2) << setw(12) << res.backward_error;
        else
            cout << setw(12) << "N/A";

        cout << "\n";
    }

    cout << string(table_width, '=') << "\n";

    cout << "Timing Statistics (ms)\n";
    cout << string(table_width, '-') << "\n";
    cout << left
         << setw(method_width) << "Method"
         << " | " << setw(parameter_width) << "Parameters"
         << " | " << right << setw(12) << "Median"
         << " | " << setw(12) << "StdDev"
         << " | " << setw(12) << "Minimum"
         << " | " << setw(12) << "Maximum"
         << " | " << setw(10) << "Samples"
         << "\n";
    cout << string(table_width, '-') << "\n";

    for (const auto& res : results) {
        cout << left
             << setw(method_width) << res.method
             << " | " << setw(parameter_width) << res.nb_str
             << " | " << right << setw(12) << fixed << setprecision(3)
             << res.median_time_ms
             << " | " << setw(12) << res.stddev_time_ms
             << " | " << setw(12) << res.min_time_ms
             << " | " << setw(12) << res.max_time_ms
             << " | " << setw(10) << res.iteration_times_ms.size()
             << "\n";
    }

    cout << string(table_width, '=') << "\n";
    write_timing_csv(title, results);

    return;
}
