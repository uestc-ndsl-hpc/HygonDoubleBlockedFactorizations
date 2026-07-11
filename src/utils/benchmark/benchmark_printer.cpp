#include "utils/benchmark/benchmark_printer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

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

    return;
}
