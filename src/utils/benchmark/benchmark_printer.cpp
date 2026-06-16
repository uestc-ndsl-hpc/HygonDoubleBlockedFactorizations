#include "utils/benchmark/benchmark_printer.h"
#include <iostream>
#include <iomanip>

void print_benchmark_results(
    const std::string& title,
    const std::vector<BenchmarkResult>& results,
    ErrorMode error_mode,
    int line_width)
{
    using namespace std;

    cout << "\n";
    cout << string(line_width, '=') << "\n";
    cout << title << "\n";
    cout << string(line_width, '=') << "\n";

    cout << left
         << setw(15) << "Method"
         << " | " << setw(6) << "NB"
         << " | " << setw(12) << "Time(ms)"
         << " | " << setw(12) << "GFLOPS"
         << " | " << setw(10) << "Speedup"
         << " | " << setw(14) << "Forward Error"
         << " | " << setw(14) << "Backward Error"
         << "\n";

    cout << string(line_width, '-') << "\n";

    for (const auto& res : results) {
        cout << left
             << setw(15) << res.method
             << " | " << setw(6) << res.nb_str
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

    cout << string(line_width, '=') << "\n";

    return;
}
