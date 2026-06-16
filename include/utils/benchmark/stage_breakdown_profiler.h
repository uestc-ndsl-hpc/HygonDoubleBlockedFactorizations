#pragma once

#include <hip/hip_runtime.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include "utils/common/error_check.h"

struct StageBreakdownEntry {
    std::string name;
    double total_time_ms = 0.0;
    double total_work = 0.0;
    int call_count = 0;
};

class StageBreakdownProfiler {
public:
    explicit StageBreakdownProfiler(std::string throughput_label = "GFLOPS")
        : throughput_label_(std::move(throughput_label)) {}

    StageBreakdownProfiler(const StageBreakdownProfiler&) = delete;
    StageBreakdownProfiler& operator=(const StageBreakdownProfiler&) = delete;

    void clear()
    {
        entries_.clear();
    }

    void add_sample(const std::string& name, double elapsed_ms, double work = 0.0)
    {
        auto it = std::find_if(entries_.begin(), entries_.end(),
                               [&](const StageBreakdownEntry& entry) {
                                   return entry.name == name;
                               });

        if (it == entries_.end()) {
            entries_.push_back({name, elapsed_ms, work, 1});
            return;
        }

        it->total_time_ms += elapsed_ms;
        it->total_work += work;
        it->call_count += 1;
    }

    template <typename Func>
    void measure(hipStream_t stream, const std::string& name, double work, Func&& func)
    {
        hipEvent_t start;
        hipEvent_t stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        HIP_CHECK(hipEventRecord(start, stream));
        func();
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        add_sample(name, static_cast<double>(elapsed_ms), work);
    }

    double total_time_ms() const
    {
        double total = 0.0;
        for (const auto& entry : entries_) {
            total += entry.total_time_ms;
        }
        return total;
    }

    bool empty() const
    {
        return entries_.empty();
    }

    void print(const std::string& title, int line_width = 100) const
    {
        using namespace std;

        cout << "\n";
        cout << string(line_width, '=') << "\n";
        cout << title << "\n";
        cout << string(line_width, '=') << "\n";

        cout << left
             << setw(24) << "Stage"
             << " | " << right << setw(12) << "Total(ms)"
             << " | " << setw(10) << "Calls"
             << " | " << setw(12) << "Avg(ms)"
             << " | " << setw(10) << "Share"
             << " | " << setw(12) << throughput_label_
             << "\n";

        cout << string(line_width, '-') << "\n";

        const double total_ms = total_time_ms();
        for (const auto& entry : entries_) {
            const double avg_ms = entry.call_count > 0
                ? entry.total_time_ms / static_cast<double>(entry.call_count)
                : 0.0;
            const double share = total_ms > 0.0 ? entry.total_time_ms * 100.0 / total_ms : 0.0;
            const double throughput = entry.total_time_ms > 0.0
                ? (entry.total_work / (entry.total_time_ms / 1000.0)) / 1e9
                : 0.0;

            cout << left << setw(24) << entry.name
                 << " | " << right << setw(12) << fixed << setprecision(3) << entry.total_time_ms
                 << " | " << setw(10) << entry.call_count
                 << " | " << setw(12) << fixed << setprecision(3) << avg_ms
                 << " | " << setw(9) << fixed << setprecision(2) << share << "%"
                 << " | " << setw(12) << fixed << setprecision(2) << throughput
                 << "\n";
        }

        cout << string(line_width, '=') << "\n";
    }

private:
    std::string throughput_label_;
    std::vector<StageBreakdownEntry> entries_;
};
