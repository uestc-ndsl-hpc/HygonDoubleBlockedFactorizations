#pragma once

#include <cstdint>
#include <numeric>

namespace benchmark_residual {

class DeterministicIndexSampler
{
public:
    explicit DeterministicIndexSampler(
        std::uint64_t total,
        std::uint64_t seed = 20260716u)
        : total_(total),
          current_(total > 0 ? seed % total : 0),
          step_(select_coprime_step(total))
    {
    }

    std::uint64_t next()
    {
        const std::uint64_t result = current_;
        if (total_ > 1) {
            const std::uint64_t wrap_point = total_ - step_;
            current_ = current_ >= wrap_point
                ? current_ - wrap_point
                : current_ + step_;
        }
        return result;
    }

private:
    static std::uint64_t select_coprime_step(std::uint64_t total)
    {
        if (total <= 1) {
            return 0;
        }

        std::uint64_t step = 0x9e3779b97f4a7c15ULL % total;
        if (step == 0) {
            step = 1;
        }
        while (std::gcd(step, total) != 1) {
            ++step;
            if (step == total) {
                step = 1;
            }
        }
        return step;
    }

    std::uint64_t total_;
    std::uint64_t current_;
    std::uint64_t step_;
};

} // namespace benchmark_residual
