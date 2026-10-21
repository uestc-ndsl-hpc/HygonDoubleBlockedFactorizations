#pragma once

#include <limits>

inline bool parse_benchmark_seed(const char* text, unsigned int& seed)
{
    if (text == nullptr || *text == '\0') return false;
    unsigned int value = 0;
    for (const char* digit = text; *digit != '\0'; ++digit) {
        if (*digit < '0' || *digit > '9') return false;
        const unsigned int next = static_cast<unsigned int>(*digit - '0');
        if (value > (std::numeric_limits<unsigned int>::max() - next) / 10u) {
            return false;
        }
        value = value * 10u + next;
    }
    seed = value;
    return true;
}
