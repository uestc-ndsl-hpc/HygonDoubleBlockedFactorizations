#pragma once

enum class ErrorMode
{
    None = 0,
    Forward = 1,
    Backward = 2,
    Both = 3
};

inline ErrorMode parse_benchmark_error_mode(int error_input)
{
    if (error_input == 1) {
        return ErrorMode::Forward;
    }
    if (error_input == 2) {
        return ErrorMode::Backward;
    }
    if (error_input == 3) {
        return ErrorMode::Both;
    }

    return ErrorMode::None;
}
