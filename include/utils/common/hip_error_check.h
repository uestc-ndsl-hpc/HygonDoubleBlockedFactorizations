#pragma once

#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>

#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t error = cmd;                                            \
        if (error != hipSuccess) {                                         \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__    \
                      << ": " << hipGetErrorString(error) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)
