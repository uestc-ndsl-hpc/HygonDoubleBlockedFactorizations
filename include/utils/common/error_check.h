#ifndef ERROR_CHECK_H
#define ERROR_CHECK_H

#include <iostream>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define HIP_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t error = cmd;                                            \
        if (error != hipSuccess) {                                         \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__    \
                      << ": " << hipGetErrorString(error) << std::endl;    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while(0)

#define ROCBLAS_CHECK(cmd)                                                 \
    do {                                                                   \
        rocblas_status status = cmd;                                       \
        if (status != rocblas_status_success) {                            \
            std::cerr << "rocBLAS error at " << __FILE__ << ":" << __LINE__\
                      << ": " << rocblas_status_to_string(status)          \
                      << std::endl;                                        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while(0)

#endif