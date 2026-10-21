cmake_minimum_required(VERSION 3.16)
project(CpuFp64ResidualTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/CpuFp64Residual.cmake")

add_executable(test_cpu_fp64_residual test_cpu_fp64_residual.cpp)
target_include_directories(test_cpu_fp64_residual PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../../include")
configure_cpu_fp64_residual(test_cpu_fp64_residual)

enable_testing()
add_test(NAME cpu_fp64_residual COMMAND test_cpu_fp64_residual)
