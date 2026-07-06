cmake_minimum_required(VERSION 3.16)

include(cmake/HygonDtk.cmake)

project(TRSM_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/BenchmarkCommon.cmake)
add_subdirectory(src)
