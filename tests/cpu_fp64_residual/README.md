# CPU FP64 residual tests

These tests run without HIP or a GPU. They compare the tiled LU and Cholesky
residual implementation with an independent, unblocked long-double oracle.
Coverage includes pivot direction, padded leading dimensions, tile tails,
ignored upper-triangle entries, FP32 product-rounding detection, seed parsing,
invalid arguments, and scaled norm accumulation. There are 170 checks.

```sh
cmake -S tests/cpu_fp64_residual -B build-cpu-residual
cmake --build build-cpu-residual
ctest --test-dir build-cpu-residual --output-on-failure
```

The default portable implementation is intended for small tests. To test a host
CBLAS implementation, add `-DHYGON_CPU_FP64_USE_CBLAS=ON`. If discovery needs
help, set `HYGON_CPU_CBLAS_INCLUDE_DIR` and `HYGON_CPU_CBLAS_LIBRARY` to the host
CBLAS header directory and library file. Do not use rocBLAS as the host BLAS.

`scipy_cblas_compat/` is an optional LP64 symbol adapter, not a BLAS library.
Use it only with a SciPy library verified to export `scipy_cblas_dgemm` with
32-bit BLAS integers. It is not compatible with an ILP64 ABI. `probe.cpp` checks
the backend configuration, thread count and preservation of the FP64 product
`(1 + 2^-23)^2`. The published experiment used this adapter with OpenBLAS 0.3.28.
Both the 170-check CBLAS test and the precision probe passed on the collection
host. No third-party library binaries are distributed here.
