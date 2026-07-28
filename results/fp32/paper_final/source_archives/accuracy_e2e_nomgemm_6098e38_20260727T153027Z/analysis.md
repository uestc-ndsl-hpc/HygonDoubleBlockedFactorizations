# Common-grid FP32 factorization residuals

This supplement evaluates full relative Frobenius-norm factorization residuals
on the five matrix orders used for the common rocSOLVER end-to-end comparison:
`8192`, `16384`, `24576`, `32768`, and `40960`.

The POTRF run compares rocSOLVER with the final `DoubleBlock+KBSYRK` method at
`B/b/T_K=8192/1024/8192`. `gemm_m_tile=0`, and the CSV files contain no MGEMM
method. The same invocation and input are used for both methods at each order.

| n | rocSOLVER POTRF | DB-POTRF |
|---:|---:|---:|
| 8192 | 5.40958290914e-7 | 6.45258317279e-7 |
| 16384 | 1.44616759225e-6 | 1.44934582551e-6 |
| 24576 | 1.69402547159e-6 | 1.77855542959e-6 |
| 32768 | 1.87561947718e-6 | 1.97447146406e-6 |
| 40960 | 2.14096120296e-6 | 2.31226494152e-6 |

The run also supplies the previously missing full-residual LU point at
`n=32768`, using the final fixed policy
`16384/4096/iterative/2048/2048/8192`.

| n | rocSOLVER GETRF | DB-GETRF |
|---:|---:|---:|
| 32768 | 9.89523832835e-5 | 8.46095460418e-5 |

All rows use the `normwise_backward_error` metric. The maximum new POTRF
residual is `2.31226494152e-6`, below the `1e-5` validation limit. Both LU rows
are below the `5e-4` limit. The driver completed with exit code zero, its
artifact audit passed, and `checksums.sha256` records the files included in the
immutable server archive.
