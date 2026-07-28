# rocSOLVER `n=49152` diagnostic probe

## Scope

- Repository revision: `6098e38e1f8f5e56beaeb1446df4bc207455cb12`
- Device: physical HCU 1, Hygon K100_AI
- Stack: DTK 26.04 installation, DCC 25.10.0-0, rocSOLVER 3.20.0,
  rocBLAS 4.3.0
- Matrix order: `n=49152`
- One process per factorization, zero warmups, one requested measurement
- Methods: rocSOLVER `spotrf` and partial-pivoting `sgetrf`
- Archive SHA256:
  `2c4bf49c1f148129fe017497aabdcf95afed0b741ef7f6ae5922af28a132935a`

The device was idle before the probe and returned to 2 MiB of reported VRAM
use after both processes terminated.

## Observed failures

| Routine | Process result | First queued kernel at the fault | Invalid address |
|---|---:|---|---|
| POTRF | signal 6, exit 134 | `syrk_lower_nc_kernel` | `0x14ec049ab000` |
| GETRF | signal 6, exit 134 | `getrf_row_permutate` | `0x145ca0806000` |

Both failures are KFD/HSA kernel virtual-memory faults with `Error code: 1`.
They are not normal rocSOLVER status returns, and neither process reaches a
valid timing or `info` result.

## Memory evidence

The device reports 65,520 MiB total VRAM.

| Routine | Peak process VRAM | Fraction of device VRAM | VRAM still unused |
|---|---:|---:|---:|
| POTRF | 20,393 MiB | 31.1% | 45,127 MiB |
| GETRF | 19,163 MiB | 29.2% | 46,357 MiB |

The failures are therefore not device-memory-capacity exhaustion. The GETRF
peak includes the benchmark harness's 384 MiB pivot tile even though the
rocSOLVER-only case does not use DB-LU pivot batching.

## Address-range evidence

An FP32 matrix at this order contains

```text
n^2 = 49152^2 = 2,415,919,104 = 0x90000000 elements
bytes = 9,663,676,416 = 0x240000000 bytes = 9 GiB
```

This element count exceeds signed 32-bit `INT_MAX = 2,147,483,647`. By
contrast, `40960^2 = 1,677,721,600` remains below `INT_MAX`, and the largest
integer order satisfying `n^2 <= INT_MAX` is 46,340.

For POTRF, the logged matrix allocation starts at `0x14ee00800000` and spans
9 GiB, whereas the fault address is 7.936 GiB below that allocation. For
GETRF, the matrix allocation starts at `0x145e9fc00000`, while the fault
address is 7.988 GiB below it. Both accesses are outside the corresponding
matrix allocations rather than near their upper bounds.

Together with the kernel identities and the transition from a successful
`n=40960` to a failing `n=49152`, these observations are consistent with a
signed 32-bit matrix-offset overflow inside the tested rocSOLVER kernels.
This is an evidence-based diagnosis, not a vendor-confirmed source-level
finding.

## Paper-safe conclusion

On the tested DTK 26.04 stack, rocSOLVER `spotrf` and partial-pivoting
`sgetrf` both terminate at `n=49152` with KFD invalid-address faults rather
than out-of-memory errors. Peak VRAM remains below 32% of the K100_AI
capacity. The fault addresses and the fact that `49152^2` exceeds the signed
32-bit element-count limit are consistent with 32-bit internal offset
overflow in the reported SYRK and row-permutation kernels. Consequently,
external rocSOLVER comparisons are restricted to the largest successful
tested order, `n=40960`.
