# Supplementary experiment evidence

This export supplements the original paper data without replacing it.

| Directory | Contents |
|---|---|
| `fp64_residual/` | 45 full CPU FP64 residual checks of GPU FP32 factorizations, collected 2026-10-21; 25 original-seed and 20 additional-seed method results |
| `stage_spread/` | Six profiles at n=40960, each with one warmup and ten measured runs, collected 2026-08-27; raw per-run stage tables and eight matched-stage statistics |
| `rocsolver_boundary/` | Six rocSOLVER process outcomes at n=46340, 46341 and 49152, collected 2026-08-27; commands, logs, exit codes and derived outcomes |

The residual logs contain single-sample timings collected for validation, not
the original end-to-end performance protocol. Method results sharing a
routine/order/seed are not independent input matrices. See
`docs/numerical_validation.md` at the repository root for the residual method,
build instructions, matrix families, parameters and coverage limits.

For stage statistics, component times are summed within each measured run
before computing sample standard deviations (ddof=1). The archived logs permit
independent reanalysis. Statistics use the `STAGE_SAMPLE` blocks, not the trailing
aggregate tables, which may contain zero totals. The temporary per-sample profiler patch was not retained;
these data do not provide a bitwise-reproducible profiler executable.

The boundary data directly establish successful execution at n=46340 and
invalid-address process failures at n=46341 and n=49152 in the tested stack.
Their alignment with n^2 crossing INT_MAX suggests, but does not prove, a 32-bit
indexing limitation. Process exit codes are not rocSOLVER API return codes.

Absolute server/account and temporary executable paths have been replaced with
relative paths or placeholders. Numerical measurements and diagnostic error
messages are retained. `export_provenance.json` records original and public
SHA256 values and transformations. Private correspondence, account information,
process monitoring and third-party binaries are excluded. `SHA256SUMS` covers
every file in this directory except itself. Checksums detect later changes;
they do not independently authenticate the original measurement process.

Run `python3 scripts/verify_paper_supplements.py` from the repository root to
verify checksums, coverage, raw/summary agreement, stage statistics, boundary
outcomes and the tested-source snapshot.
