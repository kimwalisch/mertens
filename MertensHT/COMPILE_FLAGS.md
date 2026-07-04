# Build flags

Performance-only; every configuration computes the same $M(n)$. The shared sieve's knobs also apply — see [`../sieve/COMPILE_FLAGS.md`](../sieve/COMPILE_FLAGS.md).

| Flag | Default | What it does |
|---|---|---|
| `BUCKET_SIEVE` | 1 | Large-prime bucket scheduler in the shared sieve, used by BruteM and the double-sum $\mu$ ranges. `make BUCKET_SIEVE=0` to disable. |
| `MHT_FACT_TILE` | 4096 | Tile size for the cache-blocked factorization sieve. Sized for L1; worth a sweep on x86. |
| `CPU_OVERLAP` | 0 | Runs LargeFree and LargeNonFree concurrently on two threads (they are independent phases, combined only at the end). Off by default: with well-tuned parameters the gain is ~0-2% and it raises peak memory, since both phases' working sets are live at once. Under `--profile`, per-phase timing is replaced by one combined region. See `OPTIMIZATIONS.md` item 25. Becomes `-DMHT_CPU_OVERLAP`. |

Plain defines go through the hook: `make EXTRA_CXXFLAGS="-DMHT_FACT_TILE=8192"`. Tuning parameters (`--vdiv`, `--C`, `--D`, `--ntasks`) are runtime options, not build flags — see the README. The defaults are the record-machine sweep values from Section 10 of the paper.
