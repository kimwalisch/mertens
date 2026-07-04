# Optimizations: Original to Current Implementation

This documents performance optimizations applied to the Helfgott-Thompson
$O(n^{3/5})$ Mertens function code ([arXiv:2101.08773](https://arxiv.org/abs/2101.08773)).
The starting point was the authors' own implementation, published as the
[ancillary files of the arXiv submission](https://arxiv.org/src/2101.08773v4/anc).

Overall measured speedup vs. the original author code: **3.4x-4.5x** across $10^{16}$-$10^{23}$ on the M3 Ultra record machine (Table 6 of the paper).

Dependency changes vs. the original:

- **GMP dropped.** The original used GMP for exact integer roots of 128-bit
  quantities — square, cube, fourth, and fifth roots, mostly in setup code.
  All of these are now native integer root routines (float estimate plus
  exact fixup), so no arbitrary-precision library is needed.
- **Boost.Multiprecision added** (header-only, nothing linked). The
  256-bit fallbacks of §29 need exact `int256_t` arithmetic for Diophantine
  discriminants and products that overflow `__int128` past $\sim 10^{23}$.

---

## 1. Quotient Predictor (SArr + LargeNonFree)

**What:** Replaced direct `x/r` (128-bit division, ~40 cycles) with a linear extrapolation
scheme: `qEst = 2*qCur - qPrev`, verified/corrected by a single multiply. Cost: ~5-8 cycles.

**Where:**
- `sArr()` inner loop in `MertensHT.cpp`: predicts `x/r` as `r` increments by 1
- `largeNonFree()` main loop in `MertensHT.cpp`: predicts `x/n` as `n` decrements by 1

**Details:** Seeds with the first 2 true divisions, then predicts subsequent quotients.
Includes both a 64-bit fast path (when `x` fits in `unsigned long`) and a 128-bit fallback.

---

## 2. Precomputed Factorization Stencil

**What:** The original `fillfactblock()` called `prefillfact()` each invocation, allocating
and filling temporary arrays for primes {2,3,5,7,11} over a period P=2310. Replaced with
a one-time precomputed stencil of period FACT_P = 360360 (= 2^3 * 3^2 * 5 * 7 * 11 * 13),
stored in global arrays and copied with rotation each call.

**Where:** `Factorization.h`: `initFactStencil()` + rotated memcpy in `fillFactBlock()`

**Details:**
- Added prime 13 to the stencil (original only covered up to 11)
- `initFactStencil()` called once at startup
- Per-call overhead reduced from 3 mallocs + fills + frees to a rotated memcpy
- Stencil stored in `g_stencilFun`, `g_stencilSqfprod`, `g_stencilTabla`

---

## 3. Cache-Blocked Large-Prime Sieving

**What:** For large sieve blocks (m > 4096), the original single-pass over all primes
p > FACT_PMAX caused cache thrashing. Replaced with L1-sized sub-blocks of 4096 elements,
iterating primes within each sub-block for locality.

**Where:** `Factorization.h`: `fillFactBlock()`, large-block path

**Details:**
- Pre-collects all primes into arrays (`pv`, `pkv`, `nmv`, `tpv`, `nxt`)
- Processes in 4096-element blocks, advancing each prime's next-hit pointer
- Higher powers (p^2, p^3, ...) still single-pass (stride too large to benefit)

---

## 4. 64-bit Fast Path in BruteDoubleSum

**What:** Once `x/m` fits in a `long` (which it does for most of the summation range),
the inner n-loop uses 64-bit division instead of 128-bit `__divti3`.

**Where:** `MertensHT.cpp`: `bruteDoubleSum()`

**Details:**
- Computes `xm = x/m` once per outer iteration
- If `xm <= __LONG_MAX__`, casts to `long` and runs a pure 64-bit inner loop
- Eliminated the switch-on-g[n] branching; uses direct multiplication `g[n-n0] * (lxm/n)`
- Skips `f[m-m0] == 0` entries early

---

## 5. 64-bit Fast Paths in mod(), divf(), modl()

**What:** Added `__builtin_expect`-annotated fast paths that detect when both operands
fit in a `long`, bypassing 128-bit division.

**Where:** `IntegerMath.h`: `mod()`, `divf()`, `divfPos()`, `modl()`

**Details:**
- Uses `(uI)(a + (Int128)__LONG_MAX__) <= (uI)2*__LONG_MAX__` to test if `a` fits in `long`
- Falls through to original 128-bit path when needed
- Added `divfPos()` variant that skips the sign check on denominator

---

## 6. Incremental R0floor and r0 in SumByLin

**What:** The original computed `divf(R0num, denm)` (a 128-bit division) every iteration
of the main m-loop. Replaced with incremental remainder tracking.

**Where:** `SumByLin.h`: `sumByLin()` main loop

**Details:**
- Precomputes `xDivDenm` and `xModDenm` once
- Tracks `R0floor` and `R0rem` incrementally (subtract + conditional add)
- Similarly tracks `r0` via `RSub` and `r0Raw`
- `betanum` computed directly from `RSub` and correction term
- Eliminates 2 expensive 128-bit divisions per m-iteration in the hot loop

---

## 7. 64-bit Fast Path for divfc in SumByLin Hot Loop

**What:** The `divfc(betanum*ncirc, delnum*mcirc, ...)` call in the inner
loop was replaced with a pre-checked 64-bit fast path when both dividend and
divisor fit in `long`.

**Where:** `SumByLin.h`: `sumByLin()`, divfc fast-path block

**Details:**
- Precomputes `absDelnumMcirc` and bounds-checks `maxDividend` once before the loop
- If 64-bit: uses `long` division (~8x faster than `__divti3`)
- If 128-bit: uses `divfcPos` (skips sign check since abs value precomputed)
- Handles the sign flip (`delnumMcircNeg`) by negating the dividend

---

## 8. Incremental LinearSum (Division-Free Inner Loops)

**What:** Original `linearSum` called `divf(unum, uden)` for every m and n iteration.
Replaced with incremental quotient/remainder tracking — no divisions in the loops at all.

**Where:** `SumByLin.h`: `linearSum()`

**Details:**
- Precomputes `xdiv = x/uden`, `xmod = x - xdiv*uden`
- Initial quotient `qf = unum/uden`, remainder `qr = unum - qf*uden`
- Each iteration: `qr -= xmod; qf -= xdiv; if(qr < 0) { qr += uden; qf--; }`
- Only accumulates when `f[m] != 0` or `g[n] != 0`

---

## 9. flCong64 — 64-bit Floor Congruence

**What:** `sumInter` is called millions of times and uses `FlCong(n, r, q)` which does
a 128-bit mod. Added `flCong64` that takes a pre-reduced `rModQ` (computed once per
`sumInter` call) and uses pure 64-bit arithmetic.

**Where:** `SumByLin.h`: `flCong64()`, `sumInter()`

**Details:**
- `sumInter` reduces `r mod q` once (128-bit `modl`), then calls `flCong64` twice
- Also added a `q == 1` fast path that skips all mod arithmetic

---

## 10. FacToSumMu Branchless Unrolling

**What:** The original `FacToSumMu` was purely recursive (`subFacToSumMu`). Replaced with
specialized branchless code for nprimes = 1, 2, 3, 4, 5, with recursion fallback for 6+.

**Where:** `Factorization.h`: `FacToSumMu()`

**Details:**
- nprimes=1: single comparison `(p1 > ua)`
- nprimes=2: 4 branchless comparisons combined with bitwise AND
- nprimes=3: 8 terms, all branchless
- nprimes=4: 8 terms with 3-way AND gates
- nprimes=5: uses `sub2Primes()` helper for the final 2 primes (4 comparisons each)
- nprimes>=6: unrolls first 3 levels, then falls back to `subFacToSumMu` for remainder
- Extracted `extractPrime()` inline helper for readability

---

## 11. SegmentedMobiusSieve Integration in DDSum

**What:** The original `DDSum` allocated per-block `mux`/`muy` arrays inside the OpenMP
parallel region, calling `fillmublock` per tile. Replaced with a single upfront sieve of
the full [A, Ap) and [B, Bp) ranges using `SegmentedMobiusSieve`.

**Where:** `MertensHT.cpp`: `sieveMu()` helper + `ddSum()` / `ddSumSerial()`

**Details:**
- Computes mu for the entire A-range and B-range once before the parallel loop
- Uses `SegmentedMobiusSieveCore::sieve()` + indexed access
- Avoids repeated malloc/free of small `mux`/`muy` blocks inside the parallel region
- If A==B and Ap==Bp, shares the same buffer (no double allocation)
- Eliminates per-tile `isprime` dependency (sieve has its own prime list)

---

## 12. LargeFree Restructuring (Work-Item Parallelism)

**What:** The original `LargeFree` had nested while-loops calling `DDSum` sequentially,
with OpenMP parallelism only *inside* each DDSum call. Restructured into:
1. Serial enumeration phase collecting work items
2. Parallel dispatch of many small items via `omp parallel for schedule(dynamic)`
3. Sequential processing of few large brute-force items (which use internal parallelism)

**Where:** `MertensHT.cpp`: `LFWorkItem` struct + `largeFree()`

**Details:**
- `LFWorkItem` holds A, Ap, B, Bp, Delta, flag, a, b, multiplier
- `ddSumSerial()`: serial version of ddSum for outer parallelism
- DoubleSum items run with outer-level parallelism (many small, embarrassingly parallel)
- Brute items run sequentially with internal OpenMP (few, large)
- Added timing breakdown output for DoubleSum vs Brute phases (when `--profile`)

---

## 13. SArr Buffer Reuse

**What:** Original `SArr` allocated and freed `fun`, `sqfprod`, `offset` arrays on every
call. Moved allocation to `largeNonFree` (once) and passed buffers in.

**Where:** `MertensHT.cpp`: `sArr()` signature takes pre-allocated buffers

---

## 14. calloc to malloc Conversions

**What:** Several arrays that were immediately overwritten used `calloc` (which zeros memory).
Replaced with `malloc` to avoid the unnecessary zeroing cost.

**Where:**
- `SumByLin.h`: G, rho, sigma arrays in `sumByLin()`
- `MertensHT.cpp`: mup, S arrays in `largeNonFree()`
- `Factorization.h`: tabla in `fillMuBlock()` and `fillFactBlock()`
- `MertensHT.cpp`: mu array in `bruteM()`

---

## 15. short to Int8 for Mobius Values

**What:** All arrays storing mu(n) values changed from `short` (2 bytes) to `Int8` (1 byte).
Values are only {-1, 0, 1}, so this halves memory footprint and improves cache utilization.

**Where:** All function signatures throughout all files.

---

## 16. Native Integer Root Functions (GMP Removal from Hot Paths)

**What:** Original used GMP (`mpz_class`, `mpz_root`, `sqrt(mpzify(...))`) for computing
integer square roots, cube roots, and 4th roots. Replaced with native `__int128`/`long`
functions that avoid GMP heap allocation overhead.

**Where:** `IntegerMath.h`: `isqrtLong()`, `icbrtLong()`, `iqrtLong()`, `i5rtLong()`, `isqrt128()`, `sqrt64()`

**Details:**
- `isqrtLong(long n)`: native `sqrtl` + Newton correction
- `icbrtLong(Int128 n)`: native `cbrtl` + correction
- `iqrtLong(Int128 n)`: native double-sqrt + correction
- `i5rtLong(Int128 n)`: native fifth root, added last — it replaced the final
  GMP call (the `v` split point), removing the GMP dependency entirely
- `isqrt128(Int128 n)`: dispatches to `isqrtLong` for small values, full Newton for large
- `sqrt64()`: direct 64-bit square root
- Removed GMP usage from `largeFree`, `largeNonFree`, `bruteM`, and `MertensHT()`

**Background:** the hot-path `sqrt128` in the original's `sqroot.h` already
implements the approach Helfgott asked about on
[scicomp.SE](https://scicomp.stackexchange.com/questions/36153/hack-for-using-hardware-to-take-square-roots-of-128-bit-numbers):
a single Karatsuba-style reduction (his INRIA reference) to one 64-bit
hardware square root. This item extends that idea to every root in the code,
adds the small-argument dispatch fast path, and replaces the single-step
corrections with while-loop fixups — which stay exact even where `long
double` is only 64 bits (as on Apple Silicon), settling the question left
open in that post.

---

## 17. sumInter Forced Inlining + q==1 Fast Path

**What:** Added `__attribute__((always_inline))` to `sumInter`, `quadIneqZ`,
`specialL2L1`, `special00`, `special0A` to ensure inlining at -O3.
Added early return for q==1 case (identity — no mod arithmetic needed).

**Where:** `SumByLin.h`: all major functions

---

## 18. specialL2L1 / special00 Signature Change

**What:** Changed parameter from `R0num, R0den` (requiring division inside) to pre-computed
`R0floor` (= `divf(R0num, R0den)`). This division is now done incrementally in the caller.

**Where:** `SumByLin.h`: `specialL2L1()`, `special00()` signatures

---

## 19. Command-Line Parameter Tuning

**What:** Made algorithm parameters configurable via named command-line flags instead of
positional arguments.

**Where:** `main.cpp` argument parsing + `MertensHT()` signature in `MertensHT.h`

**Details:**
- `--vdiv`: v divisor (default 3.0) — controls v = x^{2/5} / vdiv
- `--C`: crossover parameter (default 18, was 10) — controls thresholds in LargeFree
- `--D`: tile size (default 8) — tile size in DDSum
- `--ntasks`: task multiplier (default 32, was hardcoded 3) — parallelism granularity
- `--profile` / `-p`: enables timing breakdown output
- Removed the `ntasks *= 3` hardcoding; replaced with configurable multiplier

**Impact:** C=18 (vs original 10) was found to be optimal at the time. ntasks_mul=128 best at 10^16.

**Update:** the final record-machine sweeps for the paper settled on `--vdiv 2 --C 42 --D 16` (Section 10 of the paper), which are now the shipped defaults.

---

## 20. mpzify Static Local

**What:** The `twp32` constant in `mpzify()` was recomputed on every call. Made `static`.

**Where:** `IntegerMath.h`: `mpzify()`

---

## 21. divfc Branchless Remainder Check

**What:** Original used `a%b ? 1 : 0` which computes a second division. Replaced with
`(Int128)(a != fl*b)` which reuses the already-computed quotient.

**Where:** `IntegerMath.h`: `divfc()`, `divfcPos()`

---

## 22. Profiling/Timing Infrastructure

**What:** Added detailed per-phase timing output, gated by `--profile` flag.

**Where:** `MertensHT.cpp`:
- `MertensHT()`: BruteM / LargeFree / LargeNonFree breakdown
- `largeFree()`: DoubleSum vs Brute breakdown
- `largeNonFree()`: fillMuBlock vs SArr breakdown + SArr sub-breakdown
- `sArr()`: fillFactBlock / FacToSumMu / offset / fixup phases

---

## 23. BruteDoubleSum mu-Zero Compaction

**What:** The inner n-loop of `bruteDoubleSum` performed a division for every n,
including the ~39% of n with mu(n) = 0 whose terms contribute nothing. The
nonzero (n, mu(n)) pairs are now compacted into dense arrays once per
ddSum/ddSumSerial call and shared across the entire m-loop, eliminating those
divisions with no per-term branch.

**Where:** `MertensHT.cpp`: `bruteDoubleSum()` signature (takes `nzN`/`nzG`/`nnz`),
new `compactMu()` helper, hoisted compaction in `ddSum()` and `ddSumSerial()`.

**Details:**
- Compaction is O(lenB) and amortizes over all m-chunks of the call
- Both the 64-bit and 128-bit inner paths use the compact list; the win is
  larger in the 128-bit region (m < x/2^63, software `__divti3` per term),
  which only exists for x > 9.2e18
- A branch (`if (g[n])`) would NOT have worked: ~39% mispredict-prone
  skips cost more than the saved divisions. Compaction removes the work
  without adding a branch.

---

## 24. SArr Fixup Elimination + Int64 Task-Local S

**What:** SArr previously stored globally-offset Int128 prefix values, requiring
a full parallel "fixup" pass adding per-task offsets to every entry (16 bytes
RMW x bigDelta entries per rebuild). But the consumer (the LargeNonFree n-loop)
performs only ~sqrt(x) lookups against ~x^{3/5} stored entries. S now holds
task-local Int64 prefix sums and the consumer reconstructs the global value as
`offset[idx / (Delta+1)] + S[idx]` — one division per lookup instead of a full
sweep per rebuild.

**Where:** `MertensHT.cpp`: `sArr()` (fixup pass deleted, `offset[ntasks]` added
as the global end-of-block value for the next rebuild's S0), `largeNonFree()`
(S allocation halved to Int64, consumer lookup, rebuild S0 plumbing).

**Details:**
- Int64 safety: task-local prefixes are bounded by max|FacToSumMu| * (Delta+1)
  (~2^omega * Delta), far below 2^63; the global offsets stay Int128
- Halves S memory: at 10^22 the SArr arrays drop from ~228 GB to ~182 GB,
  and the bigDelta memory cap admits 25% more parallel width (cap divisor
  updated 40 -> 32 bytes/entry; stale "~8 GB" comment corrected to ~400 GB)
- Also halves the store traffic in the FacToSumMu hot loop (16B -> 8B per r)

---

## 25. CPU Phase Concurrency (optional, compile-time, off by default)

**What:** `LargeFree` and `LargeNonFree` are independent phases (combined only at
the end as `2*BruteM - LNF - LF`). Optionally run them on two threads so each
fills the other's idle cores during load-imbalance and mildly-serial stretches.
Unlike items 1-24, this is **off by default** and is a *compile-time* switch, so
the default build is byte-identical to before and carries no extra branch.

**Where:** `MertensHT.cpp`: `MertensHT()` (`#if MHT_CPU_OVERLAP` block — a
`std::thread` runs `largeFree` concurrently with `largeNonFree`); `Makefile` flag
`CPU_OVERLAP` (`-DMHT_CPU_OVERLAP`).

**Build:** `make CPU_OVERLAP=1`. A plain `make` leaves it off.

**Details:**
- Pure CPU-utilization gain, no algorithm change: running the two phases at once
  keeps cores busier than running them back-to-back.
- Trades peak memory for time: both working sets (LNF's SArr arrays and LF's
  sieve) are live simultaneously, so peak RAM rises. Validate headroom before
  using it at memory-bound scales.
- Machine-dependent: helps most where spare cores absorb the 2x OpenMP
  oversubscription.
- Under `--profile`, per-phase timing is suppressed (not meaningful while the
  phases overlap) in favor of a single combined region time.

---

## 26. Thread-Local Scratch Buffers in SumByLin

**What:** `sumByLin()` previously allocated and freed three temporary arrays on
every call:

- `G`
- `rho`
- `sigma`

These are now thread-local `std::vector<long>` buffers that grow as needed and
are reused by subsequent `sumByLin()` calls on the same OpenMP worker.

**Where:** `SumByLin.h`: `sumByLin()` temporary array allocation.

**Details:**
- Removes three `malloc/free` pairs per `sumByLin()` call.
- At `10^16`, profiling showed about 9.26 million `sumByLin()` calls, so this
  avoids tens of millions of allocator calls inside `LargeFree::DoubleSum`.
- Behavior-preserving; buffers are overwritten by the existing `sumTable()`
  logic before use.

---

## 27. SpecialL2L1 Zero and Cancellation Skips

**What:** The `specialL2L1()` correction frequently computes contributions that
are known to be zero or cancel algebraically. The implementation now detects
these cases before doing the expensive interval sums, and in one case before
doing the quadratic root computation.

**Where:** `SumByLin.h`: `specialL2L1()`.

**Details:**

1. First contribution: compute the cheap base interval before the first
   `quadIneqZ()` call. If the base interval is empty, the contribution is zero,
   so skip one `quadIneqZ()` and two `sumInter()` calls.
2. First contribution: if the clipped quadratic interval equals the base
   interval, skip both interval sums because
   `sumInter(J) - sumInter(J) = 0`.
3. Second contribution: if the clipped interval covers the full base interval,
   skip the contribution because it is
   `sumInter(full) - sumInter(full) = 0`.
4. Second contribution: detect the common full-coverage case before
   `quadIneqZ()` by evaluating the quadratic at the two endpoints of the full
   interval. Since the solution set is contiguous, both endpoints inside imply
   full coverage and avoid the root computation entirely.

---

## 28. Special0A Empty-Interval Skip

**What:** Avoid calling `sumInter()` from `special0A()` when the interval has
already been proven empty, and delay/reuse the 128-bit residue product needed
by nonempty interval sums.

**Where:** `SumByLin.h`.

**Details:** After the `specialL2L1()` skips, profiling showed `special0A()`
was still about `18%` of correction-loop time at `10^16`. Its final interval
was empty about `28%` of the time, but the code still entered `sumInter()`,
where it paid the timing/profiling wrapper, branch path, and return machinery
to discover the same empty interval.

The change adds a local `sumInterNonEmpty()` wrapper and uses it only in
`special0A()`, including the two split branches. The final empty branch returns
before computing `-r0 * ainv`, and the split branches compute that residue once
instead of once per interval.

---

## 29. Guarded 256-bit Fallbacks + SumByLin Residue Reduction (High Decades)

**What:** Removes the arithmetic obstruction noted in Helfgott-Thompson for
inputs beyond $10^{23}$. Two changes: (1) the relevant `SumByLin` residue is
reduced before the final multiplication by $n_\circ$, which eliminates the
overflow that previously capped runs at $10^{23}$; (2) the few `SumByLin`
products and divisions that can still exceed signed 128-bit range at
$10^{24}$-$10^{25}$ fall back to guarded 256-bit arithmetic (`int256_t`).

**Where:** `SumByLin.h`: `quadEvalFitsI128()` / `quadEvalWide()`,
`mulFitsI128()`, the `EnableWide` template parameter threaded through
`quadIneqZ()`, `quadIntervalCovers()`, `specialL2L1()`, `special00()`.

**Details:**
- The wide path is compiled in only where preconditions can fail; the ordinary
  64- and 128-bit paths remain in place whenever their bounds are satisfied,
  so low decades pay only a cheap fits-in-128 guard.
- `quadIntervalCovers()` evaluates the quadratic at the two endpoints of the
  summation window (convexity makes this sufficient); when the solution
  interval covers the window, the root computation is skipped entirely
  (see item 27).
- Together these make inputs beyond $10^{23}$ attainable without further
  modification. Projected timings at $10^{24}$/$10^{25}$ appear in the paper
  (Section 10); no full runs have been executed at that scale.
