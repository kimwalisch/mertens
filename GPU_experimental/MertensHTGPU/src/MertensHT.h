#pragma once

// ============================================================================
// MertensHT.h — Compute the Mertens function M(n) via the Helfgott--Thompson
//                O(n^{3/5}) combinatorial algorithm.
//
// Single entry point: MertensHT(x). Tested through 10^23 (CPU path); larger
// inputs are enabled by guarded 256-bit fallbacks (see the paper, Section 10).
//
// Internally decomposes M(x) = 2*BruteM(u) - LargeNonFree(x,v,u) - LargeFree(x,v):
//   1. BruteM: direct sieve of mu over [1, u] where u = floor(sqrt(x))
//   2. LargeFree: double sums over squarefree pairs via Diophantine approximation
//   3. LargeNonFree: factorized partial sums via SArr with quotient prediction
//
// Tunable parameters: vdiv, C, D, ntasksMul (see MertensHT() signature).
// ============================================================================

#include "types.h"


Int64 MertensHT(
    Int128 x,
    bool profile = false,
    double vdiv = 2.0,
    Int64 paramC = 42,
    Int64 paramD = 16,
    Int64 ntasksMul = 32,
    Int64 sarrCapBytes = 12000000000LL   // SArr memory cap; --sarr-cap on the CLI
);
