// bench_mobius_regimes.cpp — Time the Mobius sieve in its distinct regimes.
//
// Each regime starts the sieve window at a different lo, which controls how
// far up the prime phases reach (sqrt(hi)):
//   lo = 1      : small + a thin band of medium primes (standalone use)
//   lo = 1e12   : full medium-prime range, no large primes
//   lo = 1e14   : large primes via the bucket scheduler (~6.6e5 primes)
//   lo = 4e16   : deep large-prime regime (~1.1e7 primes; scheduler
//                 construction dominates — the case for cross-segment
//                 bucket persistence)
//
// Usage:
//   ./bench_mobius_regimes                   — run all four regimes
//   ./bench_mobius_regimes <lo> [len] [seg]  — run one custom range
//
// Set CHECK=1 in the environment to also print a position-weighted checksum
// (for bit-exactness comparison across code versions; slower).

#include "SegmentedMobiusSieve.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

static void runRange(UInt64 lo, UInt64 len, UInt64 seg) {
    const UInt64 hi = lo + len - 1;
    const UInt32 sqrtHi = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                           (Int64)std::round(std::sqrt((double)hi)));
    auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtHi);

    SegmentedMobiusSieveCore core(seg);
    core.sieve(lo, std::min(lo + seg - 1, hi), primes);  // warm-up

    const bool doCheck = std::getenv("CHECK") != nullptr;
    Int64 check = 0;
    double tCheck = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (UInt64 slo = lo; slo <= hi; slo += seg) {
        const UInt64 shi = std::min(slo + seg - 1, hi);
        core.sieve(slo, shi, primes);
        if (doCheck) {
            auto c0 = std::chrono::high_resolution_clock::now();
            const Int8* m = core.data();
            Int64 c = 0;
            #pragma omp parallel for schedule(static) reduction(+:c)
            for (UInt64 k = slo; k <= shi; ++k)
                c += (Int64)m[k - slo] * (Int64)(k % 65521 + 1);
            check += c;
            tCheck += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - c0).count();
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count() - tCheck;

    std::printf("lo=%-8.3g len=%.3g seg=%.3g sqrtHi=%-9u : %8.4f s  (%.4f ns/elem)",
                (double)lo, (double)len, (double)seg, sqrtHi,
                wall, wall * 1e9 / (double)len);
    if (doCheck) std::printf("  CHECKSUM=%lld", (long long)check);
    std::printf("\n");
}

int main(int argc, char* argv[]) {
    std::printf("threads = %d\n", omp_get_max_threads());

    if (argc >= 2) {
        const UInt64 lo  = (UInt64)std::atoll(argv[1]);
        const UInt64 len = argc >= 3 ? (UInt64)std::atoll(argv[2]) : 2000000000ULL;
        const UInt64 seg = argc >= 4 ? (UInt64)std::atoll(argv[3]) : 110880000ULL;
        runRange(lo, len, seg);
        return 0;
    }

    const UInt64 len = 2000000000ULL, seg = 110880000ULL;
    runRange(1ULL,                 len, seg);
    runRange(1000000000000ULL,     len, seg);
    runRange(100000000000000ULL,   len, seg);
    runRange(40000000000000000ULL, len, seg);
    return 0;
}
