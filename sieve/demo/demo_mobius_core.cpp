// demo_mobius.cpp — Time the segmented Mobius sieve over [1, N].
//
// Usage: ./demo_mobius <N> [segment_size]
//   N            — sieve mu(1)..mu(N)
//   segment_size — elements per segment (default: 13860000, ~13.2M)

#include "SegmentedMobiusSieve.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N> [segment_size]\n", argv[0]);
        return 1;
    }

    const UInt64 N = (UInt64)std::atoll(argv[1]);
    const UInt64 segSize = argc >= 3 ? (UInt64)std::atoll(argv[2]) : 13860000ULL;

    const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                          (Int64)std::round(std::sqrt((double)N)));
    auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);

    SegmentedMobiusSieveCore sieve(segSize);

    std::printf("Mobius sieve: N = %lld, segment = %lld\n", (long long)N, (long long)segSize);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (UInt64 slo = 1; slo <= N; slo += segSize) {
        UInt64 shi = std::min(slo + segSize - 1, N);
        sieve.sieve(slo, shi, primes);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
