// demo_mobius_segmented.cpp — SegmentedMobiusSieve iterator demo.
//
// Usage: ./demo_mobius_segmented <N> [segment_size]

#include "SegmentedMobiusSieve.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N> [segment_size]\n", argv[0]);
        return 1;
    }

    const UInt64 N = (UInt64)std::atoll(argv[1]);
    const UInt64 segSize = argc >= 3 ? (UInt64)std::atoll(argv[2]) : 13860000ULL;

    std::printf("SegmentedMobiusSieve: N = %lld, segment = %lld\n",
                (long long)N, (long long)segSize);

    auto t0 = std::chrono::high_resolution_clock::now();

    SegmentedMobiusSieve sieve(segSize);
    Int64 sum = 0;

    while (sieve.next()) {
        const Int8* mu = sieve.getSegmentData();
        const UInt64 count = std::min(sieve.hi(), N) - sieve.lo() + 1;

        Int64 segSum = 0;
        #pragma omp parallel for reduction(+:segSum) schedule(static)
        for (UInt64 i = 0; i < count; ++i) {
            segSum += mu[i];
        }
        sum += segSum;

        if (sieve.hi() >= N) break;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("M(%lld) = %lld  (via summing mu)\n", (long long)N, (long long)sum);
    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
