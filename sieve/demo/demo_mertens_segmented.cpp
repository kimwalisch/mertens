// demo_mertens_segmented.cpp — SegmentedMertensSieve iterator demo.
//
// Usage: ./demo_mertens_segmented <N> [segment_size]

#include "SegmentedMertensSieve.h"
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

    std::printf("SegmentedMertensSieve: N = %lld, segment = %lld\n",
                (long long)N, (long long)segSize);

    auto t0 = std::chrono::high_resolution_clock::now();

    SegmentedMertensSieve sieve(segSize);
    Int32 result = 0;

    while (sieve.next()) {
        if (sieve.hi() >= N) {
            result = sieve.getMertens(N);
            break;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("M(%lld) = %d\n", (long long)N, (int)result);
    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
