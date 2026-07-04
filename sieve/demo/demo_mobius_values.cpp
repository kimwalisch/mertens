// demo_mobius_values.cpp — MobiusSieveValues(N) free function demo.
//
// Usage: ./demo_mobius_values <N> [segment_size]
//   Prints mu(1)..mu(min(N,20)) and a summary.

#include "SegmentedMobiusSieve.h"
#include <algorithm>
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

    std::printf("MobiusSieveValues: N = %lld, segment = %lld\n",
                (long long)N, (long long)segSize);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Int8> mu = MobiusSieveValues(N, segSize);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    UInt64 show = std::min(N, (UInt64)20);
    for (UInt64 k = 1; k <= show; ++k) {
        std::printf("mu(%lld) = %d\n", (long long)k, (int)mu[k - 1]);
    }

    if (N > show) {
        // Sum to get M(N) as a check
        Int64 sum = 0;
        for (UInt64 k = 0; k < (UInt64)N; ++k) sum += mu[k];
        std::printf("...\nM(%lld) = %lld  (sum of mu)\n", (long long)N, (long long)sum);
    }

    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
