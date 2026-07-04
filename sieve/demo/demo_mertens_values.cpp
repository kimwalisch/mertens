// demo_mertens_values.cpp — MertensSieveValues(N) free function demo.
//
// Usage: ./demo_mertens_values <N>
//   Prints M(1)..M(min(N,20)) and M(N).

#include "SegmentedMertensSieve.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    const UInt64 N = (UInt64)std::atoll(argv[1]);

    std::printf("MertensSieveValues: N = %lld\n", (long long)N);

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Int32> M = MertensSieveValues(N);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    UInt64 show = std::min(N, (UInt64)20);
    for (UInt64 k = 1; k <= show; ++k) {
        std::printf("M(%lld) = %d\n", (long long)k, (int)M[k - 1]);
    }
    if (N > show) {
        std::printf("...\n");
        std::printf("M(%lld) = %d\n", (long long)N, (int)M[N - 1]);
    }

    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
