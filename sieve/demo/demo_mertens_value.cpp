// demo_mertens_value.cpp — MertensSieve(N) free function demo.
//
// Usage: ./demo_mertens_value <N>

#include "SegmentedMertensSieve.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    const UInt64 N = (UInt64)std::atoll(argv[1]);

    std::printf("MertensSieve: N = %lld\n", (long long)N);

    auto t0 = std::chrono::high_resolution_clock::now();

    Int32 result = MertensSieve(N);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::printf("M(%lld) = %d\n", (long long)N, (int)result);
    std::printf("Time: %.6f s\n", elapsed);

    return 0;
}
