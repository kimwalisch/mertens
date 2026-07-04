// bench_prefix.cpp — Measure the cost of the Mertens prefix-sum step and
// compare the current Stage A kernel against candidate optimizations.
//
// Part 1: split timing — Mobius sieve alone vs full Mertens sieveInPlace
//         over the same segments. The difference is the prefix-sum cost.
// Part 2: micro-benchmark of prefixSum variants on a real mu buffer:
//         V0 — faithful copy of the current implementation
//         V1 — V0 with the Mub pre-pass fused into the Stage A loop
//         V2 — V1 with int8-domain scan + vector-register carry
//
// Usage: ./bench_prefix [N] [segSize] [microLen]

#include "SegmentedMertensSieve.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Clock = std::chrono::high_resolution_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static constexpr int STRIDE_LOG = MertensSieveDetail::STRIDE_LOG;
static constexpr UInt64 STRIDE = MertensSieveDetail::STRIDE;

// ---------------------------------------------------------------------------
// Shared Stage B + C (identical across variants; copied from the header)
// ---------------------------------------------------------------------------
static void stageBC(Int64* M, Int64* intervalSums, Int64* suma,
                    const Int8* Mub, UInt64 numIntervals, Int64 MPrev) {
    #pragma omp parallel
    {
        const UInt32 tid  = omp_get_thread_num();
        const UInt32 nthr = omp_get_num_threads();
        Int64 local = 0;
        #pragma omp for schedule(static)
        for (UInt64 i = 0; i < numIntervals; ++i) {
            local += intervalSums[i];
            intervalSums[i] = local;
        }
        suma[tid] = local;
        #pragma omp barrier
        #pragma omp single
        {
            Int64 offset = 0;
            for (UInt32 t = 0; t < nthr; ++t) {
                const Int64 tmp = suma[t];
                suma[t] = offset;
                offset += tmp;
            }
        }
        #pragma omp barrier
        const Int64 offset = suma[tid];
        #pragma omp for schedule(static)
        for (UInt64 i = 0; i < numIntervals; ++i)
            intervalSums[i] += offset;
    }
    M[0] = MPrev + (Int64)Mub[0];
    #pragma omp parallel for schedule(static)
    for (Int64 b = 1; b < (Int64)numIntervals; ++b)
        M[b] = MPrev + intervalSums[b - 1] + (Int64)Mub[b];
}

// ---------------------------------------------------------------------------
// V0: faithful copy of the current prefixSum (in-place, R aliases Mu)
// ---------------------------------------------------------------------------
#if SIEVE_SIMD_NEON
static int16x8_t scan8_int16_inclusive(int16x8_t x) {
    const int16x8_t z = vdupq_n_s16(0);
    x = vaddq_s16(x, vextq_s16(z, x, 7));
    x = vaddq_s16(x, vextq_s16(z, x, 6));
    x = vaddq_s16(x, vextq_s16(z, x, 4));
    return x;
}
static int8x16_t prefix16_i8_inclusive(int8x16_t v) {
    int16x8_t lo = vmovl_s8(vget_low_s8(v));
    int16x8_t hi = vmovl_s8(vget_high_s8(v));
    lo = scan8_int16_inclusive(lo);
    hi = scan8_int16_inclusive(hi);
    const int16_t sum_lo = vgetq_lane_s16(lo, 7);
    hi = vaddq_s16(hi, vdupq_n_s16(sum_lo));
    const int8x8_t lo8 = vmovn_s16(lo);
    const int8x8_t hi8 = vmovn_s16(hi);
    return vcombine_s8(lo8, hi8);
}
#endif

static void prefixSumV0(Int64* M, Int8* R, Int8* Mu, const UInt64 len,
                        const Int64 MPrev, Int64* intervalSums, Int64* suma,
                        Int8* Mub, double* tStageA) {
    const UInt64 fullIntervals = len >> STRIDE_LOG;
    const UInt64 rem = len & (STRIDE - 1);
    const UInt64 numIntervals = fullIntervals + (rem ? 1 : 0);

    auto tA0 = Clock::now();
    #pragma omp parallel for
    for (UInt64 b = 0; b < numIntervals; ++b)
        Mub[b] = Mu[b * STRIDE];

    #pragma omp parallel for schedule(static)
    for (UInt64 b = 0; b < numIntervals; ++b) {
        const UInt64 start = b * STRIDE;
        const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
        const Int8 MuStart = Mu[start];
        Int8 local = (Int8)(-MuStart);
        UInt64 k = start;
#if SIEVE_SIMD_NEON
        for (; k + 16 <= end; k += 16) {
            const int8x16_t v = vld1q_s8((const Int8*)(Mu + k));
            const int8x16_t ps = prefix16_i8_inclusive(v);
            const int8x16_t carryv = vdupq_n_s8((Int8)local);
            const int8x16_t out = vaddq_s8(ps, carryv);
            vst1q_s8((Int8*)(R + k), out);
            local = (Int8)vgetq_lane_s8(out, 15);
        }
#endif
        for (; k < end; ++k) {
            local = (Int8)(local + Mu[k]);
            R[k] = local;
        }
        local = (Int8)(local + MuStart);
        intervalSums[b] = (Int64)local;
    }
    auto tA1 = Clock::now();
    if (tStageA) *tStageA = secs(tA0, tA1);

    stageBC(M, intervalSums, suma, Mub, numIntervals, MPrev);
}

// ---------------------------------------------------------------------------
// V1: Mub pre-pass fused into the Stage A loop (no separate strided pass)
// ---------------------------------------------------------------------------
static void prefixSumV1(Int64* M, Int8* R, Int8* Mu, const UInt64 len,
                        const Int64 MPrev, Int64* intervalSums, Int64* suma,
                        Int8* Mub, double* tStageA) {
    const UInt64 fullIntervals = len >> STRIDE_LOG;
    const UInt64 rem = len & (STRIDE - 1);
    const UInt64 numIntervals = fullIntervals + (rem ? 1 : 0);

    auto tA0 = Clock::now();
    #pragma omp parallel for schedule(static)
    for (UInt64 b = 0; b < numIntervals; ++b) {
        const UInt64 start = b * STRIDE;
        const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
        const Int8 MuStart = Mu[start];
        Mub[b] = MuStart;
        Int8 local = (Int8)(-MuStart);
        UInt64 k = start;
#if SIEVE_SIMD_NEON
        for (; k + 16 <= end; k += 16) {
            const int8x16_t v = vld1q_s8((const Int8*)(Mu + k));
            const int8x16_t ps = prefix16_i8_inclusive(v);
            const int8x16_t carryv = vdupq_n_s8((Int8)local);
            const int8x16_t out = vaddq_s8(ps, carryv);
            vst1q_s8((Int8*)(R + k), out);
            local = (Int8)vgetq_lane_s8(out, 15);
        }
#endif
        for (; k < end; ++k) {
            local = (Int8)(local + Mu[k]);
            R[k] = local;
        }
        local = (Int8)(local + MuStart);
        intervalSums[b] = (Int64)local;
    }
    auto tA1 = Clock::now();
    if (tStageA) *tStageA = secs(tA0, tA1);

    stageBC(M, intervalSums, suma, Mub, numIntervals, MPrev);
}

// ---------------------------------------------------------------------------
// V2: V1 + int8-domain scan + carry kept in a vector register.
// Safe because |R| <= 96 within a 256-interval and intra-vector partial sums
// add at most +/-16, so all intermediates fit in int8.
// ---------------------------------------------------------------------------
#if SIEVE_SIMD_NEON
static inline int8x16_t scan16_i8(int8x16_t x, int8x16_t z) {
    x = vaddq_s8(x, vextq_s8(z, x, 15));
    x = vaddq_s8(x, vextq_s8(z, x, 14));
    x = vaddq_s8(x, vextq_s8(z, x, 12));
    x = vaddq_s8(x, vextq_s8(z, x, 8));
    return x;
}
#endif

static void prefixSumV2(Int64* M, Int8* R, Int8* Mu, const UInt64 len,
                        const Int64 MPrev, Int64* intervalSums, Int64* suma,
                        Int8* Mub, double* tStageA) {
    const UInt64 fullIntervals = len >> STRIDE_LOG;
    const UInt64 rem = len & (STRIDE - 1);
    const UInt64 numIntervals = fullIntervals + (rem ? 1 : 0);

    auto tA0 = Clock::now();
    #pragma omp parallel for schedule(static)
    for (UInt64 b = 0; b < numIntervals; ++b) {
        const UInt64 start = b * STRIDE;
        const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
        const Int8 MuStart = Mu[start];
        Mub[b] = MuStart;
        Int8 local = (Int8)(-MuStart);
        UInt64 k = start;
#if SIEVE_SIMD_NEON
        const int8x16_t z = vdupq_n_s8(0);
        int8x16_t carry = vdupq_n_s8((Int8)local);
        for (; k + 16 <= end; k += 16) {
            const int8x16_t v = vld1q_s8((const Int8*)(Mu + k));
            const int8x16_t ps = scan16_i8(v, z);
            const int8x16_t out = vaddq_s8(ps, carry);
            vst1q_s8((Int8*)(R + k), out);
            carry = vdupq_laneq_s8(out, 15);
        }
        local = (Int8)vgetq_lane_s8(carry, 0);
#endif
        for (; k < end; ++k) {
            local = (Int8)(local + Mu[k]);
            R[k] = local;
        }
        local = (Int8)(local + MuStart);
        intervalSums[b] = (Int64)local;
    }
    auto tA1 = Clock::now();
    if (tStageA) *tStageA = secs(tA0, tA1);

    stageBC(M, intervalSums, suma, Mub, numIntervals, MPrev);
}

// ---------------------------------------------------------------------------
// V3: V2 + 64-byte unroll, scans computed independently then carries chained
// ---------------------------------------------------------------------------
static void prefixSumV3(Int64* M, Int8* R, Int8* Mu, const UInt64 len,
                        const Int64 MPrev, Int64* intervalSums, Int64* suma,
                        Int8* Mub, double* tStageA) {
    const UInt64 fullIntervals = len >> STRIDE_LOG;
    const UInt64 rem = len & (STRIDE - 1);
    const UInt64 numIntervals = fullIntervals + (rem ? 1 : 0);

    auto tA0 = Clock::now();
    #pragma omp parallel for schedule(static)
    for (UInt64 b = 0; b < numIntervals; ++b) {
        const UInt64 start = b * STRIDE;
        const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
        const Int8 MuStart = Mu[start];
        Mub[b] = MuStart;
        Int8 local = (Int8)(-MuStart);
        UInt64 k = start;
#if SIEVE_SIMD_NEON
        const int8x16_t z = vdupq_n_s8(0);
        int8x16_t carry = vdupq_n_s8((Int8)local);
        for (; k + 64 <= end; k += 64) {
            int8x16_t s0 = scan16_i8(vld1q_s8(Mu + k),      z);
            int8x16_t s1 = scan16_i8(vld1q_s8(Mu + k + 16), z);
            int8x16_t s2 = scan16_i8(vld1q_s8(Mu + k + 32), z);
            int8x16_t s3 = scan16_i8(vld1q_s8(Mu + k + 48), z);
            const int8x16_t o0 = vaddq_s8(s0, carry);
            const int8x16_t c0 = vdupq_laneq_s8(o0, 15);
            const int8x16_t o1 = vaddq_s8(s1, c0);
            const int8x16_t c1 = vdupq_laneq_s8(o1, 15);
            const int8x16_t o2 = vaddq_s8(s2, c1);
            const int8x16_t c2 = vdupq_laneq_s8(o2, 15);
            const int8x16_t o3 = vaddq_s8(s3, c2);
            carry = vdupq_laneq_s8(o3, 15);
            vst1q_s8(R + k,      o0);
            vst1q_s8(R + k + 16, o1);
            vst1q_s8(R + k + 32, o2);
            vst1q_s8(R + k + 48, o3);
        }
        for (; k + 16 <= end; k += 16) {
            const int8x16_t ps = scan16_i8(vld1q_s8(Mu + k), z);
            const int8x16_t out = vaddq_s8(ps, carry);
            vst1q_s8(R + k, out);
            carry = vdupq_laneq_s8(out, 15);
        }
        local = (Int8)vgetq_lane_s8(carry, 0);
#endif
        for (; k < end; ++k) {
            local = (Int8)(local + Mu[k]);
            R[k] = local;
        }
        local = (Int8)(local + MuStart);
        intervalSums[b] = (Int64)local;
    }
    auto tA1 = Clock::now();
    if (tStageA) *tStageA = secs(tA0, tA1);

    stageBC(M, intervalSums, suma, Mub, numIntervals, MPrev);
}

// ---------------------------------------------------------------------------
// Scalar reference for correctness checking
// ---------------------------------------------------------------------------
static void reference(const Int8* Mu, UInt64 len, Int64 MPrev,
                      std::vector<Int64>& Mref, std::vector<Int8>& Rref) {
    const UInt64 numIntervals = (len + STRIDE - 1) >> STRIDE_LOG;
    Mref.resize(numIntervals);
    Rref.resize(len);
    Int64 running = MPrev;
    for (UInt64 b = 0; b < numIntervals; ++b) {
        const UInt64 start = b * STRIDE;
        const UInt64 end = (start + STRIDE <= len ? start + STRIDE : len);
        Mref[b] = running + Mu[start];   // M at the coarse sample position
        for (UInt64 k = start; k < end; ++k) {
            running += Mu[k];
            Rref[k] = (Int8)(running - Mref[b]);
        }
    }
}

int main(int argc, char* argv[]) {
    const UInt64 N        = argc >= 2 ? (UInt64)std::atoll(argv[1]) : 2000000000ULL;
    const UInt64 segSize  = argc >= 3 ? (UInt64)std::atoll(argv[2]) : 110880000ULL;
    const UInt64 microLen = argc >= 4 ? (UInt64)std::atoll(argv[3]) : 277200000ULL;

    std::printf("threads = %d\n\n", omp_get_max_threads());

    // ---- Part 1: split timing via public API ----
    {
        const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                              (Int64)std::round(std::sqrt((double)N)));
        auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);

        // Mobius only
        SegmentedMobiusSieveCore mob(segSize);
        // warm-up
        mob.sieve(1, std::min(segSize, N), primes);
        auto t0 = Clock::now();
        for (UInt64 slo = 1; slo <= N; slo += segSize) {
            UInt64 shi = std::min(slo + segSize - 1, N);
            mob.sieve(slo, shi, primes);
        }
        auto t1 = Clock::now();
        const double tMob = secs(t0, t1);

        // Mobius + prefix sum (Compressed, in-place — the next()/Loop-2 path)
        const UInt64 segIntervals = (segSize + STRIDE - 1) >> STRIDE_LOG;
        SegmentedMertensSieveCore mer(segSize);
        std::vector<Int64> M(segIntervals);
        Int64 MPrev = 0;
        mer.sieveInPlace(1, std::min(segSize, N), MPrev, M.data(), primes); // warm-up
        MPrev = 0;
        t0 = Clock::now();
        for (UInt64 slo = 1; slo <= N; slo += segSize) {
            UInt64 shi = std::min(slo + segSize - 1, N);
            mer.sieveInPlace(slo, shi, MPrev, M.data(), primes);
            MPrev = mer.getMertens(M.data(), shi);
        }
        t1 = Clock::now();
        const double tMer = secs(t0, t1);

        std::printf("Part 1: N = %llu, segSize = %llu\n",
                    (unsigned long long)N, (unsigned long long)segSize);
        std::printf("  Mobius sieve only:        %.4f s\n", tMob);
        std::printf("  Mertens (sieve + prefix): %.4f s   [M(N) = %lld]\n",
                    tMer, (long long)MPrev);
        std::printf("  => prefix sum step:       %.4f s  (%.1f%% of Mertens total)\n\n",
                    tMer - tMob, 100.0 * (tMer - tMob) / tMer);
    }

    // ---- Part 1b: per-segment instrumentation (sieve vs prefix, V0 vs V2) ----
    {
        const UInt32 sqrtN = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                              (Int64)std::round(std::sqrt((double)N)));
        auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtN);
        const UInt64 segIntervals = (segSize + STRIDE - 1) >> STRIDE_LOG;

        SegmentedMobiusSieveCore mob(segSize);
        std::vector<Int64> M(segIntervals), intervalSums(segIntervals);
        std::vector<Int64> suma(omp_get_max_threads());
        std::vector<Int8> Mub(segIntervals);

        for (int variant = 0; variant < 2; ++variant) {
            mob.sieve(1, std::min(segSize, N), primes); // warm-up
            double tSieve = 0, tPrefix = 0;
            Int64 MPrev = 0;
            for (UInt64 slo = 1; slo <= N; slo += segSize) {
                UInt64 shi = std::min(slo + segSize - 1, N);
                auto t0 = Clock::now();
                mob.sieve(slo, shi, primes);
                auto t1 = Clock::now();
                Int8* mu = mob.data();
                if (variant == 0)
                    prefixSumV0(M.data(), mu, mu, shi - slo + 1, MPrev,
                                intervalSums.data(), suma.data(), Mub.data(), nullptr);
                else
                    prefixSumV2(M.data(), mu, mu, shi - slo + 1, MPrev,
                                intervalSums.data(), suma.data(), Mub.data(), nullptr);
                auto t2 = Clock::now();
                tSieve += secs(t0, t1);
                tPrefix += secs(t1, t2);
                const UInt64 off = shi - slo;
                MPrev = M[off >> STRIDE_LOG] + mu[off];
            }
            std::printf("Part 1b (%s): sieve %.4f s, prefix %.4f s (%.1f%%)  [M(N) = %lld]\n",
                        variant == 0 ? "V0" : "V2", tSieve, tPrefix,
                        100.0 * tPrefix / (tSieve + tPrefix), (long long)MPrev);
        }
        std::printf("\n");
    }

    // ---- Part 2: micro-benchmark of prefixSum variants ----
    {
        const UInt64 len = microLen;
        const UInt64 numIntervals = (len + STRIDE - 1) >> STRIDE_LOG;

        std::printf("Part 2: micro-benchmark, len = %llu (%.0f MB)\n",
                    (unsigned long long)len, (double)len / 1e6);

        // Real mu values
        std::vector<Int8> muOrig(len);
        {
            const UInt32 sqrtL = (UInt32)std::max((Int64)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND,
                                                  (Int64)std::round(std::sqrt((double)len)));
            auto primes = SegmentedMobiusSieveCore::primesUpTo(sqrtL);
            SegmentedMobiusSieveCore mob(len);
            mob.sieve(1, len, primes);
            std::memcpy(muOrig.data(), mob.data(), len);
        }

        std::vector<Int8> buf(len);
        std::vector<Int64> M(numIntervals), intervalSums(numIntervals);
        std::vector<Int64> suma(omp_get_max_threads());
        std::vector<Int8> Mub(numIntervals);
        const Int64 MPrev = 12345;

        std::vector<Int64> Mref;
        std::vector<Int8> Rref;
        reference(muOrig.data(), len, MPrev, Mref, Rref);

        using Fn = void(*)(Int64*, Int8*, Int8*, UInt64, Int64,
                           Int64*, Int64*, Int8*, double*);
        struct Variant { const char* name; Fn fn; };
        Variant variants[] = {
            {"V0 current        ", prefixSumV0},
            {"V1 fused Mub      ", prefixSumV1},
            {"V2 int8+vec carry ", prefixSumV2},
            {"V3 V2 + unroll x4 ", prefixSumV3},
        };

        const int reps = 7;
        for (auto& var : variants) {
            double best = 1e30, bestA = 1e30;
            bool ok = true;
            for (int r = 0; r < reps; ++r) {
                std::memcpy(buf.data(), muOrig.data(), len);  // restore (untimed)
                double tA = 0;
                auto t0 = Clock::now();
                var.fn(M.data(), buf.data(), buf.data(), len, MPrev,
                       intervalSums.data(), suma.data(), Mub.data(), &tA);
                auto t1 = Clock::now();
                best = std::min(best, secs(t0, t1));
                bestA = std::min(bestA, tA);
                if (r == 0) {
                    ok = std::memcmp(buf.data(), Rref.data(), len) == 0 &&
                         std::memcmp(M.data(), Mref.data(), numIntervals * sizeof(Int64)) == 0;
                }
            }
            std::printf("  %s total %.4f s (%.1f GB/s)   stageA %.4f s   %s\n",
                        var.name, best, (double)len / best / 1e9, bestA,
                        ok ? "OK" : "** MISMATCH **");
        }
    }
    return 0;
}
