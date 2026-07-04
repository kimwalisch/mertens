#include "MertensHT.h"
#include "IntegerMath.h"
#include "Factorization.h"
#include "SumByLin.h"
#include "SegmentedMertensSieve.h"
#include "MertensHurst.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <sys/time.h>
#include <vector>

// Optional CPU phase concurrency (compile-time, default off). When built with
// -DMHT_CPU_OVERLAP=1, LargeFree and LargeNonFree — which are independent — run
// on two threads so each fills the other's idle cores. Compile-time so the
// default build has no extra branch and is byte-identical to before.
#if MHT_CPU_OVERLAP
#include <thread>
#endif

// ============================================================================
// Timing helpers
// ============================================================================

static inline void getDayTime(timeval& t) { gettimeofday(&t, NULL); }

static inline double getDuration(timeval& start, timeval& end) {
    return (Int64)(end.tv_sec) - (Int64)(start.tv_sec)
         + (end.tv_usec - start.tv_usec) / 1000000.0;
}

#define START_PROFILE() do { if (profile) getDayTime(tStart); } while(0)
#define END_PROFILE(t)  do { if (profile) { \
    getDayTime(tEnd); (t) += getDuration(tStart, tEnd); } } while(0)

namespace {

// ============================================================================
// BruteDoubleSum — direct O(mn) evaluation of the double sum
// ============================================================================

// The inner sum iterates a pre-compacted list of the nonzero mu(n) values
// (nzN[i] = n, nzG[i] = mu(n), built once per ddSum call). Among general
// integers ~39% have mu(n) = 0, and the same n-range is reused by every
// m-chunk, so compaction removes ~39% of the divisions — the loop's entire
// cost — with no per-term branch. (Contrast with MertensHurst's S2, where
// the mod-6 iteration already excludes most zeros and a branch would lose.)
static Int128 bruteDoubleSum(
    Int128 m0, Int128 m1, const Int8* f,
    const long* nzN, const Int8* nzG, long nnz,
    Int128 x
) {
    Int128 S = 0;

    for (Int128 m = m0; m < m1; m++) {
        long fm = f[m - m0];
        if (!fm) continue;
        Int128 xm = x / m;

        if (xm <= (Int128)__LONG_MAX__) {
            long lxm = (long)xm;
            Int128 Sm = 0;
            for (long i = 0; i < nnz; i++)
                Sm += (long)nzG[i] * (lxm / nzN[i]);
            S += (Int128)fm * Sm;
        } else {
            Int128 Sm = 0;
            for (long i = 0; i < nnz; i++)
                Sm += (Int128)nzG[i] * (xm / nzN[i]);
            S += fm * Sm;
        }
    }
    return S;
}

// Build the compact (n, mu(n)) list for bruteDoubleSum.
static void compactMu(const Int8* g, long lo, long len,
                      std::vector<long>& nzN, std::vector<Int8>& nzG) {
    nzN.reserve(len);
    nzG.reserve(len);
    for (long t = 0; t < len; t++) {
        if (g[t]) {
            nzN.push_back(lo + t);
            nzG.push_back(g[t]);
        }
    }
}

// ============================================================================
// DoubleSum — tiled evaluation via SumByLin
// ============================================================================

template <bool EnableWide>
static Int128 doubleSum(
    Int128 m0, Int128 m1, Int128 n0, Int128 n1,
    long a, long b, Int8* mux, Int8* muy, Int128 x
) {
    Int128 S = 0;
    for (Int128 mlow = m0; mlow < m1; mlow += 2 * a) {
        Int128 mhigh = mlow + 2 * a;
        if (mhigh > m1) mhigh = m1;
        Int128 mcirc = (mlow + mhigh) / 2;
        Int128 mdelt = (mhigh - mlow) / 2;
        for (Int128 nlow = n0; nlow < n1; nlow += 2 * b) {
            Int128 nhigh = nlow + 2 * b;
            if (nhigh > n1) nhigh = n1;
            Int128 ncirc = (nlow + nhigh) / 2;
            Int128 ndelt = (nhigh - nlow) / 2;
            S += sumByLin<EnableWide>(mux + mcirc - m0, muy + ncirc - n0,
                                      x, mcirc, ncirc, mdelt, ndelt);
        }
    }
    return S;
}

// ============================================================================
// Mu sieving helpers (using SegmentedMobiusSieveCore from ../clean)
// ============================================================================

static void sieveMu(Int8* out, Int128 lo, Int128 hi) {
    long len = (long)(hi - lo + 1);
    int32_t sqrtHi = (int32_t)isqrtLong((long)hi);
    int32_t primeBound = sqrtHi > (int32_t)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND
                       ? sqrtHi : (int32_t)SegmentedMobiusSieveCore::MIN_PRIMES_BOUND;
    auto primes = SegmentedMobiusSieveCore::primesUpTo(primeBound);
    SegmentedMobiusSieveCore sieve(len);
    sieve.sieve((int64_t)lo, (int64_t)hi, primes);
    for (long i = 0; i < len; i++)
        out[i] = sieve[i];
}

// ============================================================================
// DDSum — double-sum dispatch with parallelism
// ============================================================================

template <bool EnableWide>
static Int128 ddSum(
    Int128 A, Int128 Ap, Int128 B, Int128 Bp,
    Int128 x, Int128 D, int flag, long a, long b
) {
    long lenA = (long)(Ap - A);
    long lenB = (long)(Bp - B);
    std::vector<Int8> muABuf(lenA);
    Int8* muA = muABuf.data();

    sieveMu(muA, A, Ap - 1);

    std::vector<Int8> muBBuf;
    Int8* muB;
    if (A == B && Ap == Bp) {
        muB = muA;
    } else {
        muBBuf.resize(lenB);
        muB = muBBuf.data();
        sieveMu(muB, B, Bp - 1);
    }

    Int128 S = 0;
    long lA = (long)A, lAp = (long)Ap, lB = (long)B, lBp = (long)Bp, lD = (long)D;

    // Compact the nonzero mu(n) once; shared by all m-chunks below.
    std::vector<long> nzN;
    std::vector<Int8> nzG;
    if (!flag)
        compactMu(muB, lB, lenB, nzN, nzG);

#pragma omp parallel reduction(+:S)
    {
        if (flag) {
            long lm0, ln0;
#pragma omp for collapse(2) schedule(dynamic)
            for (lm0 = lA; lm0 < lAp; lm0 += lD)
                for (ln0 = lB; ln0 < lBp; ln0 += lD) {
                    Int128 m1v = lm0 + lD < lAp ? lm0 + lD : lAp;
                    Int128 n1v = ln0 + lD < lBp ? ln0 + lD : lBp;
                    S += doubleSum<EnableWide>(lm0, m1v, ln0, n1v, a, b,
                                               muA + (lm0 - lA), muB + (ln0 - lB), x);
                }
        } else {
            long lm0;
#pragma omp for schedule(dynamic)
            for (lm0 = lA; lm0 < lAp; lm0 += lD) {
                Int128 m1v = lm0 + lD < lAp ? lm0 + lD : lAp;
                S += bruteDoubleSum(lm0, m1v, muA + (lm0 - lA),
                                    nzN.data(), nzG.data(), (long)nzN.size(), x);
            }
        }
    }

    return S;
}

// Serial version of ddSum for use within outer-level parallelism
template <bool EnableWide>
static Int128 ddSumSerial(
    Int128 A, Int128 Ap, Int128 B, Int128 Bp,
    Int128 x, Int128 D, int flag, long a, long b
) {
    long lenA = (long)(Ap - A);
    long lenB = (long)(Bp - B);
    std::vector<Int8> muABuf(lenA);
    Int8* muA = muABuf.data();

    sieveMu(muA, A, Ap - 1);

    std::vector<Int8> muBBuf;
    Int8* muB;
    if (A == B && Ap == Bp) {
        muB = muA;
    } else {
        muBBuf.resize(lenB);
        muB = muBBuf.data();
        sieveMu(muB, B, Bp - 1);
    }

    Int128 S = 0;
    Int128 m0, m1, n0, n1;

    if (flag) {
        for (m0 = A; m0 < Ap; m0 += D)
            for (n0 = B; n0 < Bp; n0 += D) {
                m1 = m0 + D < Ap ? m0 + D : Ap;
                n1 = n0 + D < Bp ? n0 + D : Bp;
                S += doubleSum<EnableWide>(m0, m1, n0, n1, a, b,
                                           muA + (long)(m0 - A), muB + (long)(n0 - B), x);
            }
    } else {
        std::vector<long> nzN;
        std::vector<Int8> nzG;
        compactMu(muB, (long)B, lenB, nzN, nzG);

        for (m0 = A; m0 < Ap; m0 += D) {
            m1 = m0 + D < Ap ? m0 + D : Ap;
            S += bruteDoubleSum(m0, m1, muA + (long)(m0 - A),
                                nzN.data(), nzG.data(), (long)nzN.size(), x);
        }
    }

    return S;
}

// ============================================================================
// LargeFree — Loop 1: double sums over squarefree pairs
// ============================================================================

struct LFWorkItem {
    Int128 A, Ap, B, Bp, Delta;
    int flag;
    long a, b;
    long multiplier;  // 1 if A==B, 2 otherwise
};

template <bool EnableWide>
static Int128 largeFreeImpl(Int128 x, Int128 v, bool profile, long C, long D) {
    timeval tStart, tEnd;
    long lv = (long)v;
    long sqtv = isqrtLong(lv);

    // Phase 1: enumerate work items serially
    std::vector<LFWorkItem> dsumItems;
    std::vector<LFWorkItem> bruteItems;
    dsumItems.reserve(1024);

    Int128 Ap = v + 1;
    Int128 end1 = iqrtLong(96 * x * C * C * C);

    while (Ap >= 2 * D && Ap >= end1) {
        Int128 Bp = Ap;
        Int128 A = Ap - 2 * (Ap / (2 * D));
        long lA = (long)A;
        Int128 end2 = icbrtLong(48 * x * C * C * C / A);

        while (Bp >= 2 * D && Bp >= end2) {
            Int128 B = Bp - 2 * (Bp / (2 * D));
            long lB = (long)B;
            long a = icbrtLong((Int128)lA * lA * lA * lA / ((Int128)6 * x));
            long b = icbrtLong((Int128)lA * lB * lB * lB / ((Int128)6 * x));

            Int128 Delta = (1 + sqtv / iMax(2 * a, 2 * b)) * iMax(2 * a, 2 * b);
            dsumItems.push_back({A, Ap, B, Bp, Delta, 1, a, b, (A == B ? 1 : 2)});
            Bp = B;
        }
        bruteItems.push_back({A, Ap, 1, Bp, (Int128)sqtv + 1, 0, 0, 0, 2});
        Ap = A;
    }
    bruteItems.push_back({1, Ap, 1, Ap, (Int128)sqtv + 1, 0, 0, 0, 1});

    long nds = (long)dsumItems.size();
    long nbr = (long)bruteItems.size();
    if (profile)
        printf("  LargeFree: %ld DoubleSum + %ld brute items\n", nds, nbr);

    // Phase 2a: many small DoubleSum items (outer parallelism, each runs serial)
    double tDsum = 0, tBrute = 0;
    START_PROFILE();
    Int128 S = 0;
#pragma omp parallel reduction(+:S)
    {
#pragma omp for schedule(dynamic)
        for (long i = 0; i < nds; i++) {
            const LFWorkItem& w = dsumItems[i];
            Int128 T = ddSumSerial<EnableWide>(w.A, w.Ap, w.B, w.Bp, x, w.Delta, w.flag, w.a, w.b);
            S += T * w.multiplier;
        }
    }
    END_PROFILE(tDsum);

    // Phase 2b: few large brute items (internal parallelism)
    START_PROFILE();
    for (long i = 0; i < nbr; i++) {
        const LFWorkItem& w = bruteItems[i];
        Int128 T = ddSum<EnableWide>(w.A, w.Ap, w.B, w.Bp, x, w.Delta, w.flag, w.a, w.b);
        S += T * w.multiplier;
    }
    END_PROFILE(tBrute);

    if (profile) {
        printf("  LargeFree breakdown:\n");
        printf("        DoubleSum: %10.6f  %6.2f%%\n", tDsum, 100.0 * tDsum / (tDsum + tBrute));
        printf("            Brute: %10.6f  %6.2f%%\n", tBrute, 100.0 * tBrute / (tDsum + tBrute));
    }

    return S;
}

static inline bool htWideArithmeticEnabled(Int128 x) {
    static const Int128 kOneE23 = (Int128)1000000000000LL * (Int128)100000000000LL;
    return x > kOneE23;
}

static Int128 largeFree(Int128 x, Int128 v, bool profile, long C, long D) {
    if (htWideArithmeticEnabled(x))
        return largeFreeImpl<true>(x, v, profile, C, D);
    return largeFreeImpl<false>(x, v, profile, C, D);
}

// ============================================================================
// BruteM — direct sieve of M(u)
// ============================================================================

static Int128 bruteM(Int128 x) {
    // For large enough x, use the O(x^{2/3}) implementation
    if (x >= 100000000)
        return (Int128)MertensHurst((UInt128)x);

    return (Int128)MertensSieve((UInt64)x);
}

// ============================================================================
// SArr — build cumulative array of FacToSumMu values
// ============================================================================

// Profile accumulators
static double sarrFactblock = 0, sarrFacToSum = 0, sarrOffset = 0;

// S holds TASK-LOCAL Int64 prefix sums; the global Int128 value of entry idx
// is offset[idx / (Delta+1)] + S[idx], reconstructed by the consumer. Folding
// the offset into the ~sqrt(x) lookups avoids a full fixup sweep over all
// bigDelta entries. Int64 is safe task-locally (|FacToSumMu| <= 2^omega(r)
// over Delta+1 entries); only the global offsets need Int128. offset[] has
// ntasks+1 entries: offset[ntasks] seeds the next sArr call.
static void sArr(
    long* S, short* isprime, Int128 x, Int128 r0,
    long Delta, long bigDelta, Int128 S0,
    PrimFact* fun, ulong* sqfprod, Int128* offset,
    int ntasks, bool profile
) {
    timeval tStart, tEnd;

    // Pre-allocate per-thread tabla buffers for fillFactBlock
    int nthreads = omp_get_max_threads();
    ulong** threadTabla = new ulong*[nthreads];
    for (int t = 0; t < nthreads; t++) {
        threadTabla[t] = (ulong*)malloc((Delta + 1) * sizeof(ulong));
        if (!threadTabla[t]) {
            std::fprintf(stderr, "largeNonFree: threadTabla allocation failed\n");
            std::abort();
        }
    }

    START_PROFILE();
#pragma omp parallel for schedule(dynamic)
    for (long j = 0; j < ntasks; j++)
        fillFactBlock(fun + j * (Delta + 1), sqfprod + j * (Delta + 1),
                      isprime, r0 + j * (Delta + 1), Delta + 1,
                      threadTabla[omp_get_thread_num()]);
    END_PROFILE(sarrFactblock);

    // 64-bit fast path check
    int use64 = ((uI)x <= (uI)(ulong)-1);
    ulong lx = use64 ? (ulong)x : 0;

    START_PROFILE();
#pragma omp parallel for schedule(dynamic)
    for (long j = 0; j < ntasks; j++) {
        long r1 = r0 + j * (Delta + 1);
        long Si = 0;

        // Seed quotient predictor
        long qPrev = use64 ? (long)(lx / (ulong)r1) : (long)(x / r1);
        Si += FacToSumMu(fun[r1 - r0], qPrev, sqfprod[r1 - r0]);
        S[r1 - r0] = Si;

        long qCur, qEst;
        if (r1 + 1 <= r1 + Delta) {
            qCur = use64 ? (long)(lx / (ulong)(r1 + 1)) : (long)(x / (r1 + 1));
            Si += FacToSumMu(fun[r1 + 1 - r0], qCur, sqfprod[r1 + 1 - r0]);
            S[r1 + 1 - r0] = Si;
        }

        if (use64) {
            for (long r = r1 + 2; r <= r1 + Delta; r++) {
                qEst = 2 * qCur - qPrev;
                ulong mul = (ulong)qEst * (ulong)r;
                if (__builtin_expect(mul <= lx, 1)) {
                    if (__builtin_expect(mul + (ulong)r <= lx, 0))
                        do { qEst++; mul += (ulong)r; } while (mul + (ulong)r <= lx);
                } else {
                    do { qEst--; mul -= (ulong)r; } while (mul > lx);
                }
                qPrev = qCur;
                qCur = qEst;
                Si += FacToSumMu(fun[r - r0], qEst, sqfprod[r - r0]);
                S[r - r0] = Si;
            }
        } else {
            for (long r = r1 + 2; r <= r1 + Delta; r++) {
                qEst = 2 * qCur - qPrev;
                Int128 mul = (Int128)qEst * r;
                if (__builtin_expect(mul <= x, 1)) {
                    if (__builtin_expect(mul + r <= x, 0))
                        do { qEst++; mul += r; } while (mul + r <= x);
                } else {
                    do { qEst--; mul -= r; } while (mul > x);
                }
                qPrev = qCur;
                qCur = qEst;
                Si += FacToSumMu(fun[r - r0], qEst, sqfprod[r - r0]);
                S[r - r0] = Si;
            }
        }
    }
    END_PROFILE(sarrFacToSum);

    START_PROFILE();
    offset[0] = S0;
    for (long j = 1; j < ntasks; j++)
        offset[j] = offset[j - 1] + S[j * (Delta + 1) - 1];
    offset[ntasks] = offset[ntasks - 1] + S[bigDelta - 1];
    END_PROFILE(sarrOffset);

    for (int t = 0; t < nthreads; t++)
        free(threadTabla[t]);
    delete[] threadTabla;
}

// ============================================================================
// LargeNonFree — Loop 2: factorized partial sums
// ============================================================================

static Int128 largeNonFree(Int128 x, long v, long u, int ntasks, bool profile, Int64 sarrCapBytes) {
    timeval tStart, tEnd;
    double tSieve = 0, tSarr = 0;

    long n0 = u + 1;
    long r0 = (long)(x / (u + 1)) + 1;
    long Delta = isqrt128(iMax((Int128)u, x / v)) + 1;
    long bigDelta = ntasks * (Delta + 1);
    // Cap bigDelta so SArr working arrays stay under sarrCapBytes (default 12 GB).
    // S(8B) + fun(16B) + sqfprod(8B) = 32 bytes per entry.
    // When capped, reduce ntasks so sArr's j*Delta indexing stays in bounds.
    {
        long maxBigDelta = (long)(sarrCapBytes / 32);
        if (bigDelta > maxBigDelta) {
            bigDelta = maxBigDelta;
            ntasks = (int)(bigDelta / (Delta + 1));
            if (ntasks < 1) ntasks = 1;
            bigDelta = (long)ntasks * (Delta + 1);
        }
    }
    long safeDelta = isqrt128(iMax((Int128)u, x / v + bigDelta)) + 1;

    short* isprime = (short*)calloc(safeDelta + 1, sizeof(short));
    Int8* mup = (Int8*)malloc((Delta + 1) * sizeof(Int8));
    long* S = (long*)malloc(bigDelta * sizeof(long));

    PrimFact* sarrFun = (PrimFact*)malloc(bigDelta * sizeof(PrimFact));
    ulong* sarrSqfprod = (ulong*)malloc(bigDelta * sizeof(ulong));
    Int128* sarrOff = (Int128*)malloc((ntasks + 1) * sizeof(Int128));

    if (!isprime || !mup || !S || !sarrFun || !sarrSqfprod || !sarrOff) {
        std::fprintf(stderr, "largeNonFree: allocation failed\n");
        std::abort();
    }

    fillIsPrime(isprime, safeDelta + 1);
    initFactStencil(isprime);

    START_PROFILE();
    sArr(S, isprime, x, r0, Delta, bigDelta, 1, sarrFun, sarrSqfprod, sarrOff, ntasks, profile);
    END_PROFILE(tSarr);

    Int128 Sum = 0, sig = 0;

    {
        long nThresh = (long)(x / ((Int128)r0 + bigDelta));
        Int8 muN;

        // Quotient predictor for x/n (n decreasing)
        int use64 = ((uI)x <= (uI)(ulong)-1);
        ulong lxLnf = use64 ? (ulong)x : 0;
        long qCur  = use64 ? (long)(lxLnf / (ulong)u) : (long)(x / u);
        long qPrev = use64 ? (long)(lxLnf / (ulong)(u + 1)) : (long)(x / (u + 1));
        long qEst;

        for (long n = u; n > v; n--) {
            if (n < n0) {
                n0 = n0 - (Delta + 1);
                if (n0 < 1) n0 = 1;
                START_PROFILE();
                fillMuBlock(mup, isprime, n0, Delta + 1);
                END_PROFILE(tSieve);
            }

            while (n <= nThresh) {
                r0 += bigDelta;
                START_PROFILE();
                sArr(S, isprime, x, r0, Delta, bigDelta, sarrOff[ntasks],
                     sarrFun, sarrSqfprod, sarrOff, ntasks, profile);
                END_PROFILE(tSarr);
                nThresh = (long)(x / ((Int128)r0 + bigDelta));
            }

            // Predict floor(x/n)
            qEst = 2 * qCur - qPrev;
            if (use64) {
                ulong mul = (ulong)qEst * (ulong)n;
                if (__builtin_expect(mul <= lxLnf, 1)) {
                    if (__builtin_expect(mul + (ulong)n <= lxLnf, 0))
                        do { qEst++; mul += (ulong)n; } while (mul + (ulong)n <= lxLnf);
                } else {
                    do { qEst--; mul -= (ulong)n; } while (mul > lxLnf);
                }
            } else {
                Int128 mul = (Int128)qEst * n;
                if (__builtin_expect(mul <= x, 1)) {
                    if (__builtin_expect(mul + n <= x, 0))
                        do { qEst++; mul += n; } while (mul + n <= x);
                } else {
                    do { qEst--; mul -= n; } while (mul > x);
                }
            }
            qPrev = qCur;
            qCur = qEst;

            muN = mup[n - n0];
            if (muN) {
                long xn = qEst;
                Int128 mu_quot = muN * (xn / n);
                sig += mu_quot;
                // Global S value = per-task offset + task-local Int64 prefix
                const long idx = xn - r0;
                Sum += muN * mu_quot
                     + 2 * muN * (-sig + (sarrOff[idx / (Delta + 1)] + (Int128)S[idx]));
            }
        }
    }

    if (profile) {
        double tTotal = tSieve + tSarr;
        printf("  LargeNonFree breakdown:\n");
        printf("        fillMuBlock: %10.6f  %6.2f%%\n", tSieve, tTotal > 0 ? 100.0 * tSieve / tTotal : 0);
        printf("               SArr: %10.6f  %6.2f%%\n", tSarr, tTotal > 0 ? 100.0 * tSarr / tTotal : 0);
        double tSarrTotal = sarrFactblock + sarrFacToSum + sarrOffset;
        printf("    SArr breakdown:\n");
        printf("      fillFactBlock: %10.6f  %6.2f%%\n", sarrFactblock, tSarrTotal > 0 ? 100.0 * sarrFactblock / tSarrTotal : 0);
        printf("        FacToSumMu : %10.6f  %6.2f%%\n", sarrFacToSum, tSarrTotal > 0 ? 100.0 * sarrFacToSum / tSarrTotal : 0);
        printf("            offset : %10.6f  %6.2f%%\n", sarrOffset, tSarrTotal > 0 ? 100.0 * sarrOffset / tSarrTotal : 0);
        sarrFactblock = sarrFacToSum = sarrOffset = 0;
    }

    free(sarrFun); free(sarrSqfprod); free(sarrOff);
    free(isprime); free(mup); free(S);
    return Sum;
}

} // anonymous namespace

// ============================================================================
// Public entry point
// ============================================================================

Int64 MertensHT(Int128 x, bool profile, double vdiv, Int64 paramC, Int64 paramD, Int64 ntasksMul, Int64 sarrCapBytes) {
    timeval t0, t1, t2, t3;

    int ntasks;
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            ntasks = omp_get_num_threads();
    }
    ntasks *= ntasksMul;

    long u = isqrt128(x);

    // v ~ x^{2/5} / vdiv: fifth root, then square. v is a tuning knob, so the
    // small difference from an exact floor((x^2)^{1/5}) does not matter.
    long t = i5rtLong(x);
    long v = (long)(t * t / vdiv);

    if (profile) {
        printf("v = %ld, u = %ld (vdiv=%.2f, C=%lld, D=%lld, ntasks=%d)\n",
               v, u, vdiv, (long long)paramC, (long long)paramD, ntasks);
    }

    getDayTime(t0);
    Int128 Bu = bruteM(u);
    getDayTime(t1);

#if MHT_CPU_OVERLAP
    // Run the two independent phases concurrently. Per-phase profiling is not
    // meaningful while they overlap, so the inner breakdowns are suppressed and
    // a single combined region time is reported.
    Int128 LF = 0, LNF = 0;
    std::thread lfThread([&] { LF = largeFree(x, v, false, paramC, paramD); });
    LNF = largeNonFree(x, v, u, ntasks, false, sarrCapBytes);
    lfThread.join();
    getDayTime(t2);
    getDayTime(t3);

    if (profile) {
        printf("--------- Profile (CPU OVERLAP) ----------\n");
        printf("                     BruteM: %10.6f\n", getDuration(t0, t1));
        printf("  LargeFree || LargeNonFree: %10.6f\n", getDuration(t1, t3));
        printf("                      Total: %10.6f\n", getDuration(t0, t3));
        printf("------------------------------------------\n");
    }
#else
    Int128 LF = largeFree(x, v, profile, paramC, paramD);
    getDayTime(t2);

    Int128 LNF = largeNonFree(x, v, u, ntasks, profile, sarrCapBytes);
    getDayTime(t3);

    if (profile) {
        double tBruteM       = getDuration(t0, t1);
        double tLargeFree    = getDuration(t1, t2);
        double tLargeNonFree = getDuration(t2, t3);
        double tTotal        = getDuration(t0, t3);

        printf("------------- Profile --------------\n");
        printf("       BruteM: %10.6f  %6.2f%%\n", tBruteM, 100.0 * tBruteM / tTotal);
        printf("    LargeFree: %10.6f  %6.2f%%\n", tLargeFree, 100.0 * tLargeFree / tTotal);
        printf(" LargeNonFree: %10.6f  %6.2f%%\n", tLargeNonFree, 100.0 * tLargeNonFree / tTotal);
        printf("------------------------------------\n");
        printf("        Total: %10.6f\n", tTotal);
        printf("------------------------------------\n");
    }
#endif

    return (Int64)(2 * Bu - LNF - LF);
}
