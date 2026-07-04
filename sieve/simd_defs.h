#pragma once

// ============================================================================
// simd_defs.h — SIMD platform detection for the segmented Mertens/Mobius sieve.
//
// Detection hierarchy (first match wins, supersets also define their subsets):
//
//   ARM:
//     SIEVE_SIMD_SVE2  — ARM SVE2 (Graviton 3+, Neoverse V1+)
//     SIEVE_SIMD_NEON  — ARM NEON (Apple Silicon, AArch64, ARMv7+NEON)
//
//   x86:
//     SIEVE_SIMD_AVX512 — AVX-512BW (Skylake-X 2017+, Zen 4 2022+)
//     SIEVE_SIMD_AVX2   — AVX2 (Haswell 2013+, Zen 2017+)
//     SIEVE_SIMD_SSE    — SSE2 (baseline for all x86-64)
//
// When a wider tier is detected, all narrower tiers on the same architecture
// are also defined. This lets code that only needs 128-bit ops (e.g. prefix
// sum) test for SIEVE_SIMD_SSE or SIEVE_SIMD_NEON without caring about the
// exact tier.
//
// If nothing matches, no macros are defined and plain scalar C++ is used.
// ============================================================================

// --- ARM ---
#if defined(__ARM_FEATURE_SVE2)
    #define SIEVE_SIMD_SVE2 1
    #define SIEVE_SIMD_NEON 1      // SVE2 implies NEON on AArch64
    #include <arm_sve.h>
    #include <arm_neon.h>

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SIEVE_SIMD_NEON 1
    #include <arm_neon.h>

// --- x86 ---
#elif defined(__AVX512BW__)
    #define SIEVE_SIMD_AVX512 1
    #define SIEVE_SIMD_AVX2   1    // AVX-512 implies AVX2
    #define SIEVE_SIMD_SSE    1
    #include <immintrin.h>

#elif defined(__AVX2__)
    #define SIEVE_SIMD_AVX2 1
    #define SIEVE_SIMD_SSE  1      // AVX2 implies SSE2
    #include <immintrin.h>

#elif defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
    #define SIEVE_SIMD_SSE 1
    #include <emmintrin.h>
    #if defined(__SSSE3__)
        #include <tmmintrin.h>   // pshufb, used for the prefix-sum carry broadcast
    #endif
#endif
