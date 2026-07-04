#pragma once

// ============================================================================
// IntegerMath.h — Floor-division, modular arithmetic, and integer root
//                 functions for __int128 and long operands.
//
// All functions are static inline for use in performance-critical inner loops.
// Where possible, a 64-bit fast path avoids expensive 128-bit __divti3 calls
// (gated by __builtin_expect since the fast path dominates at runtime).
// ============================================================================

#include "types.h"
#include <cmath>

typedef unsigned __int128 uI;

// ============================================================================
// Modular arithmetic
// ============================================================================

// Floor-mod: result in [0, |b|). Handles negative a correctly.
static inline Int128 mod(Int128 a, Int128 b) {
    if (b < 0) b = -b;
    if (__builtin_expect(
            (uI)(a + (Int128)__LONG_MAX__) <= (uI)2*__LONG_MAX__ &&
            b <= (Int128)__LONG_MAX__, 1)) {
        long al = (long)a, bl = (long)b;
        return (al >= 0) ? al % bl : bl - 1 - (-al - 1) % bl;
    }
    return (a >= 0) ? a % b : b - 1 - (-a - 1) % b;
}

// Floor-mod with long denominator (a may be 128-bit).
static inline long modl(Int128 a, long b) {
    if (__builtin_expect(
            (uI)(a + (Int128)__LONG_MAX__) <= (uI)2*__LONG_MAX__, 1)) {
        long al = (long)a;
        return (al >= 0) ? al % b : b - 1 - (-al - 1) % b;
    }
    return (a >= 0) ? a % b : b - 1 - (-a - 1) % b;
}

// ============================================================================
// Floor division
// ============================================================================

// Floor division: handles negative numerator and denominator.
static inline Int128 divf(Int128 a, Int128 b) {
    if (b < 0) { a = -a; b = -b; }
    if (__builtin_expect(
            (uI)(a + (Int128)__LONG_MAX__) <= (uI)2*__LONG_MAX__ &&
            b <= (Int128)__LONG_MAX__, 1)) {
        long al = (long)a, bl = (long)b;
        return (al >= 0) ? al / bl : al / bl - (al % bl != 0);
    }
    return (a >= 0) ? a / b : (a - (b - 1)) / b;
}

// Floor division for known-positive denominator (skips sign check).
static inline Int128 divfPos(Int128 a, Int128 b) {
    if (__builtin_expect(
            (uI)(a + (Int128)__LONG_MAX__) <= (uI)2*__LONG_MAX__ &&
            b <= (Int128)__LONG_MAX__, 1)) {
        long al = (long)a, bl = (long)b;
        return (al >= 0) ? al / bl : al / bl - (al % bl != 0);
    }
    return (a >= 0) ? a / b : (a - (b - 1)) / b;
}

// Simultaneous floor and ceiling division.
static inline void divfc(Int128 a, Int128 b, Int128& fl, Int128& cl) {
    if (b < 0) { a = -a; b = -b; }
    if (a >= 0) {
        fl = a / b;
        cl = fl + (Int128)(a != fl * b);
    } else {
        cl = a / b;
        fl = cl - (Int128)(a != cl * b);
    }
}

// divfc for known-positive denominator.
static inline void divfcPos(Int128 a, Int128 b, Int128& fl, Int128& cl) {
    if (a >= 0) {
        fl = a / b;
        cl = fl + (Int128)(a != fl * b);
    } else {
        cl = a / b;
        fl = cl - (Int128)(a != cl * b);
    }
}

// ============================================================================
// Integer roots
// ============================================================================

// Integer square root for long inputs.
static inline long isqrtLong(long n) {
    long s = (long)sqrtl((double)n);
    while (s > 0 && s * s > n) s--;
    while ((s + 1) * (s + 1) <= n) s++;
    return s;
}

// Integer cube root for __int128 inputs (result fits in long).
static inline long icbrtLong(Int128 n) {
    if (n <= 0) return 0;
    long s = (long)cbrtl((double)n);
    if (s < 0) s = 0;
    while (s > 0 && (Int128)s * s * s > n) s--;
    while ((Int128)(s + 1) * (s + 1) * (s + 1) <= n) s++;
    return s;
}

// Integer 4th root for __int128 inputs (result fits in long).
static inline long iqrtLong(Int128 n) {
    if (n <= 0) return 0;
    long s = (long)sqrtl(sqrtl((double)n));
    if (s < 0) s = 0;
    while (s > 0 && (Int128)s * s * s * s > n) s--;
    while ((Int128)(s + 1) * (s + 1) * (s + 1) * (s + 1) <= n) s++;
    return s;
}

// Integer 5th root for __int128 inputs (result fits in long).
static inline long i5rtLong(Int128 n) {
    if (n <= 0) return 0;
    long s = (long)powl((long double)n, 0.2L);
    if (s < 0) s = 0;
    while (s > 0 && (Int128)s * s * s * s * s > n) s--;
    while ((Int128)(s + 1) * (s + 1) * (s + 1) * (s + 1) * (s + 1) <= n) s++;
    return s;
}

// Integer square root for unsigned 64-bit inputs.
static inline unsigned long sqrt64(unsigned long n) {
    unsigned long s = (unsigned long)sqrtl((double)n);
    while (s > 0 && (uI)s * s > n) s--;
    while ((uI)(s + 1) * (s + 1) <= n) s++;
    return s;
}

// Integer square root for __int128 inputs.
// Dispatches to sqrt64 when possible, otherwise uses Newton's method.
static inline long isqrt128(Int128 n) {
    if (n <= 0) return 0;
    if (n <= (Int128)__LONG_MAX__) return isqrtLong((long)n);

    // Full Newton iteration for large values
    uI un = (uI)n;

    auto log2_128 = [](uI x) -> long {
        unsigned long high = x >> 64;
        if (high) return __builtin_clzll(high) ? 63 - __builtin_clzll(high) + 64 : 127;
        return 63 - __builtin_clzll((unsigned long)x);
    };

    int k = log2_128(un);
    int be = k / 4 + 1;
    int flag = 0;
    if ((k % 4) < 2) { flag = 1; un <<= 2; }

    unsigned long head = un >> (2 * be);
    unsigned long a1 = un - ((uI)head << (2 * be));
    a1 >>= be;

    unsigned long sp = sqrt64(head);
    unsigned long sp2 = sp * sp;
    unsigned long rp;
    if (head > sp2) {
        rp = head - sp2;
    } else {
        rp = (head + 2 * sp - 1) - sp2;
        sp--;
    }

    unsigned long num = ((rp << be) + a1);
    unsigned long den = 2 * sp;
    unsigned long q = num / den;
    unsigned long s = (sp << be) + q;

    if ((uI)s * (uI)s > un) s--;
    if (flag) s >>= 1;

    return (long)s;
}
