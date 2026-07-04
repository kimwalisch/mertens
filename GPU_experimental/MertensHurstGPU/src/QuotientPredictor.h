#pragma once

// ============================================================================
// QuotientPredictor.h — Division-free quotient estimation via recurrence.
//
// Given n and a sequence of decreasing x values, we predict floor(n/x) from
// the two previous quotients using:
//
//   qEst = 2*qCur - qPrev + e
//
// where e is a small correction verified by a single multiply.
// This avoids expensive division in the inner loops.
//
// 128-bit variants (n is UInt128):
//   update_quotients()              — stride-1 (every x)
//   update_quotients_coprime6iter() — stride-2/4 coprime-to-6 iteration
//
// 64-bit variants (n is UInt64, used with QuotientCache path):
//   update_quotients_64()              — stride-1 (every x)
//   update_quotients_coprime6iter_64() — stride-2/4 coprime-to-6 iteration
// ============================================================================

#include "types.h"

static inline void update_quotients(
    const UInt128& n,
    const UInt64& x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    qEst = 2*qCur - qPrev;

    UInt128 mul = static_cast<UInt128>(qEst) * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        --qEst;
    }

    qPrev = qCur;
    qCur  = qEst;
}

// DoubleJump=true: advance by 4 (coprime-to-6 skip pattern: +4, +2, +4, +2, ...)
// DoubleJump=false: advance by 2
template<bool DoubleJump>
static inline void update_quotients_coprime6iter(
    const UInt128& n,
    const UInt64& x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    if constexpr (DoubleJump) {
        qEst = 2*qCur - qPrev;

        qPrev = qCur;
        qCur  = qEst;
    }

    qEst = 2*qCur - qPrev;

    UInt128 mul = static_cast<UInt128>(qEst) * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        do {
            --qEst;
            mul -= x;
        } while (mul > n);
    }

    qPrev = qCur;
    qCur  = qEst;
}

// ============================================================================
// 64-bit variants: same recurrence, but n fits in 64 bits.
// For the 64-bit path, partial_args < 10^18 < 2^60, so qEst * x fits in
// 64 bits (avoiding expensive 128-bit arithmetic). Verification uses a single
// 64-bit multiply.
// ============================================================================

static inline void update_quotients_64(
    const UInt64 n,
    const UInt64 x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    qEst = 2*qCur - qPrev;

    UInt64 mul = qEst * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        --qEst;
    }

    qPrev = qCur;
    qCur  = qEst;
}

template<bool DoubleJump>
static inline void update_quotients_coprime6iter_64(
    const UInt64 n,
    const UInt64 x,
    UInt64& qCur,
    UInt64& qPrev,
    UInt64& qEst
) {
    if constexpr (DoubleJump) {
        qEst = 2*qCur - qPrev;

        qPrev = qCur;
        qCur  = qEst;
    }

    qEst = 2*qCur - qPrev;

    UInt64 mul = qEst * x;

    if (__builtin_expect(mul <= n, true)) {
        if (__builtin_expect(mul + x <= n, false)) {
            do {
                ++qEst;
                mul += x;
            } while (mul + x <= n);
        }
    } else {
        do {
            --qEst;
            mul -= x;
        } while (mul > n);
    }

    qPrev = qCur;
    qCur  = qEst;
}
