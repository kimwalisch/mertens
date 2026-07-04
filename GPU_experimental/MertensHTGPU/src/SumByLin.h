#pragma once

// ============================================================================
// SumByLin.h — Diophantine-approximation-based double sum via linearization.
//
// Implements the SumByLin method from Helfgott-Thompson: given mu values over
// m-range and n-range, computes the "free" double sum using:
//   1. LinearSum: first-order Taylor approximation
//   2. Correction via Diophantine approximation (SumTable, SumInter, etc.)
//
// All functions are force-inlined since they form the innermost hot loops of
// the LargeFree computation (Loop 1: S1+S2, ~65% of total runtime).
// ============================================================================

#include "IntegerMath.h"
#include <boost/assert/source_location.hpp>
#include <boost/multiprecision/cpp_int.hpp>
#include <cstdlib>
#include <exception>
#include <vector>

namespace boost {
inline void throw_exception(std::exception const&) {
    std::abort();
}

inline void throw_exception(std::exception const&, boost::source_location const&) {
    std::abort();
}
}

using boost::multiprecision::int256_t;
using boost::multiprecision::uint256_t;

#define sgn(x) ((x > 0) - (x < 0))

#ifndef MIN_MAX_DEFINED
#define MIN_MAX_DEFINED
#define iMin(x,y) ((x) < (y) ? (x) : (y))
#define iMax(x,y) ((x) > (y) ? (x) : (y))
#endif

// ============================================================================
// Types
// ============================================================================

struct DiophApp {
    Int128 a, q, ainv;
    int s;
};

struct Interval128 {
    Int128 left, right;
};

static thread_local std::vector<long> g_sumByLinG;
static thread_local std::vector<long> g_sumByLinRho;
static thread_local std::vector<long> g_sumByLinSigma;

static inline bool intervalEmpty(Interval128 J) {
    return J.left > J.right;
}

static inline bool intervalFull(Interval128 J, long b) {
    return J.left <= -b && J.right >= b - 1;
}

static inline bool intervalEqual(Interval128 A, Interval128 B) {
    return A.left == B.left && A.right == B.right;
}

static const Int128 kInt128MaxSafe = (((Int128)1 << 126) - 1) * 2 + 1;

static inline uI absU128(Int128 x) {
    return x >= 0 ? (uI)x : (uI)(-(x + 1)) + 1;
}

static inline bool mulFitsI128(Int128 a, Int128 b) {
    uI aa = absU128(a), bb = absU128(b);
    return aa == 0 || bb <= (uI)kInt128MaxSafe / aa;
}

static inline bool addFitsI128(Int128 a, Int128 b) {
    if (b > 0) return a <= kInt128MaxSafe - b;
    if (b < 0) return a >= -kInt128MaxSafe - b;
    return true;
}

static inline bool subFitsI128(Int128 a, Int128 b) {
    return addFitsI128(a, -b);
}

static inline int256_t toI256(Int128 x) {
    return x >= 0 ? int256_t((uI)x) : -int256_t(absU128(x));
}

static inline uint256_t toU256(uI x) {
    return uint256_t(x);
}

static inline uI u256ToU128(uint256_t x) {
    static const uint256_t mask64 = (uint256_t(1) << 64) - 1;
    unsigned long long lo = (x & mask64).convert_to<unsigned long long>();
    unsigned long long hi = ((x >> 64) & mask64).convert_to<unsigned long long>();
    return ((uI)hi << 64) | (uI)lo;
}

static inline Int128 i256ToI128(int256_t x) {
    if (x < 0) {
        uint256_t ax = (uint256_t)(-x);
        return -(Int128)u256ToU128(ax);
    }
    return (Int128)u256ToU128((uint256_t)x);
}

static inline Int128 isqrt256(uint256_t n) {
    if (n == 0) return 0;
    unsigned int bits = boost::multiprecision::msb(n) + 1;
    uint256_t x = uint256_t(1) << ((bits + 1) / 2);
    for (;;) {
        uint256_t y = (x + n / x) >> 1;
        if (y >= x) {
            while (x * x > n) --x;
            while ((x + 1) * (x + 1) <= n) ++x;
            return (Int128)u256ToU128(x);
        }
        x = y;
    }
}

static inline void divfcPosWide(int256_t a, Int128 b, Int128& fl, Int128& cl) {
    int256_t den = toI256(b);
    int256_t q = a / den;
    int256_t r = a - q * den;
    if (a >= 0) {
        fl = i256ToI128(q);
        cl = i256ToI128(q + (r != 0));
    } else {
        cl = i256ToI128(q);
        fl = i256ToI128(q - (r != 0));
    }
}

static inline bool quadDeltaFitsI128(Int128 a, Int128 b, Int128 c, Int128& Delta) {
    if (!mulFitsI128(b, b)) return false;
    Int128 b2 = b * b;
    if (!mulFitsI128(a, c)) return false;
    Int128 ac = a * c;
    if (!mulFitsI128(4, ac)) return false;
    Int128 fourac = 4 * ac;
    if (!subFitsI128(b2, fourac)) return false;
    Delta = b2 - fourac;
    return true;
}

static inline bool quadEvalFitsI128(Int128 a, Int128 b, Int128 c,
                                    Int128 x, Int128& y) {
    if (!mulFitsI128(x, x)) return false;
    Int128 x2 = x * x;
    if (!mulFitsI128(a, x2)) return false;
    Int128 ax2 = a * x2;
    if (!mulFitsI128(b, x)) return false;
    Int128 bx = b * x;
    if (!addFitsI128(ax2, bx)) return false;
    Int128 s = ax2 + bx;
    if (!addFitsI128(s, c)) return false;
    y = s + c;
    return true;
}

static inline int256_t quadEvalWide(Int128 a, Int128 b, Int128 c, Int128 x) {
    int256_t xx = toI256(x);
    return toI256(a) * xx * xx + toI256(b) * xx + toI256(c);
}

template <bool EnableWide>
static inline bool quadIntervalCovers(Int128 a, Int128 b, Int128 c,
                                      Int128 left, Int128 right) {
    if constexpr (!EnableWide) {
        Int128 pLeft = a * left * left + b * left + c;
        Int128 pRight = a * right * right + b * right + c;
        if (a > 0)
            return pLeft <= 0 && pRight <= 0;
        if (a < 0)
            return pLeft >= 0 && pRight >= 0;
        return false;
    }

    Int128 pLeft, pRight;
    if (!quadEvalFitsI128(a, b, c, left, pLeft) ||
        !quadEvalFitsI128(a, b, c, right, pRight)) {
        int256_t pLeftWide = quadEvalWide(a, b, c, left);
        int256_t pRightWide = quadEvalWide(a, b, c, right);
        if (a > 0)
            return pLeftWide <= 0 && pRightWide <= 0;
        if (a < 0)
            return pLeftWide >= 0 && pRightWide >= 0;
        return false;
    }
    if (a > 0)
        return pLeft <= 0 && pRight <= 0;
    if (a < 0)
        return pLeft >= 0 && pRight >= 0;
    return false;
}

// ============================================================================
// Diophantine approximation
// ============================================================================

// Constructs a/q with q <= Q approximating alpa/alpq, and computes a^{-1} mod q.
static inline DiophApp diophApp(Int128 alpa, Int128 alpq, long Q) {
    Int128 b, p, q, pmin, qmin, pplus, qplus, flip, qacc;
    int s;
    DiophApp res;

    b = divf(alpa, alpq);
    p = b; q = 1; pmin = 1; qmin = 0; s = 1;
    while (q <= Q) {
        if (alpa == b * alpq) {
            flip = (s == 1 ? -qmin : qmin);
            res.a = p; res.q = q; res.ainv = mod(flip, q); res.s = 0;
            return res;
        }

        qacc = alpq; alpq = alpa - b * alpq; alpa = qacc;
        b = divf(alpa, alpq);

        pplus = b * p + pmin; qplus = b * q + qmin;
        pmin = p; qmin = q; p = pplus; q = qplus; s = -s;
    }

    flip = (s == 1 ? q : -q);
    res.a = pmin; res.ainv = mod(flip, qmin);
    res.q = qmin; res.s = -s;
    return res;
}

// ============================================================================
// LinearSum — first-order approximation with incremental quotient tracking
// ============================================================================

static inline Int128 linearSum(
    const Int8* f, const Int8* g,
    long a, long b, Int128 x, Int128 m0, Int128 n0
) {
    Int128 S1, S2, S10, S20;
    Int128 uden;

    // Sum over m: floor(x*(m0+a-m) / (m0^2*n0))
    uden = m0 * m0 * n0;
    {
        Int128 unum = x * (m0 + a);
        Int128 xdiv = x / uden, xmod = x - xdiv * uden;
        Int128 qf = unum / uden, qr = unum - qf * uden;
        S1 = 0; S10 = 0;
        for (long m = -a; m < a; m++) {
            if (f[m]) { S1 += f[m] * qf; S10 += f[m]; }
            qr -= xmod; qf -= xdiv;
            if (qr < 0) { qr += uden; qf--; }
        }
    }

    // Sum over n: floor(x*b-n*x) / (m0*n0^2))
    uden = m0 * n0 * n0;
    {
        Int128 unum = x * b;
        Int128 xdiv = x / uden, xmod = x - xdiv * uden;
        Int128 qf = unum / uden, qr = unum - qf * uden;
        S2 = 0; S20 = 0;
        for (long n = -b; n < b; n++) {
            if (g[n]) { S2 += g[n] * qf; S20 += g[n]; }
            qr -= xmod; qf -= xdiv;
            if (qr < 0) { qr += uden; qf--; }
        }
    }

    return S1 * S20 + S10 * S2;
}

// ============================================================================
// SumTable — precompute partial sums for SumInter
// ============================================================================

static inline void sumTable(
    const Int8* f, long b, long a0, long q,
    long* F, long* rho, long* sigma
) {
    for (long n = -b; n < -b + q; n++)
        F[n] = f[n];
    for (long n = -b + q; n < b; n++)
        F[n] = F[n - q] + f[n];

    long a0mod = modl(a0, q);
    long r = modl(a0mod * ((Int128)(b - q)), q);
    for (long n = b - q; n < b; n++) {
        rho[r] = F[n];
        r += a0mod;
        if (r >= q) r -= q;
    }

    sigma[0] = 0;
    if (q > 1) sigma[1] = 0;
    for (long r2 = 1; r2 < q; r2++)
        sigma[r2 + 1] = sigma[r2] + rho[q - r2];
}

// ============================================================================
// FlCong — floor congruence helpers
// ============================================================================

static inline long flCong64(long n, long rModQ, long q) {
    long diff = n - rModQ;
    long r = diff % q;
    if (r < 0) r += q;
    return n - r;
}

// ============================================================================
// SumInter — sum over an interval with congruence constraint
// ============================================================================

__attribute__((always_inline))
static inline long sumInter(long* G, Int128 r, Interval128 J, long b, long q) {
    if (J.left > J.right) return 0;

    if (q == 1) {
        long r0 = (long)(J.left - 1);
        long r1 = (long)(J.right < (Int128)(b - 1) ? J.right : (Int128)(b - 1));
        if (r0 > r1 || r1 < -b) return 0;
        return (r0 >= -b) ? G[r1] - G[r0] : G[r1];
    }

    long rModQ = modl(r, q);
    long r0 = flCong64(J.left - 1, rModQ, q);
    long r1 = flCong64((long)(J.right < (Int128)(b - 1) ? J.right : (Int128)(b - 1)), rModQ, q);

    if (r0 > r1) return 0;
    if (r1 < -b) return 0;
    return (r0 >= -b) ? G[r1] - G[r0] : G[r1];
}

static inline long sumInterNonEmpty(long* G, Int128 r, Interval128 J, long b, long q) {
    return intervalEmpty(J) ? 0 : sumInter(G, r, J, b, q);
}

// ============================================================================
// QuadIneqZ — integer solutions of a quadratic inequality
// ============================================================================

template <bool EnableWide>
__attribute__((always_inline))
static inline Interval128 quadIneqZ(Int128 a, Int128 b, Int128 c) {
    Int128 Delta;
    bool deltaFits;
    if constexpr (EnableWide)
        deltaFits = quadDeltaFitsI128(a, b, c, Delta);
    else {
        Delta = b * b - 4 * a * c;
        deltaFits = true;
    }
    Interval128 J;

    if (deltaFits && Delta < 0) {
        J.left = 1; J.right = 0;  // empty interval
        return J;
    }

    Int128 Q;
    bool perfectSquare;
    if (deltaFits) {
        if constexpr (EnableWide) {
            uint256_t DeltaUnsigned = toU256(absU128(Delta));
            Q = isqrt256(DeltaUnsigned);
            uint256_t qWide = toU256(absU128(Q));
            perfectSquare = (qWide * qWide == DeltaUnsigned);
        } else {
            Q = isqrt128(Delta);
            perfectSquare = ((Int128)Q * Q == Delta);
        }
    } else {
        int256_t DeltaWide = toI256(b) * toI256(b) - int256_t(4) * toI256(a) * toI256(c);
        if (DeltaWide < 0) {
            J.left = 1; J.right = 0;  // empty interval
            return J;
        }
        uint256_t DeltaUnsigned = (uint256_t)DeltaWide;
        Q = isqrt256(DeltaUnsigned);
        uint256_t qWide = toU256(absU128(Q));
        perfectSquare = (qWide * qWide == DeltaUnsigned);
    }

    if (a < 0) {
        J.left  = -divf(-(b + Q), -2 * a);
        J.right =  divf(-(-b + Q), -2 * a);
    } else if (!perfectSquare) {
        J.left  = -divf(b + Q, 2 * a);
        J.right =  divf(-b + Q, 2 * a);
    } else {
        J.left  = divf(-b - Q + 2 * a, 2 * a);
        J.right = -divf(-(-b + Q - 2 * a), 2 * a);
    }

    return J;
}

// ============================================================================
// SpecialL2L1 — correction for q > 1 case
// ============================================================================

template <bool EnableWide>
__attribute__((always_inline))
static inline Int128 specialL2L1(
    long* G, Int128 xq, DiophApp appr, Int128 R0floor, Int128 r0, Int128 A2,
    Int128 ncirc, Int128 ancirc, Int128 m, long b,
    Int128 Qfloor, Int128 Qceil, long betsign, long delsign
) {
    Int128 gamma1, S, r;
    Interval128 JI, J;

    // Contribution from a_0(n-n_0)+r_0 = 1 mod q
    if (delsign > 0) {
        J.left = -b; J.right = -Qfloor - 1;
    } else if (delsign < 0) {
        J.left = -Qceil + 1; J.right = b - 1;
    } else if (betsign >= 0) {
        J.left = 1; J.right = 0;
    } else {
        J.left = -b; J.right = b - 1;
    }

    r = -r0 * appr.ainv;
    gamma1 = (-R0floor * appr.q - r0 + ancirc) * m;
    if (intervalEmpty(J)) {
        S = 0;
    } else {
        JI = quadIneqZ<EnableWide>(A2, gamma1, xq);
        JI.left -= ncirc; JI.right -= ncirc;
        JI.left  = (JI.left  > J.left)  ? JI.left  : J.left;
        JI.right = (JI.right < J.right) ? JI.right : J.right;
        if (intervalEmpty(JI)) {
            S = sumInter(G, r, J, b, appr.q);
        } else if (intervalEqual(JI, J)) {
            S = 0;
        } else {
            S = sumInter(G, r, J, b, appr.q);
            S -= sumInter(G, r, JI, b, appr.q);
        }
    }

    // Contribution from a_0(n-n_0)+r_0 = 0 mod q
    gamma1 -= m;
    r -= appr.ainv;
    J.left = -b; J.right = b - 1;
    if (!quadIntervalCovers<EnableWide>(A2, gamma1, xq, ncirc - b, ncirc + b - 1)) {
        JI = quadIneqZ<EnableWide>(A2, gamma1, xq);
        JI.left -= ncirc; JI.right -= ncirc;
        if (intervalEmpty(JI)) {
            S += sumInter(G, r, J, b, appr.q);
        } else if (!intervalFull(JI, b)) {
            S += sumInter(G, r, J, b, appr.q) - sumInter(G, r, JI, b, appr.q);
        }
    }

    return S;
}

// ============================================================================
// Special00 — correction for q == 1 case
// ============================================================================

template <bool EnableWide>
__attribute__((always_inline))
static inline Int128 special00(
    long* G, Int128 x, DiophApp appr, Int128 R0floor, Int128 r0,
    Int128 ncirc, Int128 m, long b,
    Int128 Qfloor, Int128 Qceil, long delsign
) {
    Int128 minbetdel, flank0, flank1, S0, S1, gamma1;
    Interval128 J, Iv[2];

    if (delsign > 0)
        minbetdel = -Qfloor - 1;
    else
        minbetdel = -Qceil + 1;

    if (!appr.a) {
        flank0 = divf(divf(x, m), (R0floor + r0)) - ncirc;
        flank1 = divf(divf(x, m), (R0floor + r0 + 1)) - ncirc;
        if (delsign > 0) {
            J.left = -b;
            J.right = flank0 < minbetdel ? flank0 : minbetdel;
            S0 = sumInter(G, 0, J, b, appr.q);
            J.left = minbetdel + 1;
            J.right = flank1;
            S1 = sumInter(G, 0, J, b, appr.q);
        } else if (delsign < 0) {
            J.left = minbetdel;
            J.right = flank0;
            S0 = sumInter(G, 0, J, b, appr.q);
            J.left = -b;
            J.right = flank1 < minbetdel - 1 ? flank1 : minbetdel - 1;
            S1 = sumInter(G, 0, J, b, appr.q);
        } else {
            S0 = 0;
            J.left = -b; J.right = flank1;
            S1 = sumInter(G, 0, J, b, appr.q);
        }
        return S0 + S1;
    } else {
        for (int j = 0; j <= 1; j++) {
            gamma1 = (-R0floor * appr.q - (r0 + j) + appr.a * ncirc) * m;
            Iv[j] = quadIneqZ<EnableWide>(-appr.a * m, gamma1, x * appr.q);
            Iv[j].left -= ncirc; Iv[j].right -= ncirc;
        }
        if (delsign > 0) {
            J.left = Iv[0].left;
            J.right = Iv[0].right < minbetdel ? Iv[0].right : minbetdel;
            S0 = sumInter(G, 0, J, b, appr.q);
            J.left = Iv[1].left > minbetdel + 1 ? Iv[1].left : minbetdel + 1;
            J.right = Iv[1].right;
            S1 = sumInter(G, 0, J, b, appr.q);
        } else if (delsign < 0) {
            J.left = Iv[0].left > minbetdel ? Iv[0].left : minbetdel;
            J.right = Iv[0].right;
            S0 = sumInter(G, 0, J, b, appr.q);
            J.left = Iv[1].left;
            J.right = Iv[1].right < minbetdel - 1 ? Iv[1].right : minbetdel - 1;
            S1 = sumInter(G, 0, J, b, appr.q);
        } else {
            S0 = 0;
            S1 = sumInter(G, 0, Iv[1], b, appr.q);
        }
        J.left = -b; J.right = b - 1;
        return sumInter(G, 0, J, b, appr.q) - (S0 + S1);
    }
}

// ============================================================================
// Special0A — boundary correction
// ============================================================================

__attribute__((always_inline))
static inline Int128 special0A(
    long* G, DiophApp appr, Int128 r0, long b,
    Int128 Qfloor, Int128 Qceil, long betasign, long delsign
) {
    Interval128 J;
    Int128 S;

    if (r0 > 0 && r0 < appr.q) {
        if (delsign) {
            if (delsign > 0) {
                J.left = -Qfloor; J.right = b - 1;
            } else {
                J.left = -b; J.right = -Qceil;
            }
        } else {
            if (betasign >= 0) {
                J.left = -b; J.right = b - 1;
            } else {
                J.left = 1; J.right = 0;
            }
        }
    } else {
        if (!delsign || !betasign) {
            J.left = 1; J.right = 0;
        } else if (betasign < 0) {
            if (delsign < 0) {
                Int128 residue = -r0 * appr.ainv;
                J.left = -b; J.right = -Qceil;
                S = sumInterNonEmpty(G, residue, J, b, appr.q);
                J.left = 1; J.right = b - 1;
                S += sumInterNonEmpty(G, residue, J, b, appr.q);
                return S;
            } else if (delsign > 0) {
                Int128 residue = -r0 * appr.ainv;
                J.left = -b; J.right = -1;
                S = sumInterNonEmpty(G, residue, J, b, appr.q);
                J.left = -Qfloor; J.right = b - 1;
                S += sumInterNonEmpty(G, residue, J, b, appr.q);
                return S;
            }
        } else {
            if (delsign > 0) {
                J.left = -Qfloor; J.right = -1;
            } else {
                J.left = 1; J.right = -Qceil;
            }
        }
    }
    if (intervalEmpty(J))
        return 0;
    Int128 residue = -r0 * appr.ainv;
    return sumInter(G, residue, J, b, appr.q);
}

// ============================================================================
// RaySum — sum along a ray (multiples of q)
// ============================================================================

static inline long raySum(const Int8* f, long q, long b, int deltasign) {
    long S = 0;
    if (deltasign < 0)
        for (long n = q; n <= b - 1; n += q)
            S += f[n];
    if (deltasign > 0)
        for (long n = -q; n >= -b; n -= q)
            S += f[n];
    return S;
}

// ============================================================================
// SumByLin — full double-sum via linearization + Diophantine correction
// ============================================================================

template <bool EnableWide>
static inline Int128 sumByLin(
    const Int8* f, const Int8* g,
    Int128 x, Int128 mcirc, Int128 ncirc, long a, long b
) {
    if (a == 0 || b == 0) return 0;

    Int128 S = linearSum(f, g, a, b, x, mcirc, ncirc);

    Int128 den1 = mcirc * ncirc;
    Int128 denm = mcirc * den1;
    Int128 denn = ncirc * den1;

    DiophApp appr = diophApp(-x, denn, 2 * b);

    Int128 denmq = appr.q * denm;
    Int128 delnum = -x * appr.q - appr.a * denn;
    long delsign = sgn(delnum);

    if ((long)g_sumByLinG.size() < 2 * b)
        g_sumByLinG.resize(2 * b);
    if ((long)g_sumByLinRho.size() < appr.q)
        g_sumByLinRho.resize((long)appr.q);
    if ((long)g_sumByLinSigma.size() < appr.q + 1)
        g_sumByLinSigma.resize((long)appr.q + 1);

    long* G = g_sumByLinG.data();
    long* rho = g_sumByLinRho.data();
    long* sigma = g_sumByLinSigma.data();

    long* Gbase = G + b;
    sumTable(g, b, appr.a, appr.q, Gbase, rho, sigma);

    long Z = raySum(g, appr.q, b, sgn(delnum));
    Int128 delnumMcirc = delnum * mcirc;

    // Precompute for 64-bit divfc fast path
    Int128 absDelnumMcirc = (delnumMcirc >= 0) ? delnumMcirc : -delnumMcirc;
    int delnumMcircNeg = (delnumMcirc < 0);
    Int128 absNcirc = (ncirc >= 0) ? ncirc : -ncirc;
    bool maxDividendFits = EnableWide ? mulFitsI128(denm, absNcirc) : true;
    Int128 maxDividend = maxDividendFits ? denm * absNcirc : kInt128MaxSafe;
    int divfcUse64 = maxDividendFits &&
                     (maxDividend <= (Int128)__LONG_MAX__) &&
                     (absDelnumMcirc <= (Int128)__LONG_MAX__);
    long absDelLong = divfcUse64 ? (long)absDelnumMcirc : 0;

    Int128 R0numInit = x * (mcirc + a);
    Int128 R0numModDenmQ = mod(R0numInit, denm) * appr.q;
    Int128 xModDenmQ = mod(x, denm) * appr.q;
    Int128 aNcirc = appr.a * ncirc;
    Int128 A2 = -appr.a * (mcirc - a);
    Int128 xQ = x * appr.q;

    // Incremental R0floor tracking
    Int128 xDivDenm = x / denm;
    Int128 xModDenm = x - xDivDenm * denm;
    Int128 R0floor = R0numInit / denm;
    Int128 R0rem = R0numInit - R0floor * denm;

    // Incremental r0 tracking
    Int128 xmtqDiv = xModDenmQ / denm;
    Int128 xmtqRem = xModDenmQ - xmtqDiv * denm;
    Int128 RSub = R0numModDenmQ % denm;
    Int128 r0Raw = (R0numModDenmQ - RSub) / denm;

    Int128 mPlusMcirc = mcirc - a;
    Int128 r0, betanum, betanumNcirc, Qfloor = 0, Qceil = 0, T;
    long betsign;
    bool divfcDenomWide = EnableWide ? (absDelnumMcirc > kInt128MaxSafe) : false;

    for (long m = -a; m < a; m++, mPlusMcirc++, A2 -= appr.a) {
        if (f[m]) {
            Int128 r0Corr = (2 * RSub >= denm) ? 1 : 0;
            r0 = r0Raw + r0Corr;
            betanum = RSub - r0Corr * denm;
            betsign = sgn(betanum);

            if (delnum) {
                bool betanumNcircFits = EnableWide ? mulFitsI128(betanum, ncirc) : true;
                if (divfcUse64) {
                    betanumNcirc = betanum * ncirc;
                    long a64 = (long)(delnumMcircNeg ? -betanumNcirc : betanumNcirc);
                    if (a64 >= 0) {
                        Qfloor = a64 / absDelLong;
                        Qceil = Qfloor + (long)(a64 != Qfloor * absDelLong);
                    } else {
                        Qceil = a64 / absDelLong;
                        Qfloor = Qceil - (long)(a64 != Qceil * absDelLong);
                    }
                } else if (betanumNcircFits && !divfcDenomWide) {
                    betanumNcirc = betanum * ncirc;
                    Int128 signedDividend = delnumMcircNeg ? -betanumNcirc : betanumNcirc;
                    divfcPos(signedDividend, absDelnumMcirc, Qfloor, Qceil);
                } else {
                    int256_t signedDividendWide = toI256(betanum) * toI256(ncirc);
                    if (delnumMcircNeg) signedDividendWide = -signedDividendWide;
                    divfcPosWide(signedDividendWide, absDelnumMcirc, Qfloor, Qceil);
                }
            }

            T = sigma[r0];
            if (appr.q > 1)
                T += specialL2L1<EnableWide>(Gbase, xQ, appr, R0floor, r0, A2,
                                 ncirc, aNcirc, mPlusMcirc, b,
                                 Qfloor, Qceil, betsign, delsign);
            else
                T += special00<EnableWide>(Gbase, x, appr, R0floor, r0,
                               ncirc, mPlusMcirc, b, Qfloor, Qceil, delsign);

            T += special0A(Gbase, appr, r0, b, Qfloor, Qceil, betsign, delsign);
            if (r0 > 0 && r0 < appr.q)
                T += Z;
            S += f[m] * T;
        }

        // Incremental updates
        RSub -= xmtqRem;
        r0Raw -= xmtqDiv;
        if (RSub < 0) { RSub += denm; r0Raw--; }
        if (r0Raw < 0) r0Raw += appr.q;

        R0rem -= xModDenm;
        R0floor -= xDivDenm;
        if (R0rem < 0) { R0rem += denm; R0floor--; }
    }

    return S;
}
