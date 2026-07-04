#include "MertensHT.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/time.h>

// ============================================================================
// Input parsing
// ============================================================================

// Parse a nonnegative integer in plain decimal ("2000000") or scientific
// notation ("2e6", "2.5e21"). Returns false if the input is malformed, not
// an exact integer, or too large for UInt128.
static bool parseNumber(const char* s, UInt128& out) {
    UInt128 mant = 0;
    int frac = -1;
    long exp10 = 0;

    if (*s == '\0') return false;
    for (; *s && *s != 'e' && *s != 'E'; ++s) {
        if (*s == '.') {
            if (frac >= 0) return false;
            frac = 0;
        } else if (*s >= '0' && *s <= '9') {
            if (mant > ((UInt128)-1 - 9) / 10) return false;
            mant = mant * 10 + (*s - '0');
            if (frac >= 0) ++frac;
        } else {
            return false;
        }
    }
    if (*s != '\0') {
        if (*(++s) == '\0') return false;
        for (; *s; ++s) {
            if (*s < '0' || *s > '9') return false;
            exp10 = exp10 * 10 + (*s - '0');
            if (exp10 > 60) return false;
        }
    }
    if (frac > 0) exp10 -= frac;
    if (exp10 < 0) return false;
    while (exp10-- > 0) {
        if (mant > ((UInt128)-1) / 10) return false;
        mant *= 10;
    }
    out = mant;
    return true;
}

// Decimal digits of a UInt128, for printing.
static std::string toString(UInt128 v) {
    if (v == 0) return "0";
    std::string s;
    for (; v > 0; v /= 10) s.insert(s.begin(), (char)('0' + (int)(v % 10)));
    return s;
}

static void printUsage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <n> [--profile] [--vdiv <v>]"
                 " [--C <c>] [--D <d>] [--ntasks <n>] [--sarr-cap <bytes>]" << std::endl;
    std::cerr << "  n           computes M(n) for n >= 10^6;"
                 " scientific notation accepted (1e22)" << std::endl;
    std::cerr << "  --profile   print timing breakdown" << std::endl;
    std::cerr << "  --vdiv      v divisor (default: 2.0)" << std::endl;
    std::cerr << "  --C         crossover parameter (default: 42)" << std::endl;
    std::cerr << "  --D         tile size (default: 16)" << std::endl;
    std::cerr << "  --ntasks    task multiplier (default: 32)" << std::endl;
    std::cerr << "  --sarr-cap  memory cap in bytes for the SArr working arrays"
                 " (default: 12000000000)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    bool profile = false;
    double vdiv = 2.0;
    Int64 paramC = 42;
    Int64 paramD = 16;
    Int64 ntasksMul = 32;
    Int64 sarrCapBytes = 12000000000LL;
    const char* nstr = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--profile") == 0 || std::strcmp(argv[i], "-p") == 0) {
            profile = true;
        } else if (std::strcmp(argv[i], "--vdiv") == 0 && i + 1 < argc) {
            vdiv = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--C") == 0 && i + 1 < argc) {
            paramC = std::atol(argv[++i]);
        } else if (std::strcmp(argv[i], "--D") == 0 && i + 1 < argc) {
            paramD = std::atol(argv[++i]);
        } else if (std::strcmp(argv[i], "--ntasks") == 0 && i + 1 < argc) {
            ntasksMul = std::atol(argv[++i]);
        } else if (std::strcmp(argv[i], "--sarr-cap") == 0 && i + 1 < argc) {
            UInt128 capParsed;
            if (!parseNumber(argv[++i], capParsed) || (capParsed >> 62) != 0) {
                std::cerr << "Error: could not parse --sarr-cap value." << std::endl;
                return 1;
            }
            sarrCapBytes = (Int64)capParsed;
            if (sarrCapBytes < 32) {
                std::cerr << "Error: --sarr-cap must be at least 32 bytes." << std::endl;
                return 1;
            }
        } else {
            nstr = argv[i];
        }
    }

    if (!nstr) {
        std::cerr << "Error: no value for n provided." << std::endl;
        return 1;
    }

    UInt128 xParsed;
    if (!parseNumber(nstr, xParsed)) {
        std::cerr << "Error: could not parse n = '" << nstr << "'." << std::endl;
        return 1;
    }
    if (xParsed < 1000000) {
        std::cerr << "Error: n must be at least 10^6."
                     " For small n use MertensHurst or the sieve." << std::endl;
        return 1;
    }
    if ((xParsed >> 127) != 0) {
        std::cerr << "Error: n is too large." << std::endl;
        return 1;
    }
    Int128 x = (Int128)xParsed;

    struct timeval start, end;
    gettimeofday(&start, NULL);
    Int64 M = MertensHT(x, profile, vdiv, paramC, paramD, ntasksMul, sarrCapBytes);
    gettimeofday(&end, NULL);

    double t = (Int64)(end.tv_sec) - (Int64)(start.tv_sec)
             + (end.tv_usec - start.tv_usec) / 1000000.0;

    std::cout << "M(" << toString(xParsed) << ") = " << M << " in " << t << " seconds" << std::endl;
    return 0;
}
