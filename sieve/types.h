#pragma once

// ============================================================================
// types.h — Fundamental type aliases, enums, and constants for the Mertens
//           function computation.
// ============================================================================

#include <cstdint>

using UInt128 = unsigned __int128;
using Int128  = __int128_t;

using UInt64 = uint64_t;
using Int64  = int64_t;

using UInt32 = uint32_t;
using Int32  = int32_t;

using UInt16 = uint16_t;
using Int16  = int16_t;

using UInt8 = uint8_t;
using Int8  = int8_t;

// Controls which parity of summation indices to include in S2 updates.
enum class ParityMode : UInt8 { All, Even, Odd };
