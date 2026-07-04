#pragma once

// ============================================================================
// gpu_s2.h — GPU (Metal) offload of MertensHurst's 64-bit S2 updates.
//
// Pure-C++ interface (the Metal implementation lives in gpu_s2.mm) so that
// MertensHurst.cpp can use it without becoming Objective-C++.
//
// Usage per sieve block, from a serial region:
//   gpus2_begin_segment(g, MuP, len, L1, partialValues.data());
//   gpus2_add_task(...) for every 64-bit S2 task of the block
//        (replicates update_S2<Half>'s mode-range decomposition host-side);
//   gpus2_dispatch_async(g);          // returns immediately; GPU works
//   ... CPU runs S1 (independent of S2) ...
//   gpus2_wait_apply(g, partialValues.data());   // partialValues[i] -= sums
//
// The engine streams work in bounded WAVES (~4M items each) instead of
// materializing a block's full item list: at n = 1e18 a single block can
// produce ~2e9 items (~48 GB), which is what motivated this. Waves recycle
// fixed buffers, so GPU/host memory stays constant in n; completed waves
// are applied to partialValues opportunistically (hence the pointer at
// begin_block — all engine calls happen from the same serial region, so
// there is no concurrency with the S1 writers).
//
// Scope: 64-bit args only (n <= ~1e18). Their iteration indices satisfy
// k <= nu_i = c*sqrt(n) < 2^32, which the implementation asserts.
// 128-bit args stay on the CPU.
// ============================================================================

#include <cstdint>

class GpuS2;

// Returns nullptr if no Metal device is available.
GpuS2* gpus2_create();
void   gpus2_destroy(GpuS2* g);

void gpus2_begin_segment(GpuS2* g, const int8_t* mu, uint64_t len, uint64_t L1,
                       int64_t* partialValues);

// One S2 task: subtracts sum_{k in [x1,x2], (k,6)=1} mu(k)*S2_term(n/k)
// from partialValues[outIdx], with the mode decomposition given by nu, nu2
// and the Half flag (true: 8-range odd/even split; false: 4-range split) —
// exactly mirroring update_S2<Half> in S2.h.
void gpus2_add_task(GpuS2* g, uint64_t n, uint64_t x1, uint64_t x2,
                    uint64_t nu, uint64_t nu2, bool half, uint32_t outIdx);

void gpus2_dispatch_async(GpuS2* g);

// Blocks until the GPU finishes, validates command-buffer status, and applies
// all remaining wave results. Aborts on GPU errors.
void gpus2_wait_apply(GpuS2* g, int64_t* partialValues);
