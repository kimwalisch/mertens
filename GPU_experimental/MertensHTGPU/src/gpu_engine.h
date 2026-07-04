#pragma once
// ============================================================================
// gpu_engine.h — Metal compute engine for MertensHTGPU (CPU-facing interface).
//
// Pure C++ surface (no Metal types) so the rest of the code stays platform
// agnostic. The implementation (gpu_engine.mm) enumerates *all* Metal devices
// (MTLCopyAllDevices) and spreads work across them: 1 GPU on Apple Silicon
// (M2 Max / M3 Ultra), 2 GPUs on the 2019 Mac Pro (dual AMD). No CUDA.
//
// Everything here is compiled only when USE_GPU=1.
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace gpu {

// ---- Runtime toggles (env vars, read once; all default off) ----------------
// One GPU=1 binary; flip these per run to A/B across machines.
bool useFacsum();   // MHT_GPU_FACSUM=1  : SArr FacToSumMu on GPU
bool useBrute();    // MHT_GPU_BRUTE=1   : LargeFree bruteDoubleSum on GPU
bool usePipeline(); // MHT_GPU_PIPELINE=1: overlap GPU FacToSumMu w/ CPU fillFactBlock
bool useOverlap();  // MHT_GPU_OVERLAP=1 : LargeFree(GPU brute) || LargeNonFree(CPU)
bool zeroCopy();    // default on; MHT_GPU_NOZEROCOPY=1 forces the copy path

// ---- Zero-copy shared allocation -------------------------------------------
// Returns a pointer into an engine-owned shared Metal buffer (single unified
// GPU only; nullptr otherwise -> caller should malloc and use the copy path).
// The CPU writes into it directly and the GPU reads it with no copy.
void* sharedAlloc(size_t bytes);
void  sharedFree(void* p);

// ---- Device discovery ------------------------------------------------------
int         numDevices();          // 0 if no Metal devices / not a GPU build
const char* deviceName(int i);     // name of device i (i in [0,numDevices))
void        printDevices();        // one-line summary of every device

// ---- FacToSumMu map (SArr inner kernel) ------------------------------------
// For each i in [0,count): r = r0+i, val[i] = FacToSumMu(fun[i], floor(x/r),
// sqfprod[i]). `fun` points at a PrimFact[count] array ({plist,pmark} u64
// pairs). Work is split across all Metal devices. Requires x <= UINT64_MAX.
void facsumMap(const void* fun, const uint64_t* sqfprod,
               uint64_t x, int64_t r0, long count, int64_t* valOut);

// Async variant for the pipeline: dispatch returns immediately (inputs are
// copied), letting the caller refill fun/sqfprod for the next block while the
// GPU runs; facsumWait() blocks and copies results back. Depth 1.
void facsumDispatch(const void* fun, const uint64_t* sqfprod,
                    uint64_t x, int64_t r0, long count);
void facsumWait(int64_t* valOut);

// ---- bruteDoubleSum map (LargeFree) ----------------------------------------
// out[i] = mu(m_i) * sum_j nzG[j]*floor(lxm[i]/nzN[j]) for i in [0,mCount), where
// lxm[i] = floor(x/m_i) is precomputed by the caller. x is unconstrained; only
// each lxm[i] < 2^60 and all nzN < 2^32 are required (caller keeps larger-
// quotient m on the CPU). nzN/nzG (length nnz) shared; work split across devices.
void bruteMap(const signed char* muA, const uint64_t* lxm, long mCount,
              const uint32_t* nzN, const signed char* nzG, long nnz,
              int64_t* out);

// ---- Plumbing self-test ----------------------------------------------------
// Computes a deterministic 64-bit function f(i) over i in [0,N), split across
// all devices, and checks every element against the CPU. Returns true on a
// bit-exact match everywhere. Used to validate the Metal path before wiring
// the real kernels.
bool selftest(long N);

} // namespace gpu
