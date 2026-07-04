// ============================================================================
// gpu_engine.mm — Metal compute engine implementation.
//
// Objective-C++ (compiled with -fobjc-arc). Enumerates all Metal devices and
// spreads work across them. Kernels are embedded as MSL source strings and
// compiled at runtime (no .metallib build step).
// ============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu_engine.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <sys/time.h>

static double gpu_wall() {
    struct timeval t; gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec / 1e6;
}
static bool gpu_timing() { static int v = getenv("GPU_TIMING") ? 1 : 0; return v; }

// A command buffer that completes with any status other than Completed left a
// garbage output buffer. For an exact integer computation we must NOT silently
// use it — abort loudly so a transient GPU error can never become a wrong M(x).
static void gpu_check(id<MTLCommandBuffer> cb, const char* where) {
    if (cb.status != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "[gpu] FATAL: command buffer failed in %s (status=%ld): %s\n",
                     where, (long)cb.status,
                     cb.error ? [[cb.error localizedDescription] UTF8String] : "no error info");
        std::abort();
    }
}

// ----------------------------------------------------------------------------
// Embedded Metal shader library (kernels added as we go).
// ----------------------------------------------------------------------------
static const char* kShaderSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// Deterministic 64-bit map used by the plumbing self-test. Must match the
// CPU reference f(i) in selftest() exactly.
kernel void selftest_map(device long*       out  [[buffer(0)]],
                         constant ulong&     base [[buffer(1)]],
                         uint                gid  [[thread_position_in_grid]])
{
    ulong i = base + (ulong)gid;
    out[gid] = (long)(i*i - 3UL*i + 7UL);
}

// ---- FacToSumMu (SArr inner kernel) ----------------------------------------
// Bit-layout of PrimFact must match the C++ struct: { ulong plist; ulong pmark; }.
struct PrimFact { ulong plist; ulong pmark; };

// `base` lets a dispatch cover sub-batch [base, base+count) of the range, so a
// large block is split into watchdog-safe command buffers. r = r0 + base + gid.
struct FacParams { ulong x; long r0; uint base; uint count; };

// Iterative form of subFacToSumMu(f, m=1, mc=1, a, n) from Factorization.h,
// which equals FacToSumMu(f, a, n). Explicit DFS stack replaces the recursion;
// the base-case order (m>a -> 0; no primes -> +1; mc>=n -> 0) is identical.
inline long facToSumMu(ulong pmark, ulong plist, ulong a, ulong nsqf)
{
    struct Frame { ulong pmark, plist, m, mc; long sign; };
    Frame st[64];
    int sp = 0;
    st[sp++] = (Frame){ pmark, plist, 1UL, 1UL, 1L };
    long result = 0;
    while (sp > 0) {
        Frame f = st[--sp];
        if (f.m > a)        continue;            // m > a  -> 0
        if (f.pmark == 0) { result += f.sign; continue; }  // no primes left -> +1
        if (f.mc >= nsqf)   continue;            // mc >= n -> 0
        uint  k  = ctz(f.pmark) + 1;
        ulong nm = f.pmark >> k;
        ulong p  = (1UL << k) | (f.plist & ~((~0UL) << k));
        ulong nl = f.plist >> k;
        // left child: keep p on the mc side (+sign); right child: m side (-sign)
        st[sp++] = (Frame){ nm, nl, f.m,     f.mc * p, f.sign };
        st[sp++] = (Frame){ nm, nl, f.m * p, f.mc,    -f.sign };
    }
    return result;
}

kernel void facsum_map(device const PrimFact* fun     [[buffer(0)]],
                       device const ulong*    sqfprod [[buffer(1)]],
                       constant FacParams&    P       [[buffer(2)]],
                       device long*           val     [[buffer(3)]],
                       uint                   gid     [[thread_position_in_grid]])
{
    if (gid >= P.count) return;
    uint idx = P.base + gid;
    ulong r = (ulong)P.r0 + (ulong)idx;
    ulong a = P.x / r;                                   // floor(x/r), 64-bit
    val[idx] = facToSumMu(fun[idx].pmark, fun[idx].plist, a, sqfprod[idx]);
}

// ---- bruteDoubleSum (LargeFree) --------------------------------------------
// Exact n/d for n < 2^60, d < 2^32 (float-seeded). Reused verbatim from the
// validated MertensHurstGPU/gpu_s2.mm divider — native 64-bit div is ~6.5x
// worse on Apple GPUs.
inline ulong div64_mertens(ulong n, uint d) {
    const float fd = (float)d;
    ulong q = (ulong)((float)n / fd);
    long r = (long)(n - q * (ulong)d);
    long qc = (long)((float)r / fd);
    q += (ulong)qc; r -= qc * (long)d;
    qc = (long)((float)r / fd);
    q += (ulong)qc; r -= qc * (long)d;
    while (r < 0)        { --q; r += (long)d; }
    while (r >= (long)d) { ++q; r -= (long)d; }
    return q;
}

// `base` lets a dispatch cover a sub-batch [base, base+mCount) of the device's
// range, so big items are split into watchdog-safe command buffers.
// lxm[idx] = floor(x/m) is precomputed on the host (so x may exceed 2^60 — only
// the per-m quotient must be < 2^60); the few m with x/m >= 2^60 are kept on the
// CPU. muA/lxm/out are indexed at base+gid.
struct BruteParams { uint base; uint mCount; uint nnz; uint pad; };

// One thread per m: Sm = sum_i nzG[i]*floor(lxm/nzN[i]), out[idx] = mu(m)*Sm.
// Matches the 64-bit path of bruteDoubleSum() in MertensHT.cpp.
kernel void brute_map(device const char*  muA   [[buffer(0)]],
                      device const ulong* lxm   [[buffer(1)]],
                      device const uint*  nzN   [[buffer(2)]],
                      device const char*  nzG   [[buffer(3)]],
                      constant BruteParams& P    [[buffer(4)]],
                      device long*        out   [[buffer(5)]],
                      uint                gid   [[thread_position_in_grid]])
{
    if (gid >= P.mCount) return;
    uint idx = P.base + gid;
    long fm = (long)muA[idx];
    if (fm == 0) { out[idx] = 0; return; }
    ulong q = lxm[idx];                                  // floor(x/m), < 2^60
    long s = 0;
    for (uint i = 0; i < P.nnz; i++)
        s += (long)nzG[i] * (long)div64_mertens(q, nzN[i]);
    out[idx] = fm * s;
}
)METAL";

// ----------------------------------------------------------------------------
// Per-device context, lazily initialized once.
// ----------------------------------------------------------------------------
namespace {

struct Device {
    id<MTLDevice>              dev = nil;
    id<MTLCommandQueue>        queue = nil;
    id<MTLLibrary>             lib = nil;
    id<MTLComputePipelineState> psoSelftest = nil;
    id<MTLComputePipelineState> psoFacsum = nil;
    id<MTLComputePipelineState> psoBrute = nil;
    std::string                name;
};

// Build a compute pipeline for the named kernel on device d.
static id<MTLComputePipelineState> makePso(id<MTLDevice> d, id<MTLLibrary> lib,
                                           const char* fnName, const char* devName) {
    NSError* err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fnName]];
    id<MTLComputePipelineState> pso = [d newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) std::fprintf(stderr, "[gpu] pipeline %s failed on %s: %s\n", fnName, devName,
                           err ? [[err localizedDescription] UTF8String] : "?");
    return pso;
}

std::vector<Device> g_devices;
bool                g_init = false;

void ensureInit() {
    if (g_init) return;
    g_init = true;

    NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
    for (id<MTLDevice> d in devs) {
        Device ctx;
        ctx.dev = d;
        ctx.name = std::string([[d name] UTF8String]);
        ctx.queue = [d newCommandQueue];

        NSError* err = nil;
        ctx.lib = [d newLibraryWithSource:[NSString stringWithUTF8String:kShaderSrc]
                                  options:nil
                                    error:&err];
        if (!ctx.lib) {
            std::fprintf(stderr, "[gpu] shader compile failed on %s: %s\n",
                         ctx.name.c_str(),
                         err ? [[err localizedDescription] UTF8String] : "?");
            continue;
        }
        ctx.psoSelftest = makePso(d, ctx.lib, "selftest_map", ctx.name.c_str());
        ctx.psoFacsum   = makePso(d, ctx.lib, "facsum_map",   ctx.name.c_str());
        ctx.psoBrute    = makePso(d, ctx.lib, "brute_map",    ctx.name.c_str());
        if (!ctx.psoSelftest || !ctx.psoFacsum || !ctx.psoBrute) continue;
        g_devices.push_back(ctx);
    }
}

} // namespace

// ----------------------------------------------------------------------------
// Public interface
// ----------------------------------------------------------------------------
namespace gpu {

bool useFacsum()   { static int v = getenv("MHT_GPU_FACSUM")   ? 1 : 0; return v; }
bool useBrute()    { static int v = getenv("MHT_GPU_BRUTE")    ? 1 : 0; return v; }
bool usePipeline() { static int v = getenv("MHT_GPU_PIPELINE") ? 1 : 0; return v; }
bool useOverlap()  { static int v = getenv("MHT_GPU_OVERLAP")  ? 1 : 0; return v; }
bool zeroCopy()    { static int v = getenv("MHT_GPU_NOZEROCOPY") ? 0 : 1; return v; }

int numDevices() {
    ensureInit();
    return (int)g_devices.size();
}

// Registry of engine-owned shared buffers, keyed by their .contents pointer.
// Used for true zero-copy on a single unified-memory GPU: the CPU producer
// writes straight into GPU-visible memory, so no copy/readback is needed.
static std::unordered_map<void*, id<MTLBuffer>> g_sharedBufs;

void* sharedAlloc(size_t bytes) {
    ensureInit();
    if (!zeroCopy() || g_devices.size() != 1) return nullptr;        // multi-GPU/discrete -> copy path
    if (![g_devices[0].dev hasUnifiedMemory]) return nullptr;
    id<MTLBuffer> b = [g_devices[0].dev newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
    if (!b) return nullptr;
    void* p = [b contents];
    g_sharedBufs[p] = b;
    return p;
}

void sharedFree(void* p) {
    if (!p) return;
    g_sharedBufs.erase(p);   // ARC releases the MTLBuffer
}

static id<MTLBuffer> sharedBufFor(const void* p) {
    auto it = g_sharedBufs.find(const_cast<void*>(p));
    return it == g_sharedBufs.end() ? nil : it->second;
}

const char* deviceName(int i) {
    ensureInit();
    if (i < 0 || i >= (int)g_devices.size()) return "?";
    return g_devices[i].name.c_str();
}

void printDevices() {
    ensureInit();
    std::printf("[gpu] %d Metal device(s):\n", (int)g_devices.size());
    for (size_t i = 0; i < g_devices.size(); i++) {
        id<MTLDevice> d = g_devices[i].dev;
        std::printf("  [%zu] %s  (unified=%d, maxThreadsPerTG=%lu, maxBuf=%.1f GB)\n",
                    i, g_devices[i].name.c_str(),
                    (int)[d hasUnifiedMemory],
                    (unsigned long)g_devices[i].psoSelftest.maxTotalThreadsPerThreadgroup,
                    (double)[d maxBufferLength] / (1024.0*1024.0*1024.0));
    }
}

bool selftest(long N) {
    ensureInit();
    int nd = (int)g_devices.size();
    if (nd == 0) { std::fprintf(stderr, "[gpu] no devices\n"); return false; }

    // Split [0,N) across devices in contiguous chunks.
    std::vector<long> lo(nd), hi(nd);
    long per = (N + nd - 1) / nd;
    for (int d = 0; d < nd; d++) {
        lo[d] = (long)d * per;
        hi[d] = std::min(N, lo[d] + per);
        if (lo[d] > N) lo[d] = N;
    }

    // Allocate one output buffer per device (shared storage works on both
    // unified-memory Apple GPUs and discrete AMD GPUs).
    std::vector<id<MTLBuffer>> outBuf(nd, nil);
    std::vector<id<MTLCommandBuffer>> cmd(nd, nil);

    for (int d = 0; d < nd; d++) {
        long cnt = hi[d] - lo[d];
        if (cnt <= 0) continue;
        Device& C = g_devices[d];
        outBuf[d] = [C.dev newBufferWithLength:cnt * sizeof(long)
                                       options:MTLResourceStorageModeShared];
        uint64_t base = (uint64_t)lo[d];
        id<MTLBuffer> baseBuf = [C.dev newBufferWithBytes:&base
                                                   length:sizeof(uint64_t)
                                                  options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [C.queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:C.psoSelftest];
        [enc setBuffer:outBuf[d] offset:0 atIndex:0];
        [enc setBuffer:baseBuf  offset:0 atIndex:1];

        NSUInteger tg = C.psoSelftest.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        [enc dispatchThreads:MTLSizeMake(cnt, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];               // launch all devices, then wait — concurrent
        cmd[d] = cb;
    }

    // Wait and verify against the CPU.
    bool ok = true;
    long mism = 0;
    for (int d = 0; d < nd; d++) {
        if (!cmd[d]) continue;
        [cmd[d] waitUntilCompleted];
        gpu_check(cmd[d], "selftest");
        long cnt = hi[d] - lo[d];
        const long* got = (const long*)[outBuf[d] contents];
        for (long t = 0; t < cnt; t++) {
            unsigned long i = (unsigned long)(lo[d] + t);
            long ref = (long)(i*i - 3UL*i + 7UL);
            if (got[t] != ref) { ok = false; if (mism < 3)
                std::fprintf(stderr, "[gpu] mismatch i=%lu got=%ld ref=%ld\n", i, got[t], ref);
                mism++; }
        }
    }
    if (ok) std::printf("[gpu] selftest OK over [0,%ld) across %d device(s)\n", N, nd);
    else    std::printf("[gpu] selftest FAILED (%ld mismatches)\n", mism);
    return ok;
}

// In-flight FacToSumMu dispatch state (depth 1) for the async pipeline path.
namespace {
struct FacInFlight {
    std::vector<id<MTLBuffer>> vbuf;
    std::vector<std::vector<id<MTLCommandBuffer>>> cmd;   // per device, batched
    std::vector<long> lo, hi;
    bool active = false;
};
FacInFlight g_fac;

// FacToSumMu per-thread work is light but variable; bound threads per dispatch
// so a large block stays under the GPU watchdog.
static const long kFacBatch = 4L << 20;   // 4M threads / command buffer
}

// Launch the FacToSumMu map across all devices WITHOUT waiting. Input buffers
// are copied into Metal storage here, so the caller may reuse fun/sqfprod as
// soon as this returns (that overlap is the point). Pair with facsumWait().
void facsumDispatch(const void* fun, const uint64_t* sqfprod,
                    uint64_t x, int64_t r0, long count) {
    ensureInit();
    int nd = (int)g_devices.size();
    g_fac = FacInFlight{};
    g_fac.vbuf.assign(nd, nil);
    g_fac.cmd.assign(nd, {});
    g_fac.lo.assign(nd, 0);
    g_fac.hi.assign(nd, 0);
    if (nd == 0 || count <= 0) { g_fac.active = (count > 0); return; }
    g_fac.active = true;

    struct FacParams { uint64_t x; int64_t r0; uint32_t base; uint32_t count; };
    struct HostPrimFact { uint64_t plist; uint64_t pmark; };  // matches C++ PrimFact
    const HostPrimFact* funArr = (const HostPrimFact*)fun;
    long per = (count + nd - 1) / nd;

    for (int d = 0; d < nd; d++) {
        g_fac.lo[d] = (long)d * per;
        g_fac.hi[d] = std::min(count, g_fac.lo[d] + per);
        if (g_fac.lo[d] >= g_fac.hi[d]) continue;
        long cnt = g_fac.hi[d] - g_fac.lo[d];
        Device& C = g_devices[d];

        id<MTLBuffer> funBuf = [C.dev newBufferWithBytes:funArr + g_fac.lo[d]
                                                  length:cnt * sizeof(HostPrimFact)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> sqfBuf = [C.dev newBufferWithBytes:sqfprod + g_fac.lo[d]
                                                  length:cnt * sizeof(uint64_t)
                                                 options:MTLResourceStorageModeShared];
        g_fac.vbuf[d] = [C.dev newBufferWithLength:cnt * sizeof(int64_t)
                                           options:MTLResourceStorageModeShared];
        NSUInteger tg = C.psoFacsum.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        for (long bo = 0; bo < cnt; bo += kFacBatch) {
            long bs = std::min(kFacBatch, cnt - bo);
            FacParams fp{ x, r0 + g_fac.lo[d], (uint32_t)bo, (uint32_t)bs };
            id<MTLCommandBuffer> cb = [C.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:C.psoFacsum];
            [enc setBuffer:funBuf offset:0 atIndex:0];
            [enc setBuffer:sqfBuf offset:0 atIndex:1];
            [enc setBytes:&fp length:sizeof(fp) atIndex:2];
            [enc setBuffer:g_fac.vbuf[d] offset:0 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(bs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            g_fac.cmd[d].push_back(cb);
        }
    }
}

// Wait for the outstanding facsumDispatch and copy results into valOut.
void facsumWait(int64_t* valOut) {
    if (!g_fac.active) return;
    for (size_t d = 0; d < g_fac.cmd.size(); d++) {
        if (g_fac.cmd[d].empty()) continue;
        for (id<MTLCommandBuffer> cb : g_fac.cmd[d]) {
            [cb waitUntilCompleted];
            gpu_check(cb, "facsumWait");
        }
        long cnt = g_fac.hi[d] - g_fac.lo[d];
        memcpy(valOut + g_fac.lo[d], [g_fac.vbuf[d] contents], cnt * sizeof(int64_t));
    }
    g_fac = FacInFlight{};
}

// Synchronous FacToSumMu map. If fun/sqfprod/valOut are all engine-owned shared
// buffers (single unified GPU, zero-copy enabled), the kernel reads/writes them
// in place — no copies, no readback. Otherwise falls back to dispatch+copy+wait.
void facsumMap(const void* fun, const uint64_t* sqfprod,
               uint64_t x, int64_t r0, long count, int64_t* valOut) {
    ensureInit();
    if (g_devices.size() == 1 && count > 0) {
        id<MTLBuffer> funBuf = sharedBufFor(fun);
        id<MTLBuffer> sqfBuf = sharedBufFor(sqfprod);
        id<MTLBuffer> outBuf = sharedBufFor(valOut);
        if (funBuf && sqfBuf && outBuf) {
            struct FacParams { uint64_t x; int64_t r0; uint32_t base; uint32_t count; };
            Device& C = g_devices[0];
            NSUInteger tg = C.psoFacsum.maxTotalThreadsPerThreadgroup;
            if (tg > 256) tg = 256;
            // Batch into watchdog-safe command buffers (zero-copy in/out).
            for (long bo = 0; bo < count; bo += kFacBatch) {
                long bs = std::min(kFacBatch, count - bo);
                FacParams fp{ x, r0, (uint32_t)bo, (uint32_t)bs };
                id<MTLCommandBuffer> cb = [C.queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:C.psoFacsum];
                [enc setBuffer:funBuf offset:0 atIndex:0];
                [enc setBuffer:sqfBuf offset:0 atIndex:1];
                [enc setBytes:&fp length:sizeof(fp) atIndex:2];
                [enc setBuffer:outBuf offset:0 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(bs, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];   // valOut is the shared buffer; nothing to copy
                gpu_check(cb, "facsumMap(zero-copy)");
            }
            return;
        }
    }
    facsumDispatch(fun, sqfprod, x, r0, count);
    facsumWait(valOut);
}

// bruteDoubleSum map: out[i] = mu(m_i) * sum_j nzG[j]*floor(lxm[i]/nzN[j]),
// for i in [0,mCount). lxm[i] = floor(x/m_i) is precomputed by the caller, so x
// itself is unconstrained; only each lxm[i] < 2^60 and all nzN < 2^32 are
// required (the caller routes the few larger-quotient m to the CPU). nzN/nzG
// (length nnz) are replicated per device; the m-range and output are split.
void bruteMap(const signed char* muA, const uint64_t* lxm, long mCount,
              const uint32_t* nzN, const signed char* nzG, long nnz,
              int64_t* out) {
    ensureInit();
    int nd = (int)g_devices.size();
    if (nd == 0 || mCount <= 0) return;

    struct BruteParams { uint32_t base; uint32_t mCount; uint32_t nnz; uint32_t pad; };
    std::vector<id<MTLBuffer>> obuf(nd, nil);
    std::vector<std::vector<id<MTLCommandBuffer>>> cmds(nd);
    std::vector<long> lo(nd), hi(nd);
    long per = (mCount + nd - 1) / nd;

    // Watchdog safety: cap GPU work per command buffer. Each thread does ~nnz
    // divisions, so bound threads-per-batch by an inner-iteration budget.
    const long kIterBudget = 32L << 20;       // ~33M inner iterations / dispatch
    long batch = kIterBudget / std::max<long>(nnz, 1);
    batch = std::min<long>(std::max<long>(batch, 4096), 8L << 20);

    for (int d = 0; d < nd; d++) {
        lo[d] = (long)d * per;
        hi[d] = std::min(mCount, lo[d] + per);
        if (lo[d] >= hi[d]) continue;
        long cnt = hi[d] - lo[d];
        Device& C = g_devices[d];

        id<MTLBuffer> muBuf = [C.dev newBufferWithBytes:muA + lo[d]
                                                 length:cnt * sizeof(char)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> lxmBuf = [C.dev newBufferWithBytes:lxm + lo[d]
                                                 length:cnt * sizeof(uint64_t)
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> nBuf = [C.dev newBufferWithBytes:nzN
                                                length:nnz * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> gBuf = [C.dev newBufferWithBytes:nzG
                                                length:nnz * sizeof(char)
                                               options:MTLResourceStorageModeShared];
        obuf[d] = [C.dev newBufferWithLength:cnt * sizeof(int64_t)
                                     options:MTLResourceStorageModeShared];

        NSUInteger tg = C.psoBrute.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;
        for (long bo = 0; bo < cnt; bo += batch) {
            long bs = std::min(batch, cnt - bo);
            BruteParams bp{ (uint32_t)bo, (uint32_t)bs, (uint32_t)nnz, 0 };
            id<MTLCommandBuffer> cb = [C.queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:C.psoBrute];
            [enc setBuffer:muBuf offset:0 atIndex:0];
            [enc setBuffer:lxmBuf offset:0 atIndex:1];
            [enc setBuffer:nBuf  offset:0 atIndex:2];
            [enc setBuffer:gBuf  offset:0 atIndex:3];
            [enc setBytes:&bp length:sizeof(bp) atIndex:4];
            [enc setBuffer:obuf[d] offset:0 atIndex:5];
            [enc dispatchThreads:MTLSizeMake(bs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            cmds[d].push_back(cb);
        }
    }

    for (int d = 0; d < nd; d++) {
        for (id<MTLCommandBuffer> cb : cmds[d]) {
            [cb waitUntilCompleted];
            gpu_check(cb, "bruteMap");
        }
        if (!cmds[d].empty()) {
            long cnt = hi[d] - lo[d];
            memcpy(out + lo[d], [obuf[d] contents], cnt * sizeof(int64_t));
        }
    }
}

} // namespace gpu
