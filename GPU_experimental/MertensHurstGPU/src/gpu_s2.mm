// gpu_s2.mm — Metal implementation of the S2 offload (see gpu_s2.h).
//
// Design notes (validated by the prototypes in this folder):
//  - Exact 64-bit division: Apple GPUs use the custom float-seeded divider
//    (3 correction stages — exact for all q < 2^69, far beyond any sieve
//    position); Metal's `ulong /` libcall is ~6.5x slower there. On AMD the
//    measured ordering is REVERSED (native 4.55e10/s vs custom 3.82e10/s on
//    a Vega II), so non-Apple devices select native division through a
//    function constant — both paths are exact integer division.
//  - One specialized pipeline per S2 mode via function constants; items are
//    bucketed by mode host-side so every SIMD group runs straight-line code.
//  - Alternating +4/+2 coprime-to-6 iteration (no per-visit modulo).
//  - Branchless mu handling (divide always, multiply by mu).
//  - Multi-GPU (e.g. Vega II Duo): every device from MTLCopyAllDevices
//    (minus low-power iGPUs) gets its own queue/pipelines/buffers; waves
//    are dealt round-robin — chunk sums are independent, so no cross-GPU
//    traffic is needed. Headless devices are ordered FIRST, so
//    MERTENS_GPU_DEVICES=1 selects a GPU that is not driving a display
//    (recommended for unattended runs). Unified-memory Macs have one
//    device, so nothing changes there.
//  - Discrete GPUs: a Shared buffer is host RAM behind PCIe (~12 ns per mu
//    read measured on a Vega II), so Shared buffers serve as staging and
//    the kernels bind Private (VRAM) mirrors, blit-updated on the in-order
//    queue. All blits are chunked (<= 256 MB per command buffer).
//
// WAVE STREAMING (added 2026-06-12 after the 1e18 incident): items are
// dispatched in bounded waves of <= kWaveCap items over a small pool of
// recycled buffers, instead of materializing a block's full item list. At
// n = 1e18 a single block emits ~2e9 items (~48 GB): the all-at-once design
// overcommitted VRAM, fell back to PCIe-resident items, inflated command
// buffers past the ~5 s GPU timeout (kIOAccelCommandBufferCallbackError-
// Timeout), and starved WindowServer on the display GPU long enough that
// watchdogd panicked the machine (panic: "no successful checkins from
// WindowServer in 120 seconds"). Waves keep GPU memory constant in n, and
// command buffers are additionally budgeted by ESTIMATED VISITS, so no
// single one can approach the watchdog timeout even when degraded.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "gpu_s2.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

// Engine profiling (MERTENS_GPU_PROF=1): accumulate where the S2 wall goes —
// GPU execution time (from each command buffer's GPUStartTime/GPUEndTime),
// host time blocked waiting on the GPU, host time spent in the serial item
// feed (memcpy + encode + commit), and how often back-pressure forced a
// blocking reap. A summary prints at destroy. Distinguishes a saturated GPU
// (gpuBusy ~ S2 wall -> co-partition) from a starved one (feed/wait dominate,
// gpuBusy small -> parallelize the feed). Zero cost when the env var is unset.
struct GpuProf {
    bool on = false;
    double gpuBusy = 0;       // summed cb GPU execution time (s)
    double hostWait = 0;      // host time in blocking waitUntilCompleted (s)
    double hostFeed = 0;      // host time in dispatchWave (memcpy/encode/commit)
    uint64_t nCbs = 0;
    uint64_t nBackpressure = 0;   // last-resort blocking reaps in acquireWave
    uint64_t nWaves = 0;
    static double now() {
        using namespace std::chrono;
        return duration<double>(steady_clock::now().time_since_epoch()).count();
    }
    static void report();   // defined after the class; registered via atexit
    GpuProf() {
        on = getenv("MERTENS_GPU_PROF") != nullptr;
        // gGpuS2 lives for the whole run (destroyed only on a mode switch),
        // so hang the summary off process exit rather than the destructor.
        if (on) atexit(&GpuProf::report);
    }
};
static GpuProf gProf;
void GpuProf::report() {
    fprintf(stderr,
        "\n[gpu_s2 prof] GPU-busy=%.3fs  host-wait(blocked on GPU)=%.3fs  "
        "host-feed(serial emit/encode)=%.3fs\n"
        "[gpu_s2 prof] command buffers=%llu  waves=%llu  "
        "back-pressure blocking reaps=%llu\n",
        gProf.gpuBusy, gProf.hostWait, gProf.hostFeed,
        (unsigned long long)gProf.nCbs, (unsigned long long)gProf.nWaves,
        (unsigned long long)gProf.nBackpressure);
}

static const char* kMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

// Exact n/d for n < 2^60, d < 2^32. Three float-seeded stages leave the
// residual error below 1 for all q < 2^69; the trailing loops settle the
// final +-1. Verified against CPU division in bench_gpu and end-to-end.
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

// S2_term modes 0..9 — must match MertensHurst/src/S2.h. Divisions are by
// constants and strength-reduce to multiplies.
inline long s2term(int mode, long q) {
    switch (mode) {
        case 0: return q;
        case 1: return (q + 1) >> 1;
        case 2: return q & 1;
        case 3: return (q >> 2) + (q & 1);
        case 4: return ((q + 1) >> 1) - q / 3;
        case 5: return ((q + 1) >> 1) - ((q / 3 + 1) >> 1);
        case 6: return (q >> 2) + (q & 1) - q / 3;
        case 7: return (q >> 2) + (q & 1) - (q + 3) / 6;
        case 8: return (q >> 2) + (q & 1) - (long)((q % 6) >= 3);
        default: return (q >> 2) + (q & 1) - ((q / 3) >> 2) - ((q / 3) & 1);
    }
}

struct S2Item {
    ulong n;
    uint  x1, x2;      // inclusive k-range; k < 2^32 for 64-bit args
    uint  outIdx;
    uint  pad;
};

constant int  MODE       [[function_constant(0)]];
constant bool NATIVE_DIV [[function_constant(1)]];

// Per-device division choice (see header comment). Function constants are
// compile-time, so the losing branch is eliminated from the pipeline.
inline ulong qdiv(ulong n, uint d) {
    return NATIVE_DIV ? (n / (ulong)d) : div64_mertens(n, d);
}

kernel void s2_items(
    device const S2Item* items [[buffer(0)]],
    device const char*   mu    [[buffer(1)]],
    device long*         out   [[buffer(2)]],
    device const ulong*  pL1   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const S2Item it = items[gid];
    const uint L1 = (uint)pL1[0];          // mu index base; len <= block size
    const ulong n = it.n;

    uint i = it.x1 | 1u;
    if (i % 3 == 0) i += 2;
    uint step = (i % 3 == 1) ? 4u : 2u;

    long s = 0;
    for (; i <= it.x2; i += step, step = 6u - step) {
        const char m = mu[i - L1];
        const long q = (long)qdiv(n, i);
        s += (long)m * s2term(MODE, q);
    }
    out[gid] = s;
}
)MSL";

// --------------------------------------------------------------------------

struct ItemHost {
    uint64_t n;
    uint32_t x1, x2;
    uint32_t outIdx;
    uint32_t pad;
};
static_assert(sizeof(ItemHost) == 24, "item layout");

static constexpr uint32_t kSpan = 8192;       // k-range per GPU thread
// Batch and queue sizing: Metal queues allow only kQueueDepth uncompleted
// command buffers before [queue commandBuffer] BLOCKS the host. The wave
// engine feeds one device many consecutive buffers per wave, so with small
// batches and the default depth of 64 (~15 ms of work at 1e16 item sizes),
// the OTHER GPU drains its queue and starves while the host is stuck —
// measured 13% end-to-end on the dual-Vega Mac Pro, on a path the original
// engine dodged by alternating devices per batch. Deeper queues + larger
// batches keep ~0.5 s of work resident per device; the visit budget still
// caps every buffer at single-digit ms of GPU time, so the watchdog
// margins from the 1e18 incident are unchanged.
static constexpr uint32_t kBatchItems = 131072;           // per command buffer
static constexpr uint64_t kBatchVisits = 32u << 20;       // est. visits per cb
static constexpr NSUInteger kQueueDepth = 512;
// Wave sizing: small waves give fine-grained load balance across multiple
// GPUs (waves carry unequal mode mixes); a generous pool ceiling means the
// host normally NEVER blocks on the GPU — at 1e16-scale every block's items
// fit in flight, preserving the full S1 overlap of the all-at-once design —
// while still bounding GPU memory to ~3 GB/device at 1e18 scale, where
// back-pressure is the point. (First wave version used 3x4M-item waves:
// safe, but it stalled the host before S1 and balanced two GPUs at whole-
// wave granularity — measured 14% slower on the dual-Vega Mac Pro.)
static constexpr uint32_t kWaveCap = 1u << 20;            // items per wave (24 MB)
static constexpr int      kMaxWavesPerDev = 48;
static constexpr uint64_t kBlitChunk = 256u << 20;        // bytes per blit cb

// A recyclable dispatch unit: fixed-size buffers, refilled per use. The
// Shared bItems doubles as the host-side copy for outIdx readback when the
// results are applied.
struct Wave {
    id<MTLBuffer> bItems = nil;   // Shared staging
    id<MTLBuffer> dItems = nil;   // Private mirror (discrete; may stay nil)
    id<MTLBuffer> bOut = nil;     // Shared
    std::vector<id<MTLCommandBuffer>> cbs;
    uint64_t estVisits = 0;       // load estimate, for least-loaded dealing
    uint32_t count = 0;
    bool inFlight = false;
};

// Per-device state. Buffers are device-owned in Metal, so everything except
// the host-side item lists is duplicated per device.
struct DevCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> psm[10] = {};
    id<MTLBuffer> bMu = nil, bL1 = nil;       // Shared
    id<MTLBuffer> dMu = nil;                  // Private mirror (discrete only)
    id<MTLCommandBuffer> muBlit = nil;        // last mu upload, if in flight
    std::vector<Wave> waves;                  // grows on demand to kMaxWavesPerDev
    uint64_t outstanding = 0;                 // est. visits currently in flight
    bool unified = true;
    uint64_t muLenCap = 0;
};

class GpuS2 {
public:
    std::vector<DevCtx> devs;
    std::vector<ItemHost> modeItems[10];      // pending, not yet in a wave
    uint64_t pendingCount = 0;
    std::vector<ItemHost> waveScratch;        // mode-ordered fill for one wave
    std::vector<std::pair<int, uint32_t>> waveRuns;  // (mode, count) runs
    // In-flight waves in dispatch order: (device index, wave index)
    std::deque<std::pair<uint32_t, uint32_t>> inflight;
    int64_t* pv = nullptr;                    // partialValues of current block
    uint64_t blockL1 = 0;

    bool initDevice(DevCtx& c, id<MTLDevice> dev) {
        c.dev = dev;
        c.unified = dev.hasUnifiedMemory;
        // Measured on Vega II: AMD's compiler-emitted 64-bit `/` beats the
        // float-seeded divider; on Apple GPUs it is ~6.5x worse.
        const char* name = dev.name.UTF8String;
        const bool nativeDiv = (strstr(name, "AMD") || strstr(name, "Radeon") ||
                                strstr(name, "Intel") || strstr(name, "NVIDIA"));
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMSL]
                                               options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "gpu_s2: MSL compile failed (%s): %s\n",
                    name, err.localizedDescription.UTF8String);
            return false;
        }
        for (int m = 0; m < 10; ++m) {
            MTLFunctionConstantValues* cv = [MTLFunctionConstantValues new];
            int32_t mv = m;
            bool nd = nativeDiv;
            [cv setConstantValue:&mv type:MTLDataTypeInt atIndex:0];
            [cv setConstantValue:&nd type:MTLDataTypeBool atIndex:1];
            id<MTLFunction> fn = [lib newFunctionWithName:@"s2_items"
                                           constantValues:cv error:&err];
            if (!fn) return false;
            c.psm[m] = [dev newComputePipelineStateWithFunction:fn error:nil];
            if (!c.psm[m]) return false;
        }
        c.queue = [dev newCommandQueueWithMaxCommandBufferCount:kQueueDepth];
        if (!c.queue) c.queue = [dev newCommandQueue];
        c.bL1 = [dev newBufferWithLength:8 options:MTLResourceStorageModeShared];
        fprintf(stderr, "gpu_s2: using %s (%s memory, %s division%s)\n",
                name, c.unified ? "unified" : "discrete",
                nativeDiv ? "native" : "float-seeded",
                [dev isHeadless] ? ", headless" : ", drives a display");
        return c.queue != nil;
    }

    bool init() {
        id<MTLDevice> dflt = MTLCreateSystemDefaultDevice();
        if (!dflt) return false;

        NSArray<id<MTLDevice>>* all = MTLCopyAllDevices();
        std::vector<id<MTLDevice>> picked;
        picked.push_back(dflt);
        for (id<MTLDevice> d in all) {
            if (d.registryID == dflt.registryID) continue;
            if (d.isLowPower) continue;       // skip iGPUs next to real GPUs
            picked.push_back(d);
        }
        // Headless devices first: they can't starve WindowServer, and
        // MERTENS_GPU_DEVICES=1 then picks a non-display GPU.
        std::stable_sort(picked.begin(), picked.end(),
                         [](id<MTLDevice> a, id<MTLDevice> b) {
                             return [a isHeadless] && ![b isHeadless];
                         });
        size_t nDev = picked.size();
        if (const char* e = getenv("MERTENS_GPU_DEVICES")) {
            long cap = atol(e);
            if (cap >= 1 && (size_t)cap < nDev) nDev = (size_t)cap;
        }
        for (size_t i = 0; i < nDev; ++i) {
            DevCtx c;
            if (!initDevice(c, picked[i])) {
                // A failed extra device just isn't used; the first must work.
                if (i == 0) return false;
                continue;
            }
            devs.push_back(std::move(c));
        }
        return !devs.empty();
    }

    // Async chunked copy src -> dst on the device's (in-order) queue: later
    // encodings on the same queue see it completed, and no single blit
    // command buffer runs long enough to interest the GPU watchdog.
    // Returns the LAST command buffer (in-order queue: it completes last).
    id<MTLCommandBuffer> blitAsync(DevCtx& c, id<MTLBuffer> src,
                                   id<MTLBuffer> dst, uint64_t len) {
        id<MTLCommandBuffer> last = nil;
        for (uint64_t off = 0; off < len; off += kBlitChunk) {
            const uint64_t cl = std::min(kBlitChunk, len - off);
            id<MTLCommandBuffer> cb = [c.queue commandBuffer];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b copyFromBuffer:src sourceOffset:off
                     toBuffer:dst destinationOffset:off size:cl];
            [b endEncoding];
            [cb commit];
            last = cb;
        }
        return last;
    }

    void beginBlock(const int8_t* mu, uint64_t len, uint64_t L1, int64_t* pv_) {
        blockL1 = L1;
        pv = pv_;
        for (DevCtx& c : devs) {
            // A zero-item block leaves waitApply nothing to sync on, so the
            // previous mu upload could still be reading the staging buffer
            // we memcpy below.
            if (c.muBlit) { [c.muBlit waitUntilCompleted]; c.muBlit = nil; }
            if (len > c.muLenCap) {
                c.muLenCap = len + (len >> 2);
                c.bMu = [c.dev newBufferWithLength:c.muLenCap
                                           options:MTLResourceStorageModeShared];
                if (!c.unified) {
                    c.dMu = [c.dev newBufferWithLength:c.muLenCap
                                               options:MTLResourceStorageModePrivate];
                    if (!c.dMu) {
                        fprintf(stderr, "gpu_s2: VRAM alloc failed (%llu bytes) — "
                                        "falling back to host-resident mu\n",
                                (unsigned long long)c.muLenCap);
                        c.unified = true;   // degrade to the Shared (PCIe) path
                    }
                }
            }
            memcpy(c.bMu.contents, mu, len);
            if (!c.unified) c.muBlit = blitAsync(c, c.bMu, c.dMu, len);
            *(uint64_t*)c.bL1.contents = L1;
        }
        for (auto& v : modeItems) v.clear();
        pendingCount = 0;
    }

    void emit(uint64_t n, uint64_t lo, uint64_t hi, int mode, uint32_t outIdx) {
        if (lo > hi) return;
        if (hi >= (1ULL << 32)) {
            fprintf(stderr, "gpu_s2: k-range exceeds 2^32 (hi=%llu) — "
                            "64-bit S2 invariant violated\n",
                    (unsigned long long)hi);
            abort();
        }
        for (uint64_t s = lo; s <= hi; s += kSpan) {
            modeItems[mode].push_back(
                {n, (uint32_t)s, (uint32_t)std::min(s + kSpan - 1, hi), outIdx, 0});
            ++pendingCount;
        }
    }

    // Mirror of update_S2<Half>'s mode-range decomposition (S2.h).
    void addTask(uint64_t n, uint64_t x1, uint64_t x2,
                 uint64_t nu, uint64_t nu2, bool half, uint32_t outIdx) {
        if (half) {
            emit(n, x1, std::min(x2, (nu2 >> 1) / 3), 9, outIdx);
            emit(n, std::max(x1, (nu2 >> 1) / 3 + 1), std::min(x2, (nu >> 1) / 3), 8, outIdx);
            emit(n, std::max(x1, (nu >> 1) / 3 + 1), std::min(x2, nu2 / 3), 7, outIdx);
            emit(n, std::max(x1, nu2 / 3 + 1), std::min(x2, nu / 3), 6, outIdx);
            emit(n, std::max(x1, nu / 3 + 1), std::min(x2, nu2 >> 1), 3, outIdx);
            emit(n, std::max(x1, (nu2 >> 1) + 1), std::min(x2, nu >> 1), 2, outIdx);
            emit(n, std::max(x1, (nu >> 1) + 1), std::min(x2, nu2), 1, outIdx);
            emit(n, std::max(x1, nu2 + 1), x2, 0, outIdx);
        } else {
            emit(n, x1, std::min(x2, (nu >> 1) / 3), 5, outIdx);
            emit(n, std::max(x1, (nu >> 1) / 3 + 1), std::min(x2, nu / 3), 4, outIdx);
            emit(n, std::max(x1, nu / 3 + 1), std::min(x2, nu >> 1), 1, outIdx);
            emit(n, std::max(x1, (nu >> 1) + 1), x2, 0, outIdx);
        }
        // Bounded memory: stream a wave out as soon as one fills.
        if (pendingCount >= kWaveCap) flushPending(false);
    }

    // Apply a completed wave's sums and recycle its buffers. All callers are
    // on the host serial region (begin/add/dispatch/wait), so writing pv is
    // race-free with respect to the S1 OpenMP writers.
    void reapWave(uint32_t di, uint32_t wi) {
        DevCtx& c = devs[di];
        Wave& w = c.waves[wi];
        if (gProf.on) {
            const double t0 = GpuProf::now();
            [w.cbs.back() waitUntilCompleted]; // queue is in-order
            gProf.hostWait += GpuProf::now() - t0;
        } else {
            [w.cbs.back() waitUntilCompleted]; // queue is in-order
        }
        for (id<MTLCommandBuffer> cb : w.cbs) {
            if (cb.status != MTLCommandBufferStatusCompleted || cb.error) {
                fprintf(stderr, "gpu_s2: command buffer failed on %s: status=%ld %s\n",
                        c.dev.name.UTF8String, (long)cb.status,
                        cb.error ? cb.error.localizedDescription.UTF8String : "");
                abort();                      // never silently drop S2 terms
            }
            if (gProf.on) {
                gProf.gpuBusy += cb.GPUEndTime - cb.GPUStartTime;
                ++gProf.nCbs;
            }
        }
        const ItemHost* it = (const ItemHost*)w.bItems.contents;
        const int64_t* out = (const int64_t*)w.bOut.contents;
        for (uint32_t k = 0; k < w.count; ++k)
            pv[it[k].outIdx] -= out[k];
        c.outstanding -= w.estVisits;
        w.estVisits = 0;
        w.cbs.clear();
        w.count = 0;
        w.inFlight = false;
    }

    // Remove the given in-flight entry from the FIFO and reap it.
    void reapEntry(std::deque<std::pair<uint32_t, uint32_t>>::iterator q) {
        const auto [di, wi] = *q;
        inflight.erase(q);
        reapWave(di, wi);
    }

    // Free wave on device di. Order of preference: an idle pool slot; a
    // FINISHED in-flight wave (non-blocking reap — the common case once the
    // pipeline is warm); growing the pool (up to the ceiling); and only as
    // a last resort blocking on the device's oldest wave — the back-pressure
    // that bounds memory on monster blocks (1e18 scale). Everything before
    // that last step keeps the host free so S1 overlap is preserved.
    uint32_t acquireWave(uint32_t di) {
        DevCtx& c = devs[di];
        for (uint32_t w = 0; w < (uint32_t)c.waves.size(); ++w)
            if (!c.waves[w].inFlight) return w;
        for (auto q = inflight.begin(); q != inflight.end(); ++q) {
            if (q->first != di) continue;
            Wave& w = c.waves[q->second];
            if (w.cbs.back().status == MTLCommandBufferStatusCompleted ||
                w.cbs.back().error) {
                const uint32_t wi = q->second;
                reapEntry(q);
                return wi;
            }
        }
        if (c.waves.size() < (size_t)kMaxWavesPerDev) {
            c.waves.emplace_back();
            return (uint32_t)(c.waves.size() - 1);
        }
        for (auto q = inflight.begin(); q != inflight.end(); ++q) {
            if (q->first == di) {
                const uint32_t wi = q->second;
                if (gProf.on) ++gProf.nBackpressure;
                reapEntry(q);
                return wi;
            }
        }
        fprintf(stderr, "gpu_s2: wave pool inconsistent\n");
        abort();
    }

    void dispatchWave() {
        // Pure host feed time = this function's wall minus any blocking reap
        // (hostWait) that back-pressure triggers inside acquireWave.
        const double tFeed0 = gProf.on ? GpuProf::now() : 0;
        const double wait0  = gProf.on ? gProf.hostWait : 0;
        // Carve up to kWaveCap items, mode-grouped, from the pending buckets
        // (taken from the back — item order within a mode is irrelevant).
        waveScratch.clear();
        waveRuns.clear();
        for (int m = 0; m < 10 && waveScratch.size() < kWaveCap; ++m) {
            auto& v = modeItems[m];
            if (v.empty()) continue;
            const uint32_t take =
                (uint32_t)std::min<uint64_t>(v.size(), kWaveCap - waveScratch.size());
            waveScratch.insert(waveScratch.end(), v.end() - take, v.end());
            v.resize(v.size() - take);
            waveRuns.push_back({m, take});
        }
        const uint32_t count = (uint32_t)waveScratch.size();
        if (count == 0) return;
        pendingCount -= count;

        // Least-loaded dealing: waves carry unequal mode mixes, so blind
        // round-robin can leave one GPU a whole expensive wave behind.
        uint32_t di = 0;
        for (uint32_t d = 1; d < (uint32_t)devs.size(); ++d)
            if (devs[d].outstanding < devs[di].outstanding) di = d;
        DevCtx& c = devs[di];
        const uint32_t wi = acquireWave(di);
        Wave& w = c.waves[wi];

        if (!w.bItems) {                      // lazy one-time pool allocation
            w.bItems = [c.dev newBufferWithLength:(uint64_t)kWaveCap * sizeof(ItemHost)
                                          options:MTLResourceStorageModeShared];
            w.bOut = [c.dev newBufferWithLength:(uint64_t)kWaveCap * 8
                                        options:MTLResourceStorageModeShared];
            if (!c.unified)
                w.dItems = [c.dev newBufferWithLength:(uint64_t)kWaveCap * sizeof(ItemHost)
                                              options:MTLResourceStorageModePrivate];
        }
        memcpy(w.bItems.contents, waveScratch.data(), (uint64_t)count * sizeof(ItemHost));
        if (!c.unified && w.dItems)
            blitAsync(c, w.bItems, w.dItems, (uint64_t)count * sizeof(ItemHost));
        id<MTLBuffer> kItems = (!c.unified && w.dItems) ? w.dItems : w.bItems;
        id<MTLBuffer> kMu    = (!c.unified && c.dMu)    ? c.dMu    : c.bMu;

        // Encode per mode run, sub-batched by item count AND estimated
        // visits: a runaway command buffer is what trips the ~5 s GPU
        // timeout / starves WindowServer, so both axes are capped.
        uint64_t waveVisits = 0;
        uint32_t pos = 0;
        for (auto [m, runCnt] : waveRuns) {
            uint32_t b0 = 0;
            while (b0 < runCnt) {
                uint32_t cnt = 0;
                uint64_t visits = 0;
                while (b0 + cnt < runCnt && cnt < kBatchItems && visits < kBatchVisits) {
                    const ItemHost& ih = waveScratch[pos + b0 + cnt];
                    visits += (ih.x2 - ih.x1) / 3 + 1;
                    ++cnt;
                }
                waveVisits += visits;
                id<MTLCommandBuffer> cb = [c.queue commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:c.psm[m]];
                [e setBuffer:kItems offset:(uint64_t)(pos + b0) * sizeof(ItemHost) atIndex:0];
                [e setBuffer:kMu offset:0 atIndex:1];
                [e setBuffer:w.bOut offset:(uint64_t)(pos + b0) * 8 atIndex:2];
                [e setBuffer:c.bL1 offset:0 atIndex:3];
                [e dispatchThreads:MTLSizeMake(cnt, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e endEncoding];
                [cb commit];                 // no wait: overlaps CPU work
                w.cbs.push_back(cb);
                b0 += cnt;
            }
            pos += runCnt;
        }
        w.count = count;
        w.estVisits = waveVisits;
        c.outstanding += waveVisits;
        w.inFlight = true;
        inflight.push_back({di, wi});
        if (gProf.on) {
            gProf.hostFeed += (GpuProf::now() - tFeed0) - (gProf.hostWait - wait0);
            ++gProf.nWaves;
        }
    }

    // all=false: stream out full waves only (called from addTask).
    // all=true: also flush the final partial wave (dispatch/wait paths).
    void flushPending(bool all) {
        while (pendingCount >= kWaveCap) dispatchWave();
        if (all && pendingCount > 0) dispatchWave();
    }

    void dispatchAsync() {
        flushPending(true);                   // everything in flight; no wait
    }

    void waitApply(int64_t* pv_) {
        pv = pv_;                             // same pointer as begin_block
        flushPending(true);                   // safety: normally already empty
        while (!inflight.empty()) {
            auto [di, wi] = inflight.front();
            inflight.pop_front();
            reapWave(di, wi);
        }
    }
};

// --------------------------------------------------------------------------

GpuS2* gpus2_create() {
    @autoreleasepool {
        GpuS2* g = new GpuS2();
        if (!g->init()) { delete g; return nullptr; }
        return g;
    }
}
void gpus2_destroy(GpuS2* g) { delete g; }

void gpus2_begin_segment(GpuS2* g, const int8_t* mu, uint64_t len, uint64_t L1,
                       int64_t* partialValues) {
    @autoreleasepool { g->beginBlock(mu, len, L1, partialValues); }
}
void gpus2_add_task(GpuS2* g, uint64_t n, uint64_t x1, uint64_t x2,
                    uint64_t nu, uint64_t nu2, bool half, uint32_t outIdx) {
    @autoreleasepool { g->addTask(n, x1, x2, nu, nu2, half, outIdx); }
}
void gpus2_dispatch_async(GpuS2* g) {
    @autoreleasepool { g->dispatchAsync(); }
}
void gpus2_wait_apply(GpuS2* g, int64_t* partialValues) {
    @autoreleasepool { g->waitApply(partialValues); }
}
