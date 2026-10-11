// gabor_metal.mm — fused differentiable Gabor render on Apple GPU, callable on
// PyTorch MPS tensors. Forward + backward Metal kernels.
//
// Matches tools/shared/gabor.py:_render_chunk EXACTLY (coord grid [-1,1],
// theta*2pi, sigma=exp(log_sigma), freq=2pi/exp(rel_freq), phase=psi*2pi, gamma
// denominator clamp 1e-3, RAW sum — no clamp/scale). Forward is a batched port
// of source/projects/jit.gaborender's kShaderSource; backward is the analytic
// gradient (one thread per (batch,gabor), reduce over pixels → no atomics).
//
// Integration: encode into PyTorch's own MPS command buffer on its serial
// dispatch queue (torch::mps), zero-copy on the tensors' MTLBuffers.

#include <torch/extension.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

static const char *kMetalSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant float TAU = 6.283185307179586f;

// ── FORWARD: one thread per (batch, pixel); loop over N gabors ────────────────
kernel void gabor_forward(
    device const float *params [[buffer(0)]],   // (B, N, 12) std-physical
    device float       *out    [[buffer(1)]],   // (B, 3, H, W)
    constant uint &B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    constant uint &H [[buffer(4)]],
    constant uint &W [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    ulong HW = (ulong)H * (ulong)W;
    if ((ulong)gid >= (ulong)B * HW) return;
    uint b = (uint)((ulong)gid / HW);
    ulong p = (ulong)gid - (ulong)b * HW;
    uint  y = (uint)(p / (ulong)W);
    uint  x = (uint)(p - (ulong)y * (ulong)W);
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;
    float px = 2.0f * float(x) / fw - 1.0f;
    float py = 2.0f * float(y) / fh - 1.0f;

    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    device const float *pb = params + (ulong)b * (ulong)N * 12ul;
    for (uint n = 0u; n < N; ++n) {
        device const float *g = pb + (ulong)n * 12ul;
        float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
        float dx = px - u, dy = py - v;
        float ct = cos(th), st = sin(th);
        float xr =  dx * ct + dy * st;
        float yr = -dx * st + dy * ct;
        float sigma = exp(ls);
        float gs = max(gam, 1e-3f);
        float gauss = exp(-xr * xr / (2.0f * sigma * sigma)
                          - yr * yr / (2.0f * gs * gs));
        float freq = TAU / exp(rf);
        float arg  = freq * xr;
        sr += g[9]  * gauss * cos(arg + g[6] * TAU);
        sg += g[10] * gauss * cos(arg + g[7] * TAU);
        sb += g[11] * gauss * cos(arg + g[8] * TAU);
    }
    ulong base = (ulong)b * 3ul * HW;
    out[base + 0ul * HW + p] = sr;
    out[base + 1ul * HW + p] = sg;
    out[base + 2ul * HW + p] = sb;
}

// ── BACKWARD: one thread per (batch, gabor); reduce over all pixels ───────────
kernel void gabor_backward_serial(
    device const float *params  [[buffer(0)]],  // (B, N, 12)
    device const float *grad    [[buffer(1)]],  // (B, 3, H, W) upstream dL/dout
    device float       *gparams [[buffer(2)]],  // (B, N, 12) out
    constant uint &B [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &W [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * N) return;
    uint b = gid / N;
    uint n = gid % N;
    device const float *g = params + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
    float pr = g[6], pg = g[7], pbb = g[8], ar = g[9], ag = g[10], ab = g[11];
    float ct = cos(th), st = sin(th);
    float sigma = exp(ls);
    float inv_s2 = 1.0f / (sigma * sigma);
    float gs = max(gam, 1e-3f);
    float inv_gs2 = 1.0f / (gs * gs);
    float inv_gs3 = inv_gs2 / gs;
    float freq = TAU / exp(rf);
    float pr2 = pr * TAU, pg2 = pg * TAU, pb2 = pbb * TAU;

    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0,d8=0,d9=0,d10=0,d11=0;
    ulong HW = (ulong)H * (ulong)W;
    ulong gbase = (ulong)b * 3ul * HW;
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;

    for (uint y = 0u; y < H; ++y) {
        float py = 2.0f * float(y) / fh - 1.0f;
        ulong row = (ulong)y * (ulong)W;
        for (uint x = 0u; x < W; ++x) {
            ulong p = row + (ulong)x;
            float px = 2.0f * float(x) / fw - 1.0f;
            float dx = px - u, dy = py - v;
            float xr =  dx * ct + dy * st;
            float yr = -dx * st + dy * ct;
            float gg = exp(-xr * xr * 0.5f * inv_s2 - yr * yr * 0.5f * inv_gs2);
            float arg = freq * xr;
            float G0 = grad[gbase + 0ul * HW + p];
            float G1 = grad[gbase + 1ul * HW + p];
            float G2 = grad[gbase + 2ul * HW + p];
            float ccr = cos(arg + pr2), sr = sin(arg + pr2);
            float ccg = cos(arg + pg2), sg = sin(arg + pg2);
            float ccb = cos(arg + pb2), sb = sin(arg + pb2);
            float P = G0 * ar * ccr + G1 * ag * ccg + G2 * ab * ccb;   // dL/dgauss
            float Q = G0 * ar * sr  + G1 * ag * sg  + G2 * ab * sb;
            float dL_dxr = -gg * (P * xr * inv_s2 + Q * freq);
            float dL_dyr = -gg * P * yr * inv_gs2;
            d0 += dL_dxr * (-ct) + dL_dyr * st;          // u
            d1 += dL_dxr * (-st) + dL_dyr * (-ct);       // v
            d2 += TAU * (dL_dxr * yr - dL_dyr * xr);     // theta
            d3 += P * gg * xr * xr * inv_s2;             // log_sigma
            d4 += P * gg * yr * yr * inv_gs3;            // gamma (mask below)
            d5 += gg * Q * freq * xr;                    // rel_freq
            d6 += -TAU * G0 * ar * gg * sr;              // psi_r
            d7 += -TAU * G1 * ag * gg * sg;              // psi_g
            d8 += -TAU * G2 * ab * gg * sb;              // psi_b
            d9  += G0 * gg * ccr;                        // amp_r
            d10 += G1 * gg * ccg;                        // amp_g
            d11 += G2 * gg * ccb;                        // amp_b
        }
    }
    if (gam <= 1e-3f) d4 = 0.0f;   // clamp_min gradient is 0 below the floor
    device float *o = gparams + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    o[0]=d0; o[1]=d1; o[2]=d2; o[3]=d3; o[4]=d4; o[5]=d5;
    o[6]=d6; o[7]=d7; o[8]=d8; o[9]=d9; o[10]=d10; o[11]=d11;
}

// ── FORWARD (quad): carrier_c = A_c*cos(arg) - B_c*sin(arg); A=6..8, B=9..11 ──
kernel void gabor_forward_quad(
    device const float *params [[buffer(0)]],
    device float       *out    [[buffer(1)]],
    constant uint &B [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    constant uint &H [[buffer(4)]],
    constant uint &W [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    ulong HW = (ulong)H * (ulong)W;
    if ((ulong)gid >= (ulong)B * HW) return;
    uint b = (uint)((ulong)gid / HW);
    ulong p = (ulong)gid - (ulong)b * HW;
    uint  y = (uint)(p / (ulong)W);
    uint  x = (uint)(p - (ulong)y * (ulong)W);
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;
    float px = 2.0f * float(x) / fw - 1.0f;
    float py = 2.0f * float(y) / fh - 1.0f;
    float sr = 0.0f, sg = 0.0f, sb = 0.0f;
    device const float *pb = params + (ulong)b * (ulong)N * 12ul;
    for (uint n = 0u; n < N; ++n) {
        device const float *g = pb + (ulong)n * 12ul;
        float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
        float dx = px - u, dy = py - v;
        float ct = cos(th), st = sin(th);
        float xr =  dx * ct + dy * st;
        float yr = -dx * st + dy * ct;
        float sigma = exp(ls);
        float gs = max(gam, 1e-3f);
        float gauss = exp(-xr * xr / (2.0f * sigma * sigma)
                          - yr * yr / (2.0f * gs * gs));
        float freq = TAU / exp(rf);
        float arg  = freq * xr;
        float ca = cos(arg), sa = sin(arg);
        sr += gauss * (g[6] * ca - g[9]  * sa);
        sg += gauss * (g[7] * ca - g[10] * sa);
        sb += gauss * (g[8] * ca - g[11] * sa);
    }
    ulong base = (ulong)b * 3ul * HW;
    out[base + 0ul * HW + p] = sr;
    out[base + 1ul * HW + p] = sg;
    out[base + 2ul * HW + p] = sb;
}

// ── BACKWARD (quad): color block linear in (A,B), no phase-wrap ───────────────
kernel void gabor_backward_quad_serial(
    device const float *params  [[buffer(0)]],
    device const float *grad    [[buffer(1)]],
    device float       *gparams [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &W [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= B * N) return;
    uint b = gid / N, n = gid % N;
    device const float *g = params + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
    float Ar = g[6], Ag = g[7], Ab = g[8], Br = g[9], Bg = g[10], Bb = g[11];
    float ct = cos(th), st = sin(th);
    float sigma = exp(ls); float inv_s2 = 1.0f / (sigma * sigma);
    float gs = max(gam, 1e-3f); float inv_gs2 = 1.0f / (gs * gs); float inv_gs3 = inv_gs2 / gs;
    float freq = TAU / exp(rf);
    float d0=0,d1=0,d2=0,d3=0,d4=0,d5=0,d6=0,d7=0,d8=0,d9=0,d10=0,d11=0;
    ulong HW = (ulong)H * (ulong)W;
    ulong gbase = (ulong)b * 3ul * HW;
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;
    for (uint y = 0u; y < H; ++y) {
        float py = 2.0f * float(y) / fh - 1.0f;
        ulong row = (ulong)y * (ulong)W;
        for (uint x = 0u; x < W; ++x) {
            ulong p = row + (ulong)x;
            float px = 2.0f * float(x) / fw - 1.0f;
            float dx = px - u, dy = py - v;
            float xr =  dx * ct + dy * st;
            float yr = -dx * st + dy * ct;
            float gg = exp(-xr * xr * 0.5f * inv_s2 - yr * yr * 0.5f * inv_gs2);
            float arg = freq * xr;
            float ca = cos(arg), sa = sin(arg);
            float G0 = grad[gbase + 0ul * HW + p];
            float G1 = grad[gbase + 1ul * HW + p];
            float G2 = grad[gbase + 2ul * HW + p];
            float P = G0 * (Ar*ca - Br*sa) + G1 * (Ag*ca - Bg*sa) + G2 * (Ab*ca - Bb*sa);
            float R = G0 * (Ar*sa + Br*ca) + G1 * (Ag*sa + Bg*ca) + G2 * (Ab*sa + Bb*ca);
            float dL_dxr = -gg * (P * xr * inv_s2 + R * freq);
            float dL_dyr = -gg * P * yr * inv_gs2;
            d0 += dL_dxr * (-ct) + dL_dyr * st;
            d1 += dL_dxr * (-st) + dL_dyr * (-ct);
            d2 += TAU * (dL_dxr * yr - dL_dyr * xr);
            d3 += P * gg * xr * xr * inv_s2;
            d4 += P * gg * yr * yr * inv_gs3;
            d5 += gg * R * freq * xr;
            d6 += G0 * gg * ca;   d7 += G1 * gg * ca;   d8 += G2 * gg * ca;   // A_rgb
            d9 += -G0 * gg * sa;  d10 += -G1 * gg * sa; d11 += -G2 * gg * sa; // B_rgb
        }
    }
    if (gam <= 1e-3f) d4 = 0.0f;
    device float *o = gparams + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    o[0]=d0; o[1]=d1; o[2]=d2; o[3]=d3; o[4]=d4; o[5]=d5;
    o[6]=d6; o[7]=d7; o[8]=d8; o[9]=d9; o[10]=d10; o[11]=d11;
}

// ─────────────────────────────────────────────────────────────────────────────
// THREADGROUP-PARALLEL BACKWARD (ported from jit.gaborfit/test/gabor_kernels.metal)
//
// The *_serial kernels above dispatch one THREAD per (batch, gabor) and have it
// sweep the whole image. At B=1, N=256 that is 256 threads each looping 262144
// pixels, which leaves the GPU almost entirely idle: measured 1.5 G atom-pixel/s
// against the forward's 56 G/s on identical work.
//
// These dispatch one THREADGROUP per (batch, gabor) instead. The pixel loop is
// strided across the threadgroup, then the 12 accumulators are reduced with
// simd_sum plus one cross-simdgroup pass through threadgroup memory. The
// arithmetic inside the loop is copied verbatim from the serial versions, so
// gradients are identical up to float summation order.
//
// The stride MUST come from [[threads_per_threadgroup]] and not a compile-time
// constant: the dispatch clamps threadgroup size to the pipeline's
// maxTotalThreadsPerThreadgroup, and when a hardcoded stride disagreed with the
// real size the loop silently skipped pixels — wrong gradients, no error. That
// bug cost a debugging session in jit.gaborfit; the test sweeps tg sizes to
// guard against it.
// ─────────────────────────────────────────────────────────────────────────────

#define GM_MAX_SIMDS 32u

kernel void gabor_backward(
    device const float *params  [[buffer(0)]],  // (B, N, 12)
    device const float *grad    [[buffer(1)]],  // (B, 3, H, W) upstream dL/dout
    device float       *gparams [[buffer(2)]],  // (B, N, 12) out
    constant uint &B [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &W [[buffer(6)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint n_simds   [[simdgroups_per_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]])
{
    threadgroup float part[GM_MAX_SIMDS][12];
    if (gid >= B * N) return;
    uint b = gid / N, n = gid % N;

    device const float *g = params + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
    float pr = g[6], pg = g[7], pbb = g[8], ar = g[9], ag = g[10], ab = g[11];
    float ct = cos(th), st = sin(th);
    float sigma = exp(ls);
    float inv_s2 = 1.0f / (sigma * sigma);
    float gs = max(gam, 1e-3f);
    float inv_gs2 = 1.0f / (gs * gs);
    float inv_gs3 = inv_gs2 / gs;
    float freq = TAU / exp(rf);
    float pr2 = pr * TAU, pg2 = pg * TAU, pb2 = pbb * TAU;

    float d[12];
    for (uint k = 0u; k < 12u; ++k) d[k] = 0.0f;

    ulong HW = (ulong)H * (ulong)W;
    ulong gbase = (ulong)b * 3ul * HW;
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;

    for (ulong i = (ulong)tid; i < HW; i += (ulong)tg_size) {
        uint y = (uint)(i / (ulong)W);
        uint x = (uint)(i - (ulong)y * (ulong)W);
        float px = 2.0f * float(x) / fw - 1.0f;
        float py = 2.0f * float(y) / fh - 1.0f;
        float dx = px - u, dy = py - v;
        float xr =  dx * ct + dy * st;
        float yr = -dx * st + dy * ct;
        float gg = exp(-xr * xr * 0.5f * inv_s2 - yr * yr * 0.5f * inv_gs2);
        float arg = freq * xr;
        float G0 = grad[gbase + 0ul * HW + i];
        float G1 = grad[gbase + 1ul * HW + i];
        float G2 = grad[gbase + 2ul * HW + i];
        float ccr = cos(arg + pr2), sr = sin(arg + pr2);
        float ccg = cos(arg + pg2), sg = sin(arg + pg2);
        float ccb = cos(arg + pb2), sb = sin(arg + pb2);
        float P = G0 * ar * ccr + G1 * ag * ccg + G2 * ab * ccb;   // dL/dgauss
        float Q = G0 * ar * sr  + G1 * ag * sg  + G2 * ab * sb;
        float dL_dxr = -gg * (P * xr * inv_s2 + Q * freq);
        float dL_dyr = -gg * P * yr * inv_gs2;
        d[0]  += dL_dxr * (-ct) + dL_dyr * st;          // u
        d[1]  += dL_dxr * (-st) + dL_dyr * (-ct);       // v
        d[2]  += TAU * (dL_dxr * yr - dL_dyr * xr);     // theta
        d[3]  += P * gg * xr * xr * inv_s2;             // log_sigma
        d[4]  += P * gg * yr * yr * inv_gs3;            // gamma (mask below)
        d[5]  += gg * Q * freq * xr;                    // rel_freq
        d[6]  += -TAU * G0 * ar * gg * sr;              // psi_r
        d[7]  += -TAU * G1 * ag * gg * sg;              // psi_g
        d[8]  += -TAU * G2 * ab * gg * sb;              // psi_b
        d[9]  += G0 * gg * ccr;                         // amp_r
        d[10] += G1 * gg * ccg;                         // amp_g
        d[11] += G2 * gg * ccb;                         // amp_b
    }

    for (uint k = 0u; k < 12u; ++k) d[k] = simd_sum(d[k]);
    if (simd_lane == 0u)
        for (uint k = 0u; k < 12u; ++k) part[simd_id][k] = d[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        device float *o = gparams + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
        for (uint k = 0u; k < 12u; ++k) {
            float s = 0.0f;
            for (uint j = 0u; j < n_simds; ++j) s += part[j][k];
            o[k] = s;
        }
        if (gam <= 1e-3f) o[4] = 0.0f;   // clamp_min gradient is 0 below the floor
    }
}

kernel void gabor_backward_quad(
    device const float *params  [[buffer(0)]],
    device const float *grad    [[buffer(1)]],
    device float       *gparams [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &H [[buffer(5)]],
    constant uint &W [[buffer(6)]],
    uint gid       [[threadgroup_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint n_simds   [[simdgroups_per_threadgroup]],
    uint tg_size   [[threads_per_threadgroup]])
{
    threadgroup float part[GM_MAX_SIMDS][12];
    if (gid >= B * N) return;
    uint b = gid / N, n = gid % N;

    device const float *g = params + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
    float u = g[0], v = g[1], th = g[2] * TAU, ls = g[3], gam = g[4], rf = g[5];
    float Ar = g[6], Ag = g[7], Ab = g[8], Br = g[9], Bg = g[10], Bb = g[11];
    float ct = cos(th), st = sin(th);
    float sigma = exp(ls); float inv_s2 = 1.0f / (sigma * sigma);
    float gs = max(gam, 1e-3f); float inv_gs2 = 1.0f / (gs * gs); float inv_gs3 = inv_gs2 / gs;
    float freq = TAU / exp(rf);

    float d[12];
    for (uint k = 0u; k < 12u; ++k) d[k] = 0.0f;

    ulong HW = (ulong)H * (ulong)W;
    ulong gbase = (ulong)b * 3ul * HW;
    float fw = (W > 1u) ? float(W - 1u) : 1.0f;
    float fh = (H > 1u) ? float(H - 1u) : 1.0f;

    for (ulong i = (ulong)tid; i < HW; i += (ulong)tg_size) {
        uint y = (uint)(i / (ulong)W);
        uint x = (uint)(i - (ulong)y * (ulong)W);
        float px = 2.0f * float(x) / fw - 1.0f;
        float py = 2.0f * float(y) / fh - 1.0f;
        float dx = px - u, dy = py - v;
        float xr =  dx * ct + dy * st;
        float yr = -dx * st + dy * ct;
        float gg = exp(-xr * xr * 0.5f * inv_s2 - yr * yr * 0.5f * inv_gs2);
        float arg = freq * xr;
        float ca = cos(arg), sa = sin(arg);
        float G0 = grad[gbase + 0ul * HW + i];
        float G1 = grad[gbase + 1ul * HW + i];
        float G2 = grad[gbase + 2ul * HW + i];
        float P = G0 * (Ar*ca - Br*sa) + G1 * (Ag*ca - Bg*sa) + G2 * (Ab*ca - Bb*sa);
        float R = G0 * (Ar*sa + Br*ca) + G1 * (Ag*sa + Bg*ca) + G2 * (Ab*sa + Bb*ca);
        float dL_dxr = -gg * (P * xr * inv_s2 + R * freq);
        float dL_dyr = -gg * P * yr * inv_gs2;
        d[0] += dL_dxr * (-ct) + dL_dyr * st;
        d[1] += dL_dxr * (-st) + dL_dyr * (-ct);
        d[2] += TAU * (dL_dxr * yr - dL_dyr * xr);
        d[3] += P * gg * xr * xr * inv_s2;
        d[4] += P * gg * yr * yr * inv_gs3;
        d[5] += gg * R * freq * xr;
        d[6] += G0 * gg * ca;   d[7] += G1 * gg * ca;   d[8] += G2 * gg * ca;   // A_rgb
        d[9] += -G0 * gg * sa;  d[10] += -G1 * gg * sa; d[11] += -G2 * gg * sa; // B_rgb
    }

    for (uint k = 0u; k < 12u; ++k) d[k] = simd_sum(d[k]);
    if (simd_lane == 0u)
        for (uint k = 0u; k < 12u; ++k) part[simd_id][k] = d[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        device float *o = gparams + ((ulong)b * (ulong)N + (ulong)n) * 12ul;
        for (uint k = 0u; k < 12u; ++k) {
            float s = 0.0f;
            for (uint j = 0u; j < n_simds; ++j) s += part[j][k];
            o[k] = s;
        }
        if (gam <= 1e-3f) o[4] = 0.0f;
    }
}
)METAL";

// ── Cached device + library + pipelines (built once) ─────────────────────────
static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_lib = nil;
static id<MTLComputePipelineState> g_fwd = nil;
static id<MTLComputePipelineState> g_bwd = nil;
static id<MTLComputePipelineState> g_fwd_q = nil;
static id<MTLComputePipelineState> g_bwd_q = nil;
static id<MTLComputePipelineState> g_bwd_s = nil;      // serial reference
static id<MTLComputePipelineState> g_bwd_q_s = nil;

static void ensure_library() {
    if (g_lib) return;
    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) throw std::runtime_error("gabor_metal: no Metal device");
    NSError *err = nil;
    g_lib = [g_device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSrc]
                                   options:nil error:&err];
    if (!g_lib) throw std::runtime_error(std::string("gabor_metal: shader compile: ") +
                                         (err ? err.localizedDescription.UTF8String : "unknown"));
}

static id<MTLComputePipelineState> pipeline(NSString *name, id<MTLComputePipelineState> *cache) {
    if (*cache) return *cache;
    ensure_library();
    id<MTLFunction> fn = [g_lib newFunctionWithName:name];
    if (!fn) throw std::runtime_error(std::string("gabor_metal: kernel not found: ") + name.UTF8String);
    NSError *err = nil;
    *cache = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!*cache) throw std::runtime_error(std::string("gabor_metal: pipeline: ") +
                                          (err ? err.localizedDescription.UTF8String : "unknown"));
    return *cache;
}

static inline id<MTLBuffer> mtl_buffer(const torch::Tensor &t) {
    return __builtin_bit_cast(id<MTLBuffer>, t.storage().data());
}

static inline NSUInteger byte_off(const torch::Tensor &t) {
    return (NSUInteger)(t.storage_offset() * t.element_size());
}

static torch::Tensor run_forward(torch::Tensor params, int64_t H, int64_t W,
                                 id<MTLComputePipelineState> pso) {
    TORCH_CHECK(params.device().is_mps(), "params must be an MPS tensor");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(params.dim() == 3 && params.size(2) == 12, "params must be (B, N, 12)");
    params = params.contiguous();
    const int64_t Bn = params.size(0), Nn = params.size(1);
    torch::Tensor out = torch::empty({Bn, 3, H, W}, params.options());
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLBuffer> pbuf = mtl_buffer(params), obuf = mtl_buffer(out);
    NSUInteger poff = byte_off(params), ooff = byte_off(out);
    uint32_t B=(uint32_t)Bn, N=(uint32_t)Nn, Hh=(uint32_t)H, Ww=(uint32_t)W;
    NSUInteger total = (NSUInteger)(Bn * H * W);
    dispatch_sync(q, ^{
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:pbuf offset:poff atIndex:0];
        [enc setBuffer:obuf offset:ooff atIndex:1];
        [enc setBytes:&B length:4 atIndex:2]; [enc setBytes:&N length:4 atIndex:3];
        [enc setBytes:&Hh length:4 atIndex:4]; [enc setBytes:&Ww length:4 atIndex:5];
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup; if (tpg > 256) tpg = 256;
        [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        [enc endEncoding];
        torch::mps::commit();
    });
    return out;
}

static torch::Tensor run_backward(torch::Tensor params, torch::Tensor grad_out,
                                  int64_t H, int64_t W, id<MTLComputePipelineState> pso) {
    TORCH_CHECK(params.device().is_mps() && grad_out.device().is_mps(), "must be MPS");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    params = params.contiguous();
    grad_out = grad_out.contiguous();
    const int64_t Bn = params.size(0), Nn = params.size(1);
    torch::Tensor gp = torch::empty_like(params);
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLBuffer> pbuf = mtl_buffer(params), gbuf = mtl_buffer(grad_out), obuf = mtl_buffer(gp);
    NSUInteger poff = byte_off(params), goff = byte_off(grad_out), ooff = byte_off(gp);
    uint32_t B=(uint32_t)Bn, N=(uint32_t)Nn, Hh=(uint32_t)H, Ww=(uint32_t)W;
    NSUInteger total = (NSUInteger)(Bn * Nn);
    dispatch_sync(q, ^{
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:pbuf offset:poff atIndex:0];
        [enc setBuffer:gbuf offset:goff atIndex:1];
        [enc setBuffer:obuf offset:ooff atIndex:2];
        [enc setBytes:&B length:4 atIndex:3]; [enc setBytes:&N length:4 atIndex:4];
        [enc setBytes:&Hh length:4 atIndex:5]; [enc setBytes:&Ww length:4 atIndex:6];
        NSUInteger tpg = pso.maxTotalThreadsPerThreadgroup; if (tpg > 256) tpg = 256;
        [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tpg,1,1)];
        [enc endEncoding];
        torch::mps::commit();
    });
    return gp;
}

// Threadgroup-per-(batch,gabor) dispatch for the parallel backward kernels.
// `tg_hint` overrides the threadgroup size; 0 = auto. The override exists so
// the test suite can sweep sizes and catch any stride/threadgroup-size
// mismatch, which fails silently (skipped pixels) rather than erroring.
static torch::Tensor run_backward_tg(torch::Tensor params, torch::Tensor grad_out,
                                     int64_t H, int64_t W,
                                     id<MTLComputePipelineState> pso,
                                     int64_t tg_hint) {
    TORCH_CHECK(params.device().is_mps() && grad_out.device().is_mps(), "must be MPS");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    params = params.contiguous();
    grad_out = grad_out.contiguous();
    const int64_t Bn = params.size(0), Nn = params.size(1);
    torch::Tensor gp = torch::empty_like(params);
    id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
    dispatch_queue_t q = torch::mps::get_dispatch_queue();
    id<MTLBuffer> pbuf = mtl_buffer(params), gbuf = mtl_buffer(grad_out), obuf = mtl_buffer(gp);
    NSUInteger poff = byte_off(params), goff = byte_off(grad_out), ooff = byte_off(gp);
    uint32_t B=(uint32_t)Bn, N=(uint32_t)Nn, Hh=(uint32_t)H, Ww=(uint32_t)W;
    NSUInteger groups = (NSUInteger)(Bn * Nn);
    dispatch_sync(q, ^{
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:pbuf offset:poff atIndex:0];
        [enc setBuffer:gbuf offset:goff atIndex:1];
        [enc setBuffer:obuf offset:ooff atIndex:2];
        [enc setBytes:&B length:4 atIndex:3]; [enc setBytes:&N length:4 atIndex:4];
        [enc setBytes:&Hh length:4 atIndex:5]; [enc setBytes:&Ww length:4 atIndex:6];
        NSUInteger tg = (tg_hint > 0) ? (NSUInteger)tg_hint : 256;
        if (tg > pso.maxTotalThreadsPerThreadgroup) tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg < 1) tg = 1;
        // GM_MAX_SIMDS=32 slots of threadgroup scratch; 32 simds * 32 lanes = 1024.
        if (tg > 1024) tg = 1024;
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1)
              threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc endEncoding];
        torch::mps::commit();
    });
    return gp;
}

torch::Tensor gabor_forward(torch::Tensor p, int64_t H, int64_t W) {
    return run_forward(p, H, W, pipeline(@"gabor_forward", &g_fwd));
}
// Which backward to use. The serial kernel gives B*N threads total; the
// parallel one gives B*N threadgroups. Parallel wins by a lot while B*N alone
// underfills the GPU, and LOSES once it does not, because the simd_sum +
// threadgroup reduction is then pure overhead on a short pixel loop.
//
// Measured crossover is B*N ~= 2800, essentially independent of image size:
//
//        B*N   H*W      serial   parallel   speedup
//        128   16384     2.67ms     0.16ms    16.3x
//       2048   16384     2.86ms     2.03ms     1.41x
//       3072   16384     2.87ms     3.05ms     0.94x
//       8192   16384     4.84ms     8.04ms     0.60x
//        256   65536    10.75ms     1.15ms     9.33x
//       4096   65536    11.01ms    17.26ms     0.64x
//
// So: single-image fitters (torch-imfit, B=1) take the parallel path and gain
// ~9x; large batched evaluation (B=64) stays on serial and is unaffected.
static const int64_t kParallelBackwardMaxBN = 2048;

// tg: 0 = auto (threshold above), >0 = force parallel with that threadgroup
// size, -1 = force serial. Nonzero values exist for the test suite.
torch::Tensor gabor_backward(torch::Tensor p, torch::Tensor g, int64_t H, int64_t W,
                             int64_t tg) {
    const int64_t bn = p.size(0) * p.size(1);
    if (tg < 0 || (tg == 0 && bn > kParallelBackwardMaxBN))
        return run_backward(p, g, H, W, pipeline(@"gabor_backward_serial", &g_bwd_s));
    return run_backward_tg(p, g, H, W, pipeline(@"gabor_backward", &g_bwd), tg);
}
torch::Tensor gabor_forward_quad(torch::Tensor p, int64_t H, int64_t W) {
    return run_forward(p, H, W, pipeline(@"gabor_forward_quad", &g_fwd_q));
}
torch::Tensor gabor_backward_quad(torch::Tensor p, torch::Tensor g, int64_t H, int64_t W,
                                  int64_t tg) {
    const int64_t bn = p.size(0) * p.size(1);
    if (tg < 0 || (tg == 0 && bn > kParallelBackwardMaxBN))
        return run_backward(p, g, H, W, pipeline(@"gabor_backward_quad_serial", &g_bwd_q_s));
    return run_backward_tg(p, g, H, W, pipeline(@"gabor_backward_quad", &g_bwd_q), tg);
}
// Reference implementations: one thread per (batch, gabor). Kept so the fast
// kernels can be diffed against them; not used on any hot path.
torch::Tensor gabor_backward_serial(torch::Tensor p, torch::Tensor g, int64_t H, int64_t W) {
    return run_backward(p, g, H, W, pipeline(@"gabor_backward_serial", &g_bwd_s));
}
torch::Tensor gabor_backward_quad_serial(torch::Tensor p, torch::Tensor g, int64_t H, int64_t W) {
    return run_backward(p, g, H, W, pipeline(@"gabor_backward_quad_serial", &g_bwd_q_s));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gabor_forward, "Fused Gabor render forward (Metal/MPS)");
    m.def("backward", &gabor_backward, "Fused Gabor render backward (Metal/MPS)",
          pybind11::arg("params"), pybind11::arg("grad"), pybind11::arg("H"),
          pybind11::arg("W"), pybind11::arg("tg") = 0);
    m.def("forward_quad", &gabor_forward_quad, "Fused Gabor quad render forward");
    m.def("backward_quad", &gabor_backward_quad, "Fused Gabor quad render backward",
          pybind11::arg("params"), pybind11::arg("grad"), pybind11::arg("H"),
          pybind11::arg("W"), pybind11::arg("tg") = 0);
    m.def("backward_serial", &gabor_backward_serial, "Reference backward (1 thread/gabor)");
    m.def("backward_quad_serial", &gabor_backward_quad_serial, "Reference quad backward");
}
