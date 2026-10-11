# Vendored: gabor_metal

Source: `/Users/andrewbenson/jitter-vision/tools/shared/gabor_metal/`
Vendored: 2026-10-10

Fused differentiable Gabor render (forward + analytic backward) as a Metal
kernel callable on PyTorch MPS tensors. Replaces the dense
`[N,3,H,W]` broadcast with O(1) intermediate memory.

Copied verbatim — `gabor_metal.mm` and `__init__.py`. If the upstream copy
changes, re-copy both files; there are no local edits.

Re-vendored 2026-10-10 after the threadgroup-parallel backward landed upstream
(one threadgroup per gabor + simd_sum; 43.4 ms -> 4.84 ms at B=1, N=256,
512px). See `BACKWARD_PORT_PLAN.md` in the upstream directory.
changes, re-copy both files; there are no local edits.

Param convention (std-physical, `(B,N,12)`):

    0=u  1=v  2=theta(turns)  3=log_sigma  4=gamma  5=rel_freq
    6..8=psi_rgb  9..11=amp_rgb

    sigma = exp(log_sigma)      freq  = 2*pi / exp(rel_freq)
    theta *= 2*pi               phase = psi * 2*pi
    gamma denominator floored at 1e-3; RAW sum, no output clamp.

Coordinate grid is `2*x/(W-1) - 1`, i.e. identical to
`torch.linspace(-1, 1, W)`.

Requires the Xcode Metal toolchain; the extension JIT-compiles on first use
(~30 s) and is then cached by torch.
