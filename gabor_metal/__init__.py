"""Fused differentiable Gabor render on Apple GPU (Metal) for PyTorch MPS.

Phase 1: forward-only parity spike. `render_forward(params, H, W)` mirrors
`tools.shared.render.render_gabors` (std-physical params, (B,3,H,W) output) but
runs a single fused Metal kernel instead of materializing the (B,N,HW,3) torch
intermediate. Backward + autograd.Function wrapping come in Phase 2.

The extension is JIT-compiled on first use via torch.utils.cpp_extension.load
(needs the Xcode Metal toolchain). Compiled artifact is cached by torch.
"""
from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(__file__)
_mod = None


def _ext():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load
        _mod = load(
            name='gabor_metal_ext',
            sources=[os.path.join(_HERE, 'gabor_metal.mm')],
            extra_cflags=['-std=c++17', '-O3'],
            extra_ldflags=['-framework', 'Metal', '-framework', 'Foundation'],
            verbose=True,
        )
    return _mod


def render_forward(params: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Fused Metal forward render. params: (B, N, 12) MPS float32 std-physical."""
    return _ext().forward(params.contiguous(), int(H), int(W))


class _GaborRenderMetal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, H, W):
        out = _ext().forward(params.contiguous(), int(H), int(W))
        ctx.save_for_backward(params)
        ctx.hw = (int(H), int(W))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (params,) = ctx.saved_tensors
        H, W = ctx.hw
        gp = _ext().backward(params.contiguous(), grad_out.contiguous(), H, W)
        return gp, None, None


def render_gabors_metal(params: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Differentiable fused Metal render — drop-in for tools.shared.render.
    render_gabors on MPS. params: (B, N, 12) std-physical. Returns (B, 3, H, W)."""
    return _GaborRenderMetal.apply(params, H, W)


class _GaborRenderMetalQuad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, H, W):
        out = _ext().forward_quad(params.contiguous(), int(H), int(W))
        ctx.save_for_backward(params)
        ctx.hw = (int(H), int(W))
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (params,) = ctx.saved_tensors
        H, W = ctx.hw
        gp = _ext().backward_quad(params.contiguous(), grad_out.contiguous(), H, W)
        return gp, None, None


def render_gabors_quad_metal(params: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Differentiable fused Metal render for QUADRATURE-form params (A=6:9, B=9:12).
    Drop-in for render_gabors_quad on MPS. Color block linear in (A,B) — no
    phase-wrap. params: (B, N, 12). Returns (B, 3, H, W)."""
    return _GaborRenderMetalQuad.apply(params, H, W)
