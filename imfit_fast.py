"""imfit_fast — alternate ImageFitter engine with a fused Metal Gabor render.

Drop-in replacement for `imfit.ImageFitter`: same constructor signature, same
methods, same `.txt` weight format, same `--init` warm-start chain. Select it
from training.py with `--engine fast`.

WHY
---
`imfit.GaborLayer.forward` builds `amplitude * gaussian * sinusoid` at shape
[N, 3, H, W] and then sums over N. At N=256 / 512px that single intermediate is
805 MB, with another 805 MB for `sinusoid` and 268 MB for `gaussian` — roughly
2 GB per forward and again per backward. It is both slow and the reason large
fits run out of memory.

This engine swaps that broadcast for the fused Metal kernel in `gabor_metal/`
(forward + analytic backward, O(1) intermediate memory, ~39x faster on MPS).
Everything else — losses, constraints, save/load, rescaling, image export — is
inherited unchanged from `imfit.ImageFitter`, so the two engines are
behaviourally identical with all guards off.

GUARDS
------
Optional fit guardrails ported from jitter-vision `tools/data/fit_guards.py`.
All default OFF, so `--engine fast` alone changes speed and nothing else.

  band_dilate_px   Mip-Splatting band-limit: sigma_eff = sqrt(sigma^2 + d^2).
                   Suppresses sub-pixel "needle" atoms. Differentiable, applied
                   during assembly, so autograd supplies the chain rule free.
  freq_floor_px    Minimum carrier period in pixels (post-step rel_freq clamp).
  amp_penalty      Penalty on amplitude carried by sub-pixel-thin atoms.
  group_lr         Per-parameter-group learning rates (position moves faster
                   than geometry, geometry faster than frequency).
  relocate_every   Every K steps, move negligible-contribution atoms to
                   high-residual locations and reset their Adam moments.
  relocate_until   Stop relocating after this fraction of the run so the final
                   cohort can converge. Long budgets otherwise accumulate dead
                   atoms: measured 3.1% -> 11.2% going from 200 to 1600 steps.

CONVENTION NOTES
----------------
Two differences from the vendored kernel are adapted here, so the on-disk
parameter format is unchanged:

  * Column 3 is `rel_sigma` (direct) in this repo but `log_sigma` in the
    kernel. Converted during assembly; autograd handles the inverse.
  * The kernel floors the gamma denominator at 1e-3 and zeroes its gradient
    below that. `imfit.GaborLayer` clamps gamma to [1e-4, 1]. This engine
    tightens that clamp to [1e-3, 1] so no atom sits in the frozen-gradient
    band. That is the one intentional behavioural difference from `imfit.py`.
"""
from __future__ import annotations

import math
import os
import subprocess
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.optim as optim

from imfit import GABOR_MAX, GABOR_MIN, GaborLayer
from imfit import ImageFitter as _BaseImageFitter

TAU = 2.0 * math.pi

# Relative per-group learning rates. jitter-vision fit_guards specifies these as
# absolute LRs against its own base of 0.05; stored here as ratios so they scale
# with whatever --global-lr is passed.
_LR_MULT = {
    'u':         0.040 / 0.05,
    'v':         0.040 / 0.05,
    'theta':     0.020 / 0.05,
    'rel_sigma': 0.020 / 0.05,
    'gamma':     0.020 / 0.05,
    'rel_freq':  0.015 / 0.05,
    'psi':       0.050 / 0.05,
    'amplitude': 0.040 / 0.05,
}
_PARAM_ORDER = ('u', 'v', 'theta', 'rel_sigma', 'gamma', 'rel_freq',
                'psi', 'amplitude')


@dataclass
class Guards:
    """Fit guardrails. All zero/off by default."""
    band_dilate_px:  float = 0.0
    freq_floor_px:   float = 0.0
    amp_penalty:     float = 0.0
    group_lr:        bool  = False
    relocate_every:  int   = 0
    relocate_thresh: float = 1e-3
    relocate_blur:   int   = 5
    relocate_until:  float = 0.75

    @classmethod
    def from_args(cls, args) -> 'Guards':
        """Build from an argparse namespace, ignoring unrelated attributes."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in vars(args).items()
                      if k in names and v is not None})

    def any_enabled(self) -> bool:
        return bool(self.band_dilate_px or self.freq_floor_px or self.amp_penalty
                    or self.group_lr or self.relocate_every)


# ── rendering ─────────────────────────────────────────────────────────────────

def render_dense(p12: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reference render matching gabor_metal exactly. (N,12) -> (3,H,W).

    Used as the non-MPS fallback and as the parity reference for the kernel.
    Materializes the same [N,3,H,W] intermediate the classic engine does, so it
    is only appropriate for small fits.
    """
    device = p12.device
    ys = torch.linspace(-1.0, 1.0, H, device=device, dtype=p12.dtype)
    xs = torch.linspace(-1.0, 1.0, W, device=device, dtype=p12.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')

    u, v = p12[:, 0, None, None], p12[:, 1, None, None]
    th = p12[:, 2, None, None] * TAU
    sigma = torch.exp(p12[:, 3, None, None])
    gam = p12[:, 4, None, None].clamp_min(1e-3)
    freq = TAU / torch.exp(p12[:, 5, None, None])

    dx, dy = gx[None] - u, gy[None] - v
    ct, st = torch.cos(th), torch.sin(th)
    xr = dx * ct + dy * st
    yr = -dx * st + dy * ct

    gauss = torch.exp(-(xr ** 2) / (2 * sigma ** 2) - (yr ** 2) / (2 * gam ** 2))
    phase = p12[:, 6:9, None, None] * TAU
    sinus = torch.cos(freq[:, None] * xr[:, None] + phase)
    return (p12[:, 9:12, None, None] * gauss[:, None] * sinus).sum(0)


_toolchain_checked = False


def _ensure_toolchain():
    """Point the extension build at a usable compiler.

    An Xcode upgrade invalidates the license agreement, after which `c++` exits
    69 on every invocation and the extension cannot JIT-compile. The Command
    Line Tools clang is not gated that way and builds this extension fine (the
    Metal shader is compiled at runtime from source, so only the frameworks are
    needed). Prefer whatever is already configured; fall back to the CLT.
    """
    global _toolchain_checked
    if _toolchain_checked or 'DEVELOPER_DIR' in os.environ:
        return
    _toolchain_checked = True
    try:
        if subprocess.run(['c++', '--version'], capture_output=True).returncode == 0:
            return
    except OSError:
        return
    clt = '/Library/Developer/CommandLineTools'
    if os.path.isdir(clt):
        os.environ['DEVELOPER_DIR'] = clt
        print("[imfit_fast] active Xcode toolchain is unusable (license not "
              "accepted for the installed version); building against the "
              "Command Line Tools instead.\n"
              "             To use Xcode's own: sudo xcodebuild -license accept")


def render(p12: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Fused Metal render on MPS, dense torch fallback elsewhere."""
    if p12.device.type == 'mps':
        _ensure_toolchain()
        from gabor_metal import render_gabors_metal
        return render_gabors_metal(p12[None], H, W)[0]
    return render_dense(p12, H, W)


# ── model ─────────────────────────────────────────────────────────────────────

class FastGaborLayer(GaborLayer):
    """GaborLayer with the dense broadcast replaced by the fused kernel.

    Parameters stay as the same eight separate nn.Parameters, so `pos_check`,
    `save_weights`, `load_weights`, `mutate_parameters` and
    `enforce_parameter_ranges` all work untouched. The (N,12) tensor the kernel
    wants is assembled per step with a `cat` — 12*N floats, free at these sizes,
    and differentiable, so gradients flow back to the eight leaves.
    """

    def __init__(self, num_gabors: int = 256, guards: Guards | None = None):
        super().__init__(num_gabors)
        self.guards = guards or Guards()

    def enforce_parameter_ranges(self):
        super().enforce_parameter_ranges()
        # The kernel floors the gamma denominator at 1e-3 and zeroes its
        # gradient below; keep every atom out of that frozen band.
        with torch.no_grad():
            self.gamma.clamp_(1e-3, 1)

    def assemble(self, res: int) -> torch.Tensor:
        """Pack the eight parameters into the kernel's (N,12) std-physical
        layout, applying any differentiable guards on the way."""
        sigma, gam = self.rel_sigma, self.gamma
        d = self.guards.band_dilate_px / float(res) if self.guards.band_dilate_px else 0.0
        if d:
            # Mip-Splatting band-limit. A pixel spans 2/res in [-1,1], so
            # band_dilate_px=1.0 gives half a pixel of extra envelope std.
            sigma = torch.sqrt(sigma ** 2 + d * d)
            gam = torch.sqrt(gam ** 2 + d * d)
        return torch.cat([
            self.u[:, None],
            self.v[:, None],
            self.theta[:, None],
            torch.log(sigma.clamp_min(1e-6))[:, None],   # col 3: kernel wants log
            gam[:, None],
            self.rel_freq[:, None],
            self.psi,
            self.amplitude,
        ], dim=1)

    def forward(self, grid_x, grid_y):
        H, W = grid_x.shape
        self.enforce_parameter_ranges()
        out = render(self.assemble(max(H, W)), H, W)
        return torch.clamp(out, -1, 1)


# ── fitter ────────────────────────────────────────────────────────────────────

class ImageFitter(_BaseImageFitter):
    """ImageFitter driving FastGaborLayer. Signature matches imfit.ImageFitter;
    `guards` is the only addition and defaults to everything off."""

    def __init__(self, image_path, weight_path=None, num_gabors=256,
                 target_size=None, device='cuda', init=None, global_lr=0.03,
                 debug=None, mutation_strength=0.01, gamma=0.85, sobel=0.,
                 gradient=0., l1=0., guards: Guards | None = None):
        self.guards = guards or Guards()
        # Build the base fitter with init deferred: it would otherwise load the
        # weights into the model we are about to replace.
        super().__init__(image_path, weight_path, num_gabors, target_size,
                         device, None, global_lr, debug, mutation_strength,
                         gamma, sobel, gradient, l1)

        fast = FastGaborLayer(num_gabors, guards=self.guards).to(device)
        fast.load_state_dict(self.model.state_dict())
        self.model = fast
        if init:
            self.init_parameters(init)
        self.init_optimizer(global_lr)

        self._relocated = 0
        if device != 'mps':
            print(f"[imfit_fast] device={device}: Metal kernel unavailable, "
                  f"using the dense fallback render (no speedup).")

    # ── optimizer: one param group per tensor so group LRs are expressible ──
    def init_optimizer(self, global_lr):
        groups, self._lr_mult = [], []
        for name in _PARAM_ORDER:
            mult = _LR_MULT[name] if self.guards.group_lr else 1.0
            groups.append({'params': [getattr(self.model, name)],
                           'lr': global_lr * mult})
            self._lr_mult.append(mult)
        self.optimizer = optim.AdamW(groups, lr=global_lr, weight_decay=1e-5,
                                     betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=self.gamma, patience=20)
        self._lr_base = global_lr

    def _sync_group_lrs(self):
        """Re-apply the per-group multipliers.

        training.py resets every group to `--global-lr` between scales, which
        flattens them. Because the multipliers differ, all-groups-equal can only
        mean that reset just happened, so that value is the new base; otherwise
        recover the base from group 0.
        """
        if not self.guards.group_lr:
            return
        lrs = [g['lr'] for g in self.optimizer.param_groups]
        self._lr_base = (lrs[0] if len(set(lrs)) == 1
                         else lrs[0] / self._lr_mult[0])
        for g, m in zip(self.optimizer.param_groups, self._lr_mult):
            g['lr'] = self._lr_base * m

    # ── guard losses ────────────────────────────────────────────────────────
    def loss_function(self, output, target):
        loss = super().loss_function(output, target)
        if self.guards.amp_penalty > 0:
            res = max(target.shape[-2:])
            gamma_floor = 2.0 / res
            amp = self.model.amplitude.abs().mean(-1)
            short = (gamma_floor - self.model.gamma).clamp(min=0)
            loss = loss + self.guards.amp_penalty * (amp * short).mean()
        return loss

    # ── G3: relocate parked atoms ───────────────────────────────────────────
    @torch.no_grad()
    def _relocate(self, output):
        g = self.guards
        m = self.model
        proxy = m.amplitude.abs().mean(-1) * m.rel_sigma * m.gamma
        parked = proxy < g.relocate_thresh
        k = int(parked.sum().item())
        if k == 0:
            return 0

        H, W = self.target.shape[-2:]
        heat = (self.target - output).abs().mean(0)
        if g.relocate_blur > 0:
            kk = g.relocate_blur * 2 + 1
            heat = torch.nn.functional.avg_pool2d(
                heat[None, None], kk, stride=1, padding=g.relocate_blur)[0, 0]
        heat = torch.nan_to_num(heat, 0.0, 0.0, 0.0).clamp_min(0) + 1e-8
        flat = (heat / heat.sum()).reshape(-1)

        # torch.multinomial over H*W categories segfaults on MPS — draw on CPU
        # and move the chosen pixel indices back.
        pix = torch.multinomial(flat.float().cpu(), k, replacement=True).to(m.u.device)
        py, px = pix // W, pix % W
        # load_weights leaves CPU params as float64; match whatever we have.
        tk = dict(device=m.u.device, dtype=m.u.dtype)
        xs = torch.linspace(-1.0, 1.0, W, **tk)
        ys = torch.linspace(-1.0, 1.0, H, **tk)

        idx = parked.nonzero(as_tuple=True)[0]
        m.u[idx] = xs[px]
        m.v[idx] = ys[py]
        for name in ('theta', 'rel_sigma', 'rel_freq', 'gamma'):
            lo, hi = GABOR_MIN[name], GABOR_MAX[name]
            getattr(m, name)[idx] = torch.rand(k, **tk) * (hi - lo) + lo
        # small colour so relocated atoms have to earn their amplitude back
        m.psi[idx] = (torch.rand((k, 3), **tk) - 0.5) * 0.1
        m.amplitude[idx] = torch.rand((k, 3), **tk) * 0.05

        for grp in self.optimizer.param_groups:
            for p in grp['params']:
                st = self.optimizer.state.get(p)
                if st:
                    st['exp_avg'][idx] = 0
                    st['exp_avg_sq'][idx] = 0
        return k

    # ── training step ───────────────────────────────────────────────────────
    def train_step(self, iteration, max_iterations, save_best=False):
        self._sync_group_lrs()
        self.update_temperature(iteration, max_iterations)
        self.mutate_parameters()

        self.optimizer.zero_grad()
        output = self.model(self.grid_x, self.grid_y)
        loss = self.loss_function(output, self.target)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss)

        g = self.guards
        if g.freq_floor_px > 0:
            # period of freq_floor_px pixels, in [-1,1] coordinate units
            res = max(self.target.shape[-2:])
            with torch.no_grad():
                self.model.rel_freq.clamp_(min=math.log(g.freq_floor_px * 2.0 / res))

        if (g.relocate_every > 0 and iteration > 0
                and iteration % g.relocate_every == 0
                and iteration < g.relocate_until * max_iterations):
            self._relocated += self._relocate(output.detach())

        if iteration < 10:
            self.pos_check(self.weights)

        if loss.item() < self.best_loss and save_best:
            self.best_loss = loss.item()
            self.best_state = self.model.state_dict()

        return loss.item()
