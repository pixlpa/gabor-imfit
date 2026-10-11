"""Parity + speed check for the fast engine.  python3 test_engines.py

Verifies that imfit_fast matches imfit, that the guards run, and — when the
Metal kernel is available — that the kernel matches the dense reference and
how much faster it is.
"""
import os
import sys
import tempfile
import time

import numpy as np
import torch
from PIL import Image

import imfit
import imfit_fast

N, S = 48, 64
OUT = tempfile.mkdtemp(prefix='imfit_parity_')
FAIL = []


def check(name, cond, detail=''):
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}{'   ' + detail if detail else ''}")
    if not cond:
        FAIL.append(name)


def make_target():
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, S), torch.linspace(-1, 1, S),
                            indexing='ij')
    img = torch.stack([(torch.sin(6 * xx) * torch.cos(5 * yy) + 1) / 2,
                       torch.exp(-(xx ** 2 + yy ** 2) * 3),
                       ((xx > 0) ^ (yy > 0)).float() * 0.8 + 0.1])
    p = os.path.join(OUT, 'target.png')
    Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(p)
    return p


def make_init():
    torch.manual_seed(1234)
    layer = imfit.GaborLayer(N)
    layer.enforce_parameter_ranges()
    p = os.path.join(OUT, 'init.txt')
    stub = imfit.ImageFitter.__new__(imfit.ImageFitter)
    stub.model = layer
    imfit.ImageFitter.save_weights(stub, p)
    return p


def engines(img, init, dev):
    def build(cls, **kw):
        torch.manual_seed(0)
        return cls(img, None, N, S, dev, init, global_lr=0.01, debug=None,
                   mutation_strength=0.0, gamma=0.997, sobel=0., gradient=0.,
                   l1=0., **kw)
    return build(imfit.ImageFitter), build(imfit_fast.ImageFitter), build


def main():
    img, init = make_target(), make_init()

    print("\n[engine parity, cpu]")
    a, b, build = engines(img, init, 'cpu')
    with torch.no_grad():
        nlow = int((a.model.gamma < 1e-3).sum())
        # The engines differ by exactly one documented thing: the fast engine
        # floors gamma at 1e-3 because the kernel zeroes its gradient below.
        a.model.gamma.clamp_(1e-3, 1)
        d = (a.model(a.grid_x, a.grid_y) - b.model(b.grid_x, b.grid_y)).abs().max().item()
    check('render matches with gamma floor matched', d < 1e-4,
          f"max|d|={d:.2e}  ({nlow}/{N} atoms init below 1e-3)")

    a.init_optimizer(0.01)
    b.init_optimizer(0.01)
    la = [a.train_step(i, 60) for i in range(60)]
    lb = [b.train_step(i, 60) for i in range(60)]
    check('step-0 loss identical', abs(la[0] - lb[0]) < 1e-5,
          f"{la[0]:.6f} / {lb[0]:.6f}")
    check('both descend', la[-1] < la[0] / 2 and lb[-1] < lb[0] / 2,
          f"{la[-1]:.6f} / {lb[-1]:.6f}")

    pa, pb = os.path.join(OUT, 'a.txt'), os.path.join(OUT, 'b.txt')
    a.save_weights(pa)
    b.save_weights(pb)
    wa, wb = np.genfromtxt(pa, delimiter=','), np.genfromtxt(pb, delimiter=',')
    check('weight format identical', wa.shape == wb.shape == (N, 12), str(wa.shape))

    print("\n[guards]")
    g = imfit_fast.Guards(band_dilate_px=1.0, freq_floor_px=2.5, amp_penalty=0.02,
                          group_lr=True, relocate_every=10, relocate_until=0.75)
    c = build(imfit_fast.ImageFitter, guards=g)
    c.init_optimizer(0.01)
    lc = [c.train_step(i, 60) for i in range(60)]
    check('guarded run is finite and descends',
          all(np.isfinite(lc)) and lc[-1] < lc[0], f"{lc[0]:.6f} -> {lc[-1]:.6f}")
    check('relocation fired', c._relocated > 0, f"{c._relocated} atoms")
    lrs = [round(x['lr'], 6) for x in c.optimizer.param_groups]
    check('group lrs differentiated', len(set(lrs)) > 1, str(lrs))
    # training.py flattens every group to --global-lr between scales
    for grp in c.optimizer.param_groups:
        grp['lr'] = 0.01
    c.train_step(60, 120)
    lrs2 = [round(x['lr'], 6) for x in c.optimizer.param_groups]
    check('group lrs survive training.py reset', len(set(lrs2)) > 1, str(lrs2))

    print("\n[metal kernel]")
    if not torch.backends.mps.is_available():
        print("  skip  no MPS device")
    else:
        try:
            torch.manual_seed(0)
            p = torch.zeros(N, 12)
            p[:, 0].uniform_(-.9, .9); p[:, 1].uniform_(-.9, .9)
            p[:, 2].uniform_(0, 1);    p[:, 3].uniform_(-3.5, -1.5)
            p[:, 4].uniform_(.02, .4); p[:, 5].uniform_(-2.5, -.5)
            p[:, 6:9].uniform_(-1, 1); p[:, 9:12].uniform_(-.4, .4)
            p = p.to('mps')
            H, W = 97, 131                      # non-square, odd
            ref = imfit_fast.render_dense(p, H, W)
            got = imfit_fast.render(p.clone(), H, W)
            torch.mps.synchronize()
            check('kernel matches dense render',
                  (ref - got).abs().max().item() < 1e-4,
                  f"max|d|={(ref - got).abs().max().item():.2e}")

            w = torch.randn(3, H, W, device='mps')
            p1, p2 = p.clone().requires_grad_(), p.clone().requires_grad_()
            (imfit_fast.render_dense(p1, H, W) * w).sum().backward()
            (imfit_fast.render(p2, H, W) * w).sum().backward()
            torch.mps.synchronize()
            rel = ((p1.grad - p2.grad).abs() / (p1.grad.abs() + 1e-6)).max().item()
            check('kernel gradients match', rel < 1e-2, f"max rel err={rel:.2e}")

            q = torch.randn(256, 12, device='mps') * 0.3
            q[:, 3], q[:, 4], q[:, 5] = -2.5, 0.1, -1.0
            for fn, nm in ((imfit_fast.render_dense, 'dense'),
                           (imfit_fast.render, 'metal')):
                try:
                    fn(q, 512, 512); torch.mps.synchronize()
                    t = time.perf_counter()
                    for _ in range(10):
                        fn(q, 512, 512)
                    torch.mps.synchronize()
                    print(f"  fwd N=256 512px {nm}  {(time.perf_counter()-t)/10*1e3:7.2f} ms")
                except RuntimeError as e:
                    print(f"  fwd N=256 512px {nm}  failed: {str(e)[:60]}")
        except Exception as e:
            msg = str(e)
            if 'Xcode license' in msg or "'c++'" in msg:
                print("  skip  Metal extension cannot build: the Xcode license has "
                      "not been accepted.\n        Run: sudo xcodebuild -license accept")
            else:
                raise

    print(f"\n{'FAILED: ' + ', '.join(FAIL) if FAIL else 'all checks passed'}")
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
