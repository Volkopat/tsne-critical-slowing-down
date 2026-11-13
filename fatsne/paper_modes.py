"""Paper, block C (mechanism; sealed predictions in novel/r12_unaided.md). Relaxation spectrum of t-SNE
optimizers. A T2I subset (n = 100k) is optimised to near convergence (Fourier, 6000 iterations). The
converged embedding y* is perturbed by one displacement mode of wavenumber index m across the embedding
(wavelength = span / m): transverse (displacement perpendicular to the wave vector, shear) or longitudinal
(parallel, compression), amplitude 2% of the wavelength. Each optimizer (std, tuned, fourier; no early
exaggeration, fresh gains and momentum) runs 60 iterations from y* + u; the mode amplitude
a(t) = <y(t) - y_ctrl(t), u> / <u, u>, with y_ctrl the same optimizer started from the unperturbed y*,
is recorded every iteration; the decay rate per iteration is fitted to log|a| over iterations 10-60.
Prediction: std's rate grows ~ m^2 at small m (long modes relax slowly), Fourier's is nearly flat.
One GPU alone.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner, affinities, tsne  # noqa: E402
from paperlib import grid_rule, load_any  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
M = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
T = 60


def path(x, aff, y0, kw):
    traj = []
    tsne(x, affin=aff, seed=0, init=y0, early_iter=0, n_iter=T, record=(), repulsion="grid", grid_h=0.25, max_grid=1536,
         callback=lambda it, y: traj.append(y.clone()), **kw)
    return traj


def main():
    n = 100_000
    x = torch.as_tensor(load_any("T2I", n), device="cuda")
    x = x - x.mean(0)
    aff = affinities(x, 30.0)
    G = grid_rule(n)
    ystar, hist, _ = tsne(x, affin=aff, seed=0, init_jitter=0.1, n_iter=5750, record=(6000,), repulsion="grid", grid_h=0.25,
                          max_grid=1536, precond=FourierPreconditioner(G, 16.0))
    print(f"  converged reference: KL {hist[6000][0]:.4f}", flush=True)
    c = ystar.mean(0)
    span = float((ystar.max(0).values - ystar.min(0).values).max())
    out = {"n": n, "span": span, "kl_ref": hist[6000][0], "grid": G, "rows": []}
    arms = {"std": lambda: {}, "tuned": lambda: dict(lr_late=float(n), momentum_late=0.9),
            "fourier": lambda: dict(precond=FourierPreconditioner(G, 16.0))}
    ctrl = {a: path(x, aff, ystar, f()) for a, f in arms.items()}
    for m in M:
        k = 2 * math.pi * m / span
        for pol in ("T", "L"):
            u = torch.zeros_like(ystar)
            u[:, 1 if pol == "T" else 0] = torch.sin(k * (ystar[:, 0] - c[0]))
            amp = 0.02 * span / m
            uu = float((u * u).sum())
            for arm, f in arms.items():
                traj = path(x, aff, ystar + amp * u, f())
                a = np.array([float(((yt - yc) * u).sum()) / uu / amp for yt, yc in zip(traj, ctrl[arm])])
                tt = np.arange(1, T + 1)
                sel = (tt >= 10) & (np.abs(a) > 1e-4)
                rate = float(-np.polyfit(tt[sel], np.log(np.abs(a[sel])), 1)[0]) if sel.sum() >= 5 else float("nan")
                out["rows"].append({"m": m, "pol": pol, "arm": arm, "k": k, "a": a.tolist(), "rate": rate})
                print(f"  m {m:2d} {pol} {arm:8s} a(10) {a[9]:+.4f} a(30) {a[29]:+.4f} a(60) {a[59]:+.4f}  rate {rate:.4f}", flush=True)
        (OUT / "paper_modes.json").write_text(json.dumps(out, indent=1))
    for pol in ("T", "L"):
        for arm in arms:
            rows = [r for r in out["rows"] if r["pol"] == pol and r["arm"] == arm and r["m"] <= 8 and r["rate"] > 0]
            if len(rows) >= 3:
                s = np.polyfit(np.log([r["m"] for r in rows]), np.log([r["rate"] for r in rows]), 1)[0]
                print(f"  {pol} {arm:8s}: rate ~ m^{s:.2f} for m <= 8", flush=True)


if __name__ == "__main__":
    main()
