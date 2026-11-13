"""Thermal neighbour embedding: Fourier-accelerated Langevin sampling of pi(y) ~ exp(-c n KL(P||Q(y))) around a
converged t-SNE map. It tests the Mermin-Wagner predictions sealed in novel/r16b_unaided.md:
  P1 the variance of absolute positions grows with log n, the variance of neighbour displacements does not;
  P2 the posterior displacement spectrum falls like 1/k^2 at small k;
  P3 Fourier-accelerated Langevin decorrelates the slowest map mode >= 5x faster than plain Langevin;
  P4 points with large positional variance are enriched for ambiguous high-dimensional neighbourhoods.
Sampler (unadjusted, preconditioner frozen within a step):
  y <- y - h P grad U + sqrt(2h) P^(1/2) xi,   U = c n KL,   h = lr / (c n)
with P = I + A^T B A, A = D^-1/2 S (cloud-in-cell spread, D = node masses), B = F^-1 (M - 1) F. Noise with covariance
exactly P needs no matrix square root: xi + A^T F^-1 sqrt(M - 1) F zeta, with xi and zeta standard normal.

"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner, attraction, repulsion_grid  # noqa: E402


def _geometry(y, G):
    """The same grid geometry as FourierPreconditioner.__call__ (cloud-in-cell weights, cells, D^-1/2, span)."""
    lo, hi = y.min(0).values, y.max(0).values
    span = (hi - lo).clamp_min(1e-6) * (1.0 + 2.0 / G)
    lo = lo - span / G
    u = (y - lo) / span * (G - 1)
    i0 = u.floor().clamp(0, G - 2).long()
    f = (u - i0).clamp(0, 1)
    w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]]
    cells = [i0[:, 0] * G + i0[:, 1], (i0[:, 0] + 1) * G + i0[:, 1], i0[:, 0] * G + i0[:, 1] + 1,
             (i0[:, 0] + 1) * G + i0[:, 1] + 1]
    mass = torch.zeros(G * G, device=y.device, dtype=y.dtype)
    for wc, cc in zip(w, cells):
        mass.index_add_(0, cc, wc)
    root = mass.clamp_min(1e-12).rsqrt() * (mass > 1e-12)
    return w, cells, root, span


def fourier_noise(y, pc, gen=None):
    """A sample with covariance exactly P (the Fourier preconditioner pc at configuration y), shape like y."""
    G = pc.grid
    w, cells, root, span = _geometry(y, G)
    k = torch.fft.fftfreq(G, device=y.device) * G
    kk = k[:, None] ** 2 + k[None, :] ** 2
    m = pc._multiplier(y, kk, span)
    zeta = torch.randn((2, G, G), generator=gen, device=y.device, dtype=y.dtype)
    eta = torch.fft.ifft2(torch.fft.fft2(zeta) * (m - 1.0).clamp_min(0).sqrt()).real  # covariance B per coordinate
    eta = eta.permute(1, 2, 0).reshape(G * G, 2) * root[:, None]
    out = torch.randn(y.shape, generator=gen, device=y.device, dtype=y.dtype)
    for wc, cc in zip(w, cells):
        out += eta[cc] * wc[:, None]
    return out


def kl_grad(y, aff, grid_h=0.25, max_grid=1536):
    i, j, v = aff
    fa, _ = attraction(y, i, j, v)
    fr, z = repulsion_grid(y, grid_h, max_grid)
    return 4.0 * (fa - fr / z)


def procrustes_rotation(y, ref):
    """Rotate centred y onto centred ref (no scaling); returns the aligned, centred copy of y."""
    a, b = y - y.mean(0), ref - ref.mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    r = u @ vt
    if torch.det(r) < 0:
        u[:, -1] = -u[:, -1]
        r = u @ vt
    return a @ r


def displacement_spectrum(u, y_ref, G=128):
    """Radially averaged |u_k|^2 of a displacement field u living on the points y_ref (cloud-in-cell, mass-normalised)."""
    w, cells, root, _ = _geometry(y_ref, G)
    field = torch.zeros(G * G, 2, device=u.device, dtype=u.dtype)
    mass = torch.zeros(G * G, device=u.device, dtype=u.dtype)
    for wc, cc in zip(w, cells):
        field.index_add_(0, cc, u * wc[:, None])
        mass.index_add_(0, cc, wc)
    field = (field / mass.clamp_min(1e-12)[:, None] * (mass > 1e-12)[:, None]).reshape(G, G, 2)
    spec = (torch.fft.fft2(field.permute(2, 0, 1)).abs() ** 2).sum(0) / (G * G)
    k = torch.fft.fftfreq(G, device=u.device) * G
    kr = (k[:, None] ** 2 + k[None, :] ** 2).sqrt().round().long()
    out = torch.zeros(G // 2, device=u.device, dtype=u.dtype)
    cnt = torch.zeros(G // 2, device=u.device, dtype=u.dtype)
    keep = (kr > 0) & (kr < G // 2)
    out.index_add_(0, kr[keep], spec[keep])
    cnt.index_add_(0, kr[keep], torch.ones_like(spec[keep]))
    return out / cnt.clamp_min(1)


def lowest_mode_amplitude(u, y_ref):
    """Amplitude of the m = 1 shear mode of the displacement field: u_x modulated by sin(2 pi y / L) along y."""
    yc = y_ref - y_ref.mean(0)
    L = float(yc[:, 1].max() - yc[:, 1].min())
    return float((u[:, 0] * torch.sin(2 * math.pi * yc[:, 1] / L)).mean())


def langevin(y0, aff, *, c=1.0, lr=None, n_steps=2000, burn=500, pc=None, every=10, track=None, seed=0,
             grid_h=0.25, max_grid=1536, nbr=None):
    """Sample around y0. c = observations per point (inverse temperature per point); lr as in t-SNE (default n/12).
    pc: FourierPreconditioner or None (plain Langevin). track: indices whose trajectories are kept.
    nbr: (n_track, k) map-neighbour indices of the tracked points for relative displacements.
    Returns a dict of time series and per-point statistics."""
    n = y0.shape[0]
    lr = lr or n / 12.0
    h = lr / (c * n)
    gen = torch.Generator(device=y0.device).manual_seed(seed)
    y = y0.clone()
    ref = y0 - y0.mean(0)
    track = track if track is not None else torch.arange(min(n, 20000), device=y0.device)
    s1 = torch.zeros(track.shape[0], 2, device=y0.device)
    s2 = torch.zeros(track.shape[0], device=y0.device)
    r2 = torch.zeros(track.shape[0], device=y0.device)
    r1 = torch.zeros(track.shape[0], 2, device=y0.device)
    amp, msd, spec, kept = [], [], None, 0
    for t in range(burn + n_steps):
        g = kl_grad(y, aff, grid_h, max_grid) * (c * n)
        drift = pc(y, g) if pc is not None else g
        noise = fourier_noise(y, pc, gen) if pc is not None else torch.randn(y.shape, generator=gen, device=y.device)
        y = y - h * drift + math.sqrt(2.0 * h) * noise
        if t >= burn and (t - burn) % every == 0:
            ya = procrustes_rotation(y, ref)
            u = ya - ref
            amp.append(lowest_mode_amplitude(u, ref))
            ut = u[track]
            msd.append(float((ut * ut).sum(1).mean()))
            s1 += ut
            s2 += (ut * ut).sum(1)
            if nbr is not None:
                rel = ut - u[nbr].mean(1)
                r1 += rel
                r2 += (rel * rel).sum(1)
            sp = displacement_spectrum(u, ref)
            spec = sp if spec is None else spec + sp
            kept += 1
    mean, rmean = s1 / kept, r1 / kept  # both are variances about the time mean (static drift from y0 removed)
    return {"var_abs": (s2 / kept - (mean * mean).sum(1)).cpu(),
            "var_rel": (r2 / kept - (rmean * rmean).sum(1)).cpu() if nbr is not None else None,
            "amp_m1": amp, "msd": msd, "spectrum": (spec / kept).cpu(), "h": h, "kept": kept}


def autocorr_time(a):
    """Integrated autocorrelation time (Sokal's automatic window, c = 5)."""
    x = torch.as_tensor(a, dtype=torch.float64)
    x = x - x.mean()
    n = x.shape[0]
    f = torch.fft.rfft(x, n=2 * n)
    acf = torch.fft.irfft(f * f.conj(), n=2 * n)[:n]
    acf = acf / acf[0]
    tau = 1.0
    for w in range(1, n):
        tau = 1.0 + 2.0 * float(acf[1:w + 1].sum())
        if w >= 5 * tau:
            break
    return tau
