"""Optimiser-level competitors of Fourier-accelerated t-SNE, implemented on the SAME affinities and grid repulsion so that
only the optimiser differs.

lambert: Lambert, Lee, Couplet, de Bodt (ESANN 2023), "Nesterov momentum and gradient normalization ... without early
  exaggeration": no early exaggeration, PCA-whitened initialisation (unit std per axis), Nesterov momentum gamma = 0.995,
  eta = 1, gradient rescaled to norm sqrt(2N)/100 before each momentum update (paper section 3).
ars: Lu and Calder, "Attraction-Repulsion Swarming" (Phil. Trans. A 2025), ARS_{2,2}, taken from the authors' code
  (github.com/jwcalder/AttractionRepulsionSwarming-tSNE, tsne.cpp / sptree.cpp): P row-normalised after symmetrisation;
  attraction f+_i = sum_j P_ij w_ij (y_i - y_j); repulsion normalised per point f-_i = sum_j w_ij^2 (y_i - y_j) / sum_j w_ij;
  update y <- y - h (alpha f+ - f-), h = 1, recentred every step; defaults alpha = 1 (no exaggeration), init N(0, 1e-4^2).
Both return (y, history) with history[it] = grid KL (t-SNE objective, for comparison), like fatsne.tsne.
"""

import math

import torch

from fatsne import attraction, kl_divergence, repulsion_exact, repulsion_grid


def _kl_grid(y, i, j, v, grid_h, max_grid):
    _, z = repulsion_grid(y, grid_h, max_grid)
    return kl_divergence(y, i, j, v, torch.tensor(z, device=y.device, dtype=torch.float64))


def lambert(x, aff, *, n_iter=3000, seed=0, record=(), grid_h=0.25, max_grid=1536, callback=None, exact=False,
            precond=None):
    """precond (r30): optional SPD operator P(y, g) applied to the look-ahead gradient before the normalisation."""
    i, j, v = aff
    n = x.shape[0]
    xc = x - x.mean(0)
    _, _, vt = torch.linalg.svd(xc[: min(n, 20000)], full_matrices=False)
    y = xc @ vt[:2].T
    g = torch.Generator(device=x.device).manual_seed(seed)
    y = y + 1e-3 * y.std(0) * torch.randn(y.shape, generator=g, device=x.device)  # seeded jitter, as for all arms
    y = (y - y.mean(0)) / y.std(0)  # PCA whitening: unit standard deviation along both axes
    vel = torch.zeros_like(y)
    gamma, eta, scale = 0.995, 1.0, math.sqrt(2 * n) / 100.0
    hist = {}
    for it in range(n_iter):
        yl = y + gamma * vel  # Nesterov look-ahead
        fa, _ = attraction(yl, i, j, v)
        fr, z = repulsion_exact(yl) if exact else repulsion_grid(yl, grid_h, max_grid)
        grad = 4.0 * (fa - fr / z)
        if precond is not None:
            grad = precond(yl, grad)
        grad = grad * (scale / (grad.norm() + 1e-30))
        vel = gamma * vel - eta * grad
        y = y + vel
        if callback is not None:
            callback(it + 1, y)
        if it + 1 in record:
            hist[it + 1] = _kl_grid(y, i, j, v, grid_h, max_grid)
    return y, hist


def ars(x, aff, *, n_iter=3000, seed=0, record=(), grid_h=0.25, max_grid=1536, h=1.0, alpha=1.0, num_early=100,
        callback=None):
    i, j, v = aff
    n = x.shape[0]
    rows = torch.zeros(n, device=x.device, dtype=v.dtype).index_add_(0, i, v)
    vr = v / rows[i]  # row-normalised P (each row sums to 1)
    g = torch.Generator(device=x.device).manual_seed(seed)
    y = 1e-4 * torch.randn((n, 2), generator=g, device=x.device)
    hist = {}
    for it in range(n_iter):
        diff = y[i] - y[j]
        w = 1.0 / (1.0 + (diff * diff).sum(1))
        fpos = torch.zeros_like(y).index_add_(0, i, diff * (vr * w)[:, None])
        fr, _, wsum = repulsion_grid(y, grid_h, max_grid, per_point=True)
        fneg = fr / wsum.clamp_min(1e-30)[:, None]
        a = alpha if it < num_early else 1.0
        y = y - h * (a * fpos - fneg)
        y = y - y.mean(0)
        if callback is not None:
            callback(it + 1, y)
        if it + 1 in record:
            hist[it + 1] = _kl_grid(y, i, j, v, grid_h, max_grid)
    return y, hist
