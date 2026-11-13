"""Fourier-accelerated t-SNE (research code, novel_engines round 1, candidate C1).

t-SNE needs hundreds of iterations mostly because collective, long-wavelength motions of the
embedding (clusters drifting apart, the layout expanding) are driven by weak long-range forces,
while local motions are fast: the "critical slowing down" of lattice field theory. Lattice
practice (hep-lat/9306018, 2108.05486) and Fourier-accelerated Langevin dynamics of particles
(cond-mat/0001418) rescale every spatial frequency of the update so that slow modes relax as fast
as fast ones. Here the same is done in the 2-D embedding plane:

    g_i            t-SNE gradient of point i (exact attraction on the kNN graph, exact or grid repulsion)
    G(x)           per-cell average of the gradients, spread on a grid by bilinear weights
    d_i = g_i + interp_i( F^-1[ (M(k) - 1) F[G] ] ),   M(k) = min(cap, (k_max/|k|)^s) for k != 0, M(0) = 1

so grid-scale detail is the raw gradient and coarser modes are amplified by up to `cap`. The
operator is I + S^T D^-1/2 A D^-1/2 S with A positive semidefinite (symmetric form below), so d is
a descent direction. Everything else is the standard optimizer (momentum, gains, learning rate
n/12, early exaggeration), so that the comparison isolates the preconditioner.

Predictions were sealed before each experiment (see preregistration/).
"""

from __future__ import annotations

import math
import time

import torch


# ------------------------------------------------------------------ affinities
def knn(x, k, chunk=None):
    """Exact k nearest neighbours (excluding self) and squared distances, on x's device."""
    n = x.shape[0]
    chunk = chunk or max(1, min(4096, 2**28 // n))  # rows per block: at most 2^28 distances in memory
    xn = (x * x).sum(1)
    idx = torch.empty(n, k, dtype=torch.int64, device=x.device)
    d2 = torch.empty(n, k, dtype=x.dtype, device=x.device)
    for b in range(0, n, chunk):
        s = xn[b : b + chunk, None] - 2.0 * x[b : b + chunk] @ x.T + xn[None, :]
        s[torch.arange(s.shape[0], device=x.device), torch.arange(b, b + s.shape[0], device=x.device)] = math.inf
        v, i = torch.topk(s, k, dim=1, largest=False)
        idx[b : b + chunk], d2[b : b + chunk] = i, v.clamp_min(0)
    return idx, d2


def affinities(x, perplexity=30.0, k=None):
    """Symmetric t-SNE input affinities P (sum 1) on the 3*perplexity nearest neighbours, as edge lists."""
    n = x.shape[0]
    k = k or min(n - 1, int(3 * perplexity))
    idx, d2 = knn(x, k)
    d2 = d2.double()
    target = math.log(perplexity)
    lo = torch.full((n,), -50.0, dtype=torch.float64, device=x.device)  # log beta bounds
    hi = torch.full((n,), 50.0, dtype=torch.float64, device=x.device)
    d2n = d2 - d2[:, :1]  # shift for stability
    for _ in range(64):
        mid = (lo + hi) / 2
        w = torch.exp(-d2n * mid.exp()[:, None])
        p = w / w.sum(1, keepdim=True)
        h = -(p * torch.log(p.clamp_min(1e-300))).sum(1)
        too_flat = h > target  # entropy too high -> increase beta
        lo = torch.where(too_flat, mid, lo)
        hi = torch.where(too_flat, hi, mid)
    rows = torch.arange(n, device=x.device).repeat_interleave(k)
    cols = idx.reshape(-1)
    vals = p.reshape(-1)
    P = torch.sparse_coo_tensor(torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])]),
                                torch.cat([vals, vals]), (n, n)).coalesce()
    i, j = P.indices()
    v = P.values() / (2.0 * n)
    return i, j, v.float()


# ------------------------------------------------------------------ gradient
def attraction(y, i, j, v):
    diff = y[i] - y[j]
    q = 1.0 / (1.0 + (diff * diff).sum(1))
    f = torch.zeros_like(y).index_add_(0, i, diff * (v * q)[:, None])
    return f, q


def repulsion_exact(y, chunk=8192):
    """Returns (sum_j q~_ij^2 (y_i - y_j) for every i, Z = sum_{i != j} q~_ij)."""
    n = y.shape[0]
    f = torch.empty_like(y)
    z = 0.0
    yn = (y * y).sum(1)
    for b in range(0, n, chunk):
        yb = y[b : b + chunk]
        d = (yn[b : b + chunk, None] - 2.0 * yb @ y.T + yn[None, :]).clamp_min(0)
        q = 1.0 / (1.0 + d)
        z += float(q.sum()) - yb.shape[0]  # remove the diagonal (q~_ii = 1)
        q2 = q * q
        f[b : b + chunk] = yb * (q2.sum(1, keepdim=True) - 1.0) - (q2 @ y - yb)
    return f, z


def _spread_weights(y, lo, h, G):
    u = (y - lo) / h
    i0 = u.floor().long().clamp(0, G - 2)
    f = (u - i0).clamp(0, 1)
    w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]]
    c = [i0[:, 0] * G + i0[:, 1], (i0[:, 0] + 1) * G + i0[:, 1], i0[:, 0] * G + i0[:, 1] + 1, (i0[:, 0] + 1) * G + i0[:, 1] + 1]
    return w, c


def repulsion_grid(y, h_target=0.2, max_grid=1024, per_point=False):
    """Grid (particle-mesh) evaluation of the repulsion: point masses and y-weighted masses spread
    on a grid by bilinear weights, convolved with q~ and q~^2 by zero-padded FFT, interpolated back.
    Error controlled by the grid step h (the kernel width is 1). Cost O(n + G^2 log G)."""
    n = y.shape[0]
    lo = y.min(0).values - 2 * h_target
    hi = y.max(0).values + 2 * h_target
    span = float((hi - lo).max())
    G = int(min(max_grid, max(16, math.ceil(span / h_target) + 2)))
    h = span / (G - 2)
    w, c = _spread_weights(y, lo, h, G)
    src = torch.zeros(G * G, 3, device=y.device, dtype=torch.float32)
    vals = torch.cat([torch.ones(n, 1, device=y.device), y], 1)
    for wc, cc in zip(w, c):
        src.index_add_(0, cc, vals * wc[:, None])
    P = 2 * G
    src = torch.nn.functional.pad(src.reshape(G, G, 3).permute(2, 0, 1), (0, G, 0, G))
    r = torch.arange(P, device=y.device, dtype=torch.float32)
    r = torch.minimum(r, P - r) * h
    d2 = r[:, None] ** 2 + r[None, :] ** 2
    k1 = 1.0 / (1.0 + d2)
    k2 = k1 * k1
    S = torch.fft.rfft2(src)
    phi1 = torch.fft.irfft2(S[:1] * torch.fft.rfft2(k1), s=(P, P))[0, :G, :G].reshape(-1)
    phi2 = torch.fft.irfft2(S * torch.fft.rfft2(k2), s=(P, P))[:, :G, :G].reshape(3, -1).T
    at1 = sum(phi1[cc] * wc for wc, cc in zip(w, c))
    at2 = sum(phi2[cc] * wc[:, None] for wc, cc in zip(w, c))
    z = float(at1.sum()) - n
    f = y * (at2[:, :1] - 1.0) - (at2[:, 1:] - y)
    if per_point:  # also each point's own kernel sum sum_j w_ij (needed by force-normalised competitors)
        return f, z, at1 - 1.0
    return f, z


def exact_z(y, chunk=None):
    """Exact t-SNE normaliser Z = sum_{i != j} 1 / (1 + |y_i - y_j|^2): float32 tiles (TF32 off), float64 sums.
    About 1e12 pair terms per second-scale on a B200; used to report KL without the grid approximation."""
    n = y.shape[0]
    chunk = chunk or max(256, min(8192, 2 ** 30 // max(n, 1)))
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yc = (y - y.mean(0)).float()
    yy = (yc * yc).sum(1)
    z = 0.0
    for s0 in range(0, n, chunk):
        d2 = (yy[s0:s0 + chunk, None] - 2.0 * yc[s0:s0 + chunk] @ yc.T + yy[None, :]).clamp_min(0)
        z += float((1.0 / (1.0 + d2)).sum(dtype=torch.float64)) - min(chunk, n - s0)
    torch.backends.cuda.matmul.allow_tf32 = prev
    return z


def kl_divergence(y, i, j, v, z):
    """KL(P || Q) over the edges of P (terms with p = 0 contribute 0), in float64."""
    keep = v > 0
    diff = (y[i[keep]] - y[j[keep]]).double()
    q = 1.0 / (1.0 + (diff * diff).sum(1)) / z
    vv = v[keep].double()
    return float((vv * torch.log(vv / q.clamp_min(1e-300))).sum())


# ------------------------------------------------------------------ Fourier preconditioner
class FourierPreconditioner:
    def __init__(self, grid=128, cap=16.0, power=2.0):
        self.grid, self.cap, self.power = int(grid), float(cap), float(power)

    def __call__(self, y, g):
        G = self.grid
        lo = y.min(0).values
        hi = y.max(0).values
        span = (hi - lo).clamp_min(1e-6) * (1.0 + 2.0 / G)
        lo = lo - span / G
        u = (y - lo) / span * (G - 1)  # grid coordinates in [0, G-1]
        i0 = u.floor().clamp(0, G - 2).long()
        f = (u - i0).clamp(0, 1)
        w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]]
        cells = [i0[:, 0] * G + i0[:, 1], (i0[:, 0] + 1) * G + i0[:, 1], i0[:, 0] * G + i0[:, 1] + 1,
                 (i0[:, 0] + 1) * G + i0[:, 1] + 1]
        mass = torch.zeros(G * G, device=y.device, dtype=y.dtype)
        field = torch.zeros(G * G, 2, device=y.device, dtype=y.dtype)
        for wc, cc in zip(w, cells):
            mass.index_add_(0, cc, wc)
            field.index_add_(0, cc, g * wc[:, None])
        root = mass.clamp_min(1e-12).rsqrt() * (mass > 1e-12)
        field = (field * root[:, None]).reshape(G, G, 2)  # D^-1/2 S g
        k = torch.fft.fftfreq(G, device=y.device) * G
        kk = k[:, None] ** 2 + k[None, :] ** 2
        m = self._multiplier(y, kk, span)
        spec = torch.fft.fft2(field.permute(2, 0, 1))
        boosted = torch.fft.ifft2(spec * (m - 1.0)).real.permute(1, 2, 0).reshape(G * G, 2)
        boosted = boosted * root[:, None]  # D^-1/2 again (symmetric form)
        out = g.clone()
        for wc, cc in zip(w, cells):
            out += boosted[cc] * wc[:, None]  # S^T
        return out

    def _multiplier(self, y, kk, span):
        # grid-anchored (original): M = min(cap, (kmax/k)^2), kmax at the grid corner. NB (adversary B2):
        # M >= 2 up to the axis Nyquist, mean ~5 at cap 16, so this is partly a scalar step increase.
        kmax2 = float(kk.max())
        m = torch.where(kk > 0, (kmax2 / kk.clamp_min(1e-12)) ** (self.power / 2), torch.ones_like(kk)).clamp(max=self.cap)
        m[0, 0] = 1.0
        return m


class AnchoredFourier(FourierPreconditioner):
    """Fourier acceleration anchored to the physics instead of the grid: M(k) = clamp(beta / (k l)^2, 1, cap),
    l = median embedding length of a fixed random sample of P's edges (the kNN spring length). Modes
    shorter than the spring length get M = 1 (no step increase); only collective modes are boosted."""

    def __init__(self, grid, cap, i, j, beta=1.0, sample=65536, seed=0):
        super().__init__(grid, cap)
        g = torch.Generator(device=i.device).manual_seed(seed)
        pick = torch.randint(0, i.shape[0], (min(sample, i.shape[0]),), generator=g, device=i.device)
        self.ei, self.ej, self.beta = i[pick], j[pick], float(beta)
        self.last_l = None

    def _multiplier(self, y, kk, span):
        l = float((y[self.ei] - y[self.ej]).norm(dim=1).median())
        self.last_l = l
        G = kk.shape[0]
        mi = torch.fft.fftfreq(G, device=y.device) * G
        fx, fy = (2 * math.pi * (G - 1) / (G * span)).tolist()  # physical k per grid index, per axis
        kl2 = ((mi[:, None] * fx) ** 2 + (mi[None, :] * fy) ** 2) * l * l
        m = (self.beta / kl2.clamp_min(1e-30)).clamp(1.0, self.cap)
        m[0, 0] = 1.0
        return m


class GraphSobolev:
    """Meshless Fourier acceleration: out = g + (cap - 1) (I + lam L)^-1 g, where L is the Laplacian of the
    symmetrised k-nearest-neighbour graph of the CURRENT embedding (a proxy for its Delaunay graph).
    In the continuum, (I + lam L)^-1 has symbol 1 / (1 + lam (k l)^2), so long modes are boosted by cap
    and short ones by ~1, with the crossover at ~sqrt(lam) neighbour hops; unlike the FFT grid, the
    resolution follows the local density. Symmetric positive-definite. The graph is rebuilt every
    `refresh` calls; the solve is warm-started CG with `cg` iterations."""

    def __init__(self, cap=16.0, lam=16.0, k=8, refresh=10, cg=8, chunk=4096):
        self.cap, self.lam, self.k, self.refresh, self.cg, self.chunk = float(cap), float(lam), int(k), int(refresh), int(cg), int(chunk)
        self.calls, self.L, self.x0 = 0, None, None

    def _graph(self, y):
        n = y.shape[0]
        idx = torch.empty(n, self.k, dtype=torch.long, device=y.device)
        yy = (y * y).sum(1)
        for s in range(0, n, self.chunk):
            d = yy[s:s + self.chunk, None] - 2 * y[s:s + self.chunk] @ y.T + yy[None, :]
            d[torch.arange(d.shape[0], device=y.device), torch.arange(s, s + d.shape[0], device=y.device)] = float("inf")
            idx[s:s + self.chunk] = d.topk(self.k, 1, largest=False).indices
        rows = torch.arange(n, device=y.device).repeat_interleave(self.k)
        cols = idx.reshape(-1)
        W = torch.sparse_coo_tensor(torch.stack([torch.cat([rows, cols]), torch.cat([cols, rows])]),
                                    torch.ones(2 * rows.numel(), device=y.device), (n, n)).coalesce()
        W = torch.sparse_coo_tensor(W.indices(), torch.ones_like(W.values()), (n, n)).coalesce()  # unweighted, symmetric
        deg = torch.sparse.sum(W, 1).to_dense()
        return W, deg

    def __call__(self, y, g):
        if self.L is None or self.calls % self.refresh == 0:
            self.L = self._graph(y)
        self.calls += 1
        W, deg = self.L

        def A(v):  # (I + lam L) v
            return v + self.lam * (deg[:, None] * v - torch.sparse.mm(W, v))

        x = self.x0 if self.x0 is not None and self.x0.shape == g.shape else g.clone()
        r = g - A(x)
        p = r.clone()
        rs = (r * r).sum(0)
        for _ in range(self.cg):  # CG on both columns at once (independent systems)
            Ap = A(p)
            a = rs / (p * Ap).sum(0).clamp_min(1e-30)
            x = x + a * p
            r = r - a * Ap
            rs_new = (r * r).sum(0)
            p = r + (rs_new / rs.clamp_min(1e-30)) * p
            rs = rs_new
        self.x0 = x
        return g + (self.cap - 1.0) * x


class AggregationBoost:
    """Two-level additive preconditioner from multigrid Monte Carlo / smoothed-aggregation AMG:
    out = inner(g) + (cap - 1) * sum_c 1_c (1_c^T g) / |c|, i.e. the per-cluster mean of the gradient
    (the force moving a whole cluster rigidly) is boosted. Clusters = k-means (a few Lloyd steps,
    warm-started) of the CURRENT embedding with K centres, refreshed every `refresh` calls. The added
    term is a projection, so the operator stays symmetric positive-definite. Targets the slow mode
    the grid cannot see: whole clusters drifting apart."""

    def __init__(self, K=64, cap=16.0, refresh=25, inner=None, lloyd=5):
        self.K, self.cap, self.refresh, self.inner, self.lloyd = int(K), float(cap), int(refresh), inner, int(lloyd)
        self.calls, self.labels, self.cent = 0, None, None

    def _cluster(self, y):
        n = y.shape[0]
        if self.cent is None:
            g = torch.Generator(device=y.device).manual_seed(0)
            self.cent = y[torch.randperm(n, generator=g, device=y.device)[: self.K]].clone()
        for _ in range(self.lloyd):
            lab = torch.cdist(y, self.cent).argmin(1)
            cnt = torch.bincount(lab, minlength=self.K).float()
            s = torch.zeros_like(self.cent).index_add_(0, lab, y)
            self.cent = torch.where(cnt[:, None] > 0, s / cnt.clamp_min(1)[:, None], self.cent)
        self.labels = torch.cdist(y, self.cent).argmin(1)

    def __call__(self, y, g):
        if self.labels is None or self.calls % self.refresh == 0:
            self._cluster(y)
        self.calls += 1
        out = self.inner(y, g) if self.inner is not None else g
        cnt = torch.bincount(self.labels, minlength=self.K).float().clamp_min(1)
        mean = torch.zeros(self.K, 2, device=g.device, dtype=g.dtype).index_add_(0, self.labels, g) / cnt[:, None]
        return out + (self.cap - 1.0) * mean[self.labels]


class MultilevelAggregation:
    """Grid-free multilevel acceleration (BPX-style additive preconditioner in aggregation form, after
    multigrid Monte Carlo): nested clusterings of the CURRENT embedding with K_0 > K_1 > ... centres
    (k-means on points, then k-means on count-weighted centroids); level l has RMS cluster radius R_l
    and boost b_l = min(cap, (K_0 / K_l)^(power/2)) (rule "count", points per cluster; 2-D elastic
    scaling rate ~ 1/R^2 ~ 1/N) or min(cap, (R_l / R_0)^power) (rule "radius"), b_0 = 1. With P_l the cluster-mean projection,
        out = inner(g) + sum_{l>=1} (b_l - b_{l-1}) P_l g,
    so a field that is piecewise constant on level-l clusters (but not coarser) is scaled by b_l: the
    1/k^2 law with "wavelength" = cluster size, adapted to the local density. Nested projections with
    increasing b_l keep the operator symmetric positive-definite."""

    def __init__(self, levels=(2048, 512, 128, 32, 8), cap=16.0, refresh=25, inner=None, power=2.0, lloyd=4, rule="count"):
        self.levels, self.cap, self.refresh, self.inner = tuple(int(k) for k in levels), float(cap), int(refresh), inner
        self.power, self.lloyd, self.rule = float(power), int(lloyd), rule
        self.calls, self.labs, self.boost, self.cents = 0, None, None, None

    @staticmethod
    def _kmeans(p, wts, K, cent, iters):
        for _ in range(iters):
            lab = torch.cdist(p, cent).argmin(1)
            m = torch.zeros(K, device=p.device).index_add_(0, lab, wts)
            s = torch.zeros(K, 2, device=p.device).index_add_(0, lab, p * wts[:, None])
            cent = torch.where(m[:, None] > 0, s / m.clamp_min(1e-12)[:, None], cent)
        return torch.cdist(p, cent).argmin(1), cent

    def _build(self, y):
        n = y.shape[0]
        g = torch.Generator(device=y.device).manual_seed(0)
        if self.cents is None:
            self.cents = []
        labs, radii = [], []
        pts, wts, prev = y, torch.ones(n, device=y.device), None
        for li, K in enumerate(self.levels):
            K = min(K, pts.shape[0])
            cent = self.cents[li] if li < len(self.cents) and self.cents[li].shape[0] == K else \
                pts[torch.randperm(pts.shape[0], generator=g, device=y.device)[:K]].clone()
            lab, cent = self._kmeans(pts, wts, K, cent, self.lloyd)
            if li < len(self.cents):
                self.cents[li] = cent
            else:
                self.cents.append(cent)
            full = lab if prev is None else lab[prev]  # point -> level-li cluster
            cnt = torch.bincount(full, minlength=K).float().clamp_min(1)
            mu = torch.zeros(K, 2, device=y.device).index_add_(0, full, y) / cnt[:, None]
            radii.append(float(((y - mu[full]) ** 2).sum(1).mean().sqrt()))
            labs.append((full, K))
            m = torch.zeros(K, device=y.device).index_add_(0, lab, wts)
            pts, wts, prev = cent, m, full
        self.labs = labs
        if self.rule == "radius":
            b = [1.0] + [min(self.cap, (r / max(radii[0], 1e-12)) ** self.power) for r in radii[1:]]
        else:  # count: 2-D elastic scaling, rate ~ 1/R^2 ~ 1/N, so boost ~ points per cluster relative to level 0
            k0 = labs[0][1]
            b = [1.0] + [min(self.cap, (k0 / K) ** (self.power / 2)) for _, K in labs[1:]]
        for i in range(1, len(b)):
            b[i] = max(b[i], b[i - 1])
        self.boost = b

    def __call__(self, y, g):
        if self.labs is None or self.calls % self.refresh == 0:
            self._build(y)
        self.calls += 1
        out = self.inner(y, g) if self.inner is not None else g.clone()
        for li in range(1, len(self.labs)):
            inc = self.boost[li] - self.boost[li - 1]
            if inc <= 0:
                continue
            full, K = self.labs[li]
            cnt = torch.bincount(full, minlength=K).float().clamp_min(1)
            mean = torch.zeros(K, 2, device=g.device, dtype=g.dtype).index_add_(0, full, g) / cnt[:, None]
            out = out + inc * mean[full]
        return out


class Safeguarded:
    """out = g + s (P(g) - g) with s in (0, 1]: scales only the boost part of a symmetric positive-definite
    preconditioner P, so the result stays SPD. The optimizer lowers s when its objective rises (see
    tsne(safeguard=True)): function-value adaptive restart (O'Donoghue and Candes) applied to the boost."""

    def __init__(self, inner, regrow=1.02):
        self.inner, self.s, self.regrow, self.cuts = inner, 1.0, float(regrow), 0

    def __call__(self, y, g):
        if self.s >= 0.999:
            return self.inner(y, g)
        return g + self.s * (self.inner(y, g) - g)

    def cut(self):
        self.s *= 0.5
        self.cuts += 1

    def grow(self):
        self.s = min(1.0, self.s * self.regrow)


class HelmholtzPreconditioner(FourierPreconditioner):
    """Fourier acceleration split by the Helmholtz decomposition of the update field.

    At scales above 1 the Cauchy repulsion acts like a two-dimensional Coulomb plasma: the
    longitudinal (compression) modes of a 2-D one-component plasma have a restoring rate that does
    not vanish as k -> 0, while the transverse (shear) modes are held only by the elastic springs
    of the kNN attraction, rate ~ k^2. So only the transverse part is slow, and amplifying the
    longitudinal part as 1/k^2 over-drives it (the instability seen at large caps). Here the
    transverse part gets min(cap_t, (k_max/k)^2) and the longitudinal part min(cap_l, (k_max/k)^2).
    """

    def __init__(self, grid=128, cap_t=64.0, cap_l=4.0, power=2.0):
        super().__init__(grid, cap_t, power)
        self.cap_t, self.cap_l = float(cap_t), float(cap_l)

    def __call__(self, y, g):
        G = self.grid
        lo = y.min(0).values
        hi = y.max(0).values
        span = (hi - lo).clamp_min(1e-6) * (1.0 + 2.0 / G)
        lo = lo - span / G
        u = (y - lo) / span * (G - 1)
        i0 = u.floor().clamp(0, G - 2).long()
        f = (u - i0).clamp(0, 1)
        w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]]
        cells = [i0[:, 0] * G + i0[:, 1], (i0[:, 0] + 1) * G + i0[:, 1], i0[:, 0] * G + i0[:, 1] + 1,
                 (i0[:, 0] + 1) * G + i0[:, 1] + 1]
        mass = torch.zeros(G * G, device=y.device, dtype=y.dtype)
        field = torch.zeros(G * G, 2, device=y.device, dtype=y.dtype)
        for wc, cc in zip(w, cells):
            mass.index_add_(0, cc, wc)
            field.index_add_(0, cc, g * wc[:, None])
        root = mass.clamp_min(1e-12).rsqrt() * (mass > 1e-12)
        field = (field * root[:, None]).reshape(G, G, 2)
        k = torch.fft.fftfreq(G, device=y.device) * G
        kx, ky = k[:, None].expand(G, G), k[None, :].expand(G, G)
        kk = kx**2 + ky**2
        base = torch.where(kk > 0, float(kk.max()) / kk.clamp_min(1e-12), torch.ones_like(kk))
        mt = base.clamp(max=self.cap_t) - 1.0
        ml = base.clamp(max=self.cap_l) - 1.0
        mt[0, 0] = 0.0
        ml[0, 0] = 0.0
        spec = torch.fft.fft2(field.permute(2, 0, 1))  # (2, G, G): components x, y
        inv = 1.0 / kk.clamp_min(1e-12)
        dot = (kx * spec[0] + ky * spec[1]) * inv  # (k . F) / |k|^2
        lx, ly = kx * dot, ky * dot  # longitudinal part
        tx, ty = spec[0] - lx, spec[1] - ly  # transverse part
        bx = mt * tx + ml * lx
        by = mt * ty + ml * ly
        boosted = torch.fft.ifft2(torch.stack([bx, by])).real.permute(1, 2, 0).reshape(G * G, 2)
        boosted = boosted * root[:, None]
        out = g.clone()
        for wc, cc in zip(w, cells):
            out += boosted[cc] * wc[:, None]
        return out


class AdaptiveHelmholtzPreconditioner(HelmholtzPreconditioner):
    """Tuning-free variant: the boost of each (radial frequency band, Helmholtz part) is the ratio
    of the stiffness measured in the finest band to the stiffness measured in that band, where a
    band's stiffness is the secant (Barzilai-Borwein) estimate <dG, dY> / <dY, dY> from successive
    gradient fields G and displacement fields Y on the grid, smoothed by a moving average. Boost
    clipped to [1, cap_max]; bands whose secant is not positive keep their previous boost.
    """

    def __init__(self, grid=128, bands=8, cap_max=256.0, memory=0.9):
        super().__init__(grid, 1.0, 1.0)
        self.bands, self.cap_max, self.memory = int(bands), float(cap_max), float(memory)
        self.prev = None  # (grid spectra of gradient field and position field, frame)
        self.stiff = None
        self.boost = None

    def _grid_fields(self, y, g, lo, span):
        G = self.grid
        u = (y - lo) / span * (G - 1)
        i0 = u.floor().clamp(0, G - 2).long()
        f = (u - i0).clamp(0, 1)
        w = [(1 - f[:, 0]) * (1 - f[:, 1]), f[:, 0] * (1 - f[:, 1]), (1 - f[:, 0]) * f[:, 1], f[:, 0] * f[:, 1]]
        cells = [i0[:, 0] * G + i0[:, 1], (i0[:, 0] + 1) * G + i0[:, 1], i0[:, 0] * G + i0[:, 1] + 1,
                 (i0[:, 0] + 1) * G + i0[:, 1] + 1]
        mass = torch.zeros(G * G, device=y.device, dtype=y.dtype)
        fg = torch.zeros(G * G, 4, device=y.device, dtype=y.dtype)
        vals = torch.cat([g, y], 1)
        for wc, cc in zip(w, cells):
            mass.index_add_(0, cc, wc)
            fg.index_add_(0, cc, vals * wc[:, None])
        root = mass.clamp_min(1e-12).rsqrt() * (mass > 1e-12)
        spec = torch.fft.fft2((fg * root[:, None]).reshape(G, G, 4).permute(2, 0, 1))
        return spec, w, cells, root

    def __call__(self, y, g):
        G = self.grid
        lo = y.min(0).values
        span = (y.max(0).values - lo).clamp_min(1e-6) * (1.0 + 2.0 / G)
        lo = lo - span / G
        spec, w, cells, root = self._grid_fields(y, g, lo, span)
        k = torch.fft.fftfreq(G, device=y.device) * G
        kx, ky = k[:, None].expand(G, G), k[None, :].expand(G, G)
        kk = kx**2 + ky**2
        kr = kk.sqrt()
        edges = torch.logspace(0, math.log10(float(kr.max()) + 1e-9), self.bands + 1, device=y.device)
        band = torch.bucketize(kr, edges).clamp(1, self.bands) - 1  # 0 .. bands-1, DC goes to band 0
        inv = 1.0 / kk.clamp_min(1e-12)

        def split(sx, sy):
            dot = (kx * sx + ky * sy) * inv
            return (kx * dot, ky * dot), (sx - kx * dot, sy - ky * dot)

        gL, gT = split(spec[0], spec[1])
        yL, yT = split(spec[2], spec[3])
        if self.boost is None:
            self.boost = torch.ones(2, self.bands, device=y.device)
            self.stiff = torch.zeros(2, self.bands, device=y.device)
        if self.prev is not None:
            for p, (gp, yp, gc, yc) in enumerate(((self.prev[0], self.prev[1], gL, yL), (self.prev[2], self.prev[3], gT, yT))):
                dg = [a - b for a, b in zip(gc, gp)]
                dy = [a - b for a, b in zip(yc, yp)]
                num = (dg[0] * dy[0].conj() + dg[1] * dy[1].conj()).real
                den = (dy[0].abs() ** 2 + dy[1].abs() ** 2)
                nb = torch.zeros(self.bands, device=y.device).index_add_(0, band.reshape(-1), num.reshape(-1))
                db = torch.zeros(self.bands, device=y.device).index_add_(0, band.reshape(-1), den.reshape(-1))
                est = nb / db.clamp_min(1e-30)
                ok = (est > 0) & (db > 0)
                self.stiff[p] = torch.where(ok, torch.where(self.stiff[p] > 0, self.memory * self.stiff[p] + (1 - self.memory) * est, est), self.stiff[p])
            ref = self.stiff[:, -1].max()
            if float(ref) > 0:
                new = (ref / self.stiff.clamp_min(1e-30)).clamp(1.0, self.cap_max)
                self.boost = torch.where(self.stiff > 0, new, self.boost)
        self.prev = (gL, yL, gT, yT)
        mL = self.boost[0][band] - 1.0
        mT = self.boost[1][band] - 1.0
        mL[0, 0] = 0.0
        mT[0, 0] = 0.0
        bx = mT * gT[0] + mL * gL[0]
        by = mT * gT[1] + mL * gL[1]
        boosted = torch.fft.ifft2(torch.stack([bx, by])).real.permute(1, 2, 0).reshape(G * G, 2) * root[:, None]
        out = g.clone()
        for wc, cc in zip(w, cells):
            out += boosted[cc] * wc[:, None]
        return out


class SpectralDirection:
    """Baseline from Vladymyrov and Carreira-Perpinan (2012): precondition by the attraction part of
    the Hessian, B = 4 L(P) + mu I with L the graph Laplacian of P, solved by a few conjugate-gradient
    steps (Jacobi preconditioned) warm-started from the previous solution. Returned direction is
    rescaled to the norm of the gradient so that it can drive the same momentum/gains optimizer."""

    def __init__(self, i, j, v, n, mu=1e-5, cg_iters=10):
        self.i, self.j, self.v, self.n = i, j, v, n
        self.deg = torch.zeros(n, device=v.device).index_add_(0, i, v)
        self.mu, self.cg_iters, self.prev = mu * float(self.deg.mean()), int(cg_iters), None

    def _apply(self, x):
        lx = self.deg[:, None] * x - torch.zeros_like(x).index_add_(0, self.i, x[self.j] * self.v[:, None])
        return 4.0 * lx + self.mu * x

    def __call__(self, y, g):
        diag = (4.0 * self.deg + self.mu)[:, None]
        x = torch.zeros_like(g) if self.prev is None else self.prev
        r = g - self._apply(x)
        z = r / diag
        p = z.clone()
        rz = (r * z).sum(0)
        for _ in range(self.cg_iters):
            ap = self._apply(p)
            a = rz / (p * ap).sum(0).clamp_min(1e-30)
            x = x + a * p
            r = r - a * ap
            z = r / diag
            rz_new = (r * z).sum(0)
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        self.prev = x
        return x * (g.norm() / x.norm().clamp_min(1e-30))


def tsne_spectral_direction(x, *, n_iter=750, early_iter=250, exaggeration=12.0, perplexity=30.0, seed=0, record=(),
                            affin=None, device="cuda", repulsion="exact", grid_h=0.2, max_grid=1024, cg_iters=20):
    """Faithful baseline: Vladymyrov and Carreira-Perpinan (ICML 2012), spectral direction with a
    backtracking (Armijo) line search, after the same early-exaggeration phase as standard t-SNE.
    history[iteration] = (KL, seconds); one iteration = one direction (several KL evaluations)."""
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    n = x.shape[0]
    i, j, v = affinities(x, perplexity) if affin is None else affin
    rep = (lambda yy: repulsion_exact(yy)) if repulsion == "exact" else (lambda yy: repulsion_grid(yy, grid_h, max_grid))
    # early exaggeration exactly as in tsne()
    y, _, _ = tsne(x, n_iter=0, early_iter=early_iter, exaggeration=exaggeration, perplexity=perplexity, seed=seed,
                   affin=(i, j, v), device=device, repulsion=repulsion, grid_h=grid_h, max_grid=max_grid)
    sd = SpectralDirection(i, j, v, n, cg_iters=cg_iters)
    history, start = {}, time.perf_counter()

    def objective(yy):
        _, z = rep(yy)
        return kl_divergence(yy, i, j, v, z)

    f = objective(y)
    for it in range(n_iter):
        fa, _ = attraction(y, i, j, v)
        fr, z = rep(y)
        g = 4.0 * (fa - fr / z)
        # direction: exact solve of B d = g (no rescaling), then descend along -d
        x0 = torch.zeros_like(g) if sd.prev is None else sd.prev  # warm start from the last direction
        diag = (4.0 * sd.deg + sd.mu)[:, None]
        r = g - sd._apply(x0)
        zz = r / diag
        p = zz.clone()
        rz = (r * zz).sum(0)
        dd = x0
        for _ in range(cg_iters):
            ap = sd._apply(p)
            a = rz / (p * ap).sum(0).clamp_min(1e-30)
            dd = dd + a * p
            r = r - a * ap
            zz = r / diag
            rz_new = (r * zz).sum(0)
            p = zz + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        sd.prev = dd
        slope = -float((g * dd).sum())
        step = 1.0
        for _ in range(30):  # Armijo backtracking
            y_new = y - step * dd
            f_new = objective(y_new)
            if f_new <= f + 1e-4 * step * slope:
                break
            step *= 0.5
        y, f = y_new - y_new.mean(0), f_new
        if (early_iter + it + 1) in record or it + 1 == n_iter:
            if y.is_cuda:
                torch.cuda.synchronize()
            history[early_iter + it + 1] = (f, time.perf_counter() - start)
    return y, history


class Chain:
    """Apply preconditioners one after another (e.g. spectral direction, then Fourier)."""

    def __init__(self, *parts):
        self.parts = parts

    def __call__(self, y, g):
        for p in self.parts:
            g = p(y, g)
        return g


# ------------------------------------------------------------------ optimizer
def tsne(x, *, n_iter=750, early_iter=250, exaggeration=12.0, perplexity=30.0, lr=None, precond=None,
         precond_early=False, seed=0, record=(), affin=None, device="cuda", verbose=False, repulsion="exact",
         grid_h=0.2, max_grid=1024, lr_late=None, momentum_late=0.8, precond_ee=None, init_jitter=0.0,
         precond_until=None, precond_fade=0, polish_lr=None, polish_momentum=None, safeguard=False,
         safeguard_tol=1e-6, max_gain=None, init=None, callback=None, kl_exact=False):
    """Standard t-SNE optimization (FIt-SNE schedule: lr n/12, momentum 0.5/0.8; openTSNE instead
    uses lr n after early exaggeration, pass lr_late=n) with an optional gradient preconditioner.
    init_jitter > 0 adds seeded Gaussian noise (relative to the init scale) so seeds differ.
    precond_until=T: the preconditioner's cap fades linearly to 1 over `precond_fade` iterations ending at T
    and is off afterwards ("accelerate, then polish"); polish_lr / polish_momentum then replace the
    post-EE learning rate / momentum.
    safeguard=True (after EE, needs a Safeguarded preconditioner or none): the KL up to a constant,
    J = sum_E p log(1 + d^2) + log Z, is tracked every iteration (Z comes with the repulsion); if J rises
    the momentum buffer is zeroed (adaptive restart) and the boost is halved; it regrows slowly.
    safeguard_tol: relative rise of J that triggers it. max_gain: optional cap on the per-coordinate gains.
    Returns (y, history, t_aff) with history[iteration] = (KL, seconds)."""
    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    n = x.shape[0]
    t_aff = time.perf_counter()
    i, j, v = affinities(x, perplexity) if affin is None else affin
    t_aff = time.perf_counter() - t_aff
    # PCA initialisation, first component std 1e-4 (Kobak and Linderman)
    if init is not None:  # start from a given embedding (e.g. a perturbed converged one), fresh optimizer state
        y = torch.as_tensor(init, dtype=torch.float32, device=device).clone()
    else:
        xc = x - x.mean(0)
        _, _, vt = torch.linalg.svd(xc[: min(n, 20000)], full_matrices=False)
        y = xc @ vt[:2].T
        y = y / y[:, 0].std() * 1e-4
    if init is None and init_jitter > 0:  # seeds must change the run (adversary B1: the PCA init alone is deterministic)
        g = torch.Generator(device=y.device).manual_seed(int(seed))
        y = y + init_jitter * 1e-4 * torch.randn(y.shape, generator=g, device=y.device)
    lr = lr or max(n / exaggeration, 50.0)
    upd = torch.zeros_like(y)
    gains = torch.ones_like(y)
    history = {}
    J_prev, restarts = None, 0
    start = time.perf_counter()
    total = early_iter + n_iter
    for it in range(total):
        early = it < early_iter
        alpha = exaggeration if early else 1.0
        mom = 0.5 if early else momentum_late
        step_lr = lr if (early or lr_late is None) else lr_late
        polishing = precond_until is not None and it >= precond_until
        if precond is not None and precond_until is not None:
            if not hasattr(precond, "_cap0"):
                precond._cap0 = precond.cap
            start_fade = precond_until - precond_fade
            if it >= precond_until:
                precond.cap = 1.0
            elif precond_fade > 0 and it >= start_fade:
                precond.cap = 1.0 + (precond._cap0 - 1.0) * (precond_until - it) / precond_fade
            else:
                precond.cap = precond._cap0
            if hasattr(precond, "_mult"):
                precond._mult = None
        if polishing:
            mom = polish_momentum if polish_momentum is not None else mom
            step_lr = polish_lr if polish_lr is not None else step_lr
        fa, _ = attraction(y, i, j, v)
        fr, z = repulsion_exact(y) if repulsion == "exact" else repulsion_grid(y, grid_h, max_grid)
        grad = 4.0 * (alpha * fa - fr / z)
        if safeguard and not early:
            J = float((v * torch.log1p(((y[i] - y[j]) ** 2).sum(1))).sum()) + math.log(float(z))
            if J_prev is not None and J > J_prev + safeguard_tol * abs(J_prev):
                upd = torch.zeros_like(upd)  # adaptive restart
                if isinstance(precond, Safeguarded):
                    precond.cut()
                restarts += 1
            elif isinstance(precond, Safeguarded):
                precond.grow()
            J_prev = J
        if early and precond_ee is not None:
            grad = precond_ee(y, grad)  # a separate (milder) preconditioner for the early-exaggeration phase
        elif precond is not None and (precond_early or not early) and not polishing:
            grad = precond(y, grad)
        same = torch.sign(grad) == torch.sign(upd)
        gains = torch.where(same, gains * 0.8, gains + 0.2).clamp_min(0.01)
        if max_gain is not None:
            gains = gains.clamp_max(max_gain)  # unbounded gains are a runaway path for boosted cluster modes
        upd = mom * upd - step_lr * gains * grad
        y = y + upd
        y = y - y.mean(0)
        if callback is not None:
            callback(it + 1, y)
        if (it + 1) in record or (it + 1) == total:
            torch.cuda.synchronize() if y.is_cuda else None
            elapsed = time.perf_counter() - start
            if kl_exact:
                z_now = torch.tensor(exact_z(y), device=y.device, dtype=torch.float64)
            else:
                _, z_now = repulsion_exact(y) if repulsion == "exact" else repulsion_grid(y, grid_h, max_grid)
            history[it + 1] = (kl_divergence(y, i, j, v, z_now), elapsed)
            start += time.perf_counter() - start - elapsed  # do not count the KL evaluation
            if verbose:
                print(f"  it {it + 1}: KL {history[it + 1][0]:.4f}  {elapsed:.2f}s", flush=True)
    return y, history, t_aff
