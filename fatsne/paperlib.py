"""Shared pieces of the paper experiments ("Critical slowing down in t-SNE and its cure by Fourier
acceleration"): data loading, the three optimizer arms, metrics and the frozen grid-size rule."""

import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner, tsne  # noqa: E402


def sampled_knn_preservation(x, y, k=10, m=10000, seed=0):
    """10-NN preservation: for m random points, the fraction of their k nearest neighbours in the input space x
    (searched among all n points) that are also among their k nearest neighbours in the map y (also among all n)."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    q = torch.randperm(x.shape[0], generator=g, device=x.device)[:m]

    def top(z):
        zn = (z * z).sum(1)
        out = []
        for b in range(0, m, 1024):
            s = (z[q[b:b + 1024]] * z[q[b:b + 1024]]).sum(1)[:, None] - 2 * z[q[b:b + 1024]] @ z.T + zn[None, :]
            s[torch.arange(s.shape[0], device=z.device), q[b:b + 1024]] = float("inf")
            out.append(torch.topk(s, k, dim=1, largest=False).indices)
        return torch.cat(out)

    a, b = top(x), top(y)
    return float((b[:, :, None] == a[:, None, :]).any(-1).float().mean())

DATA = Path(__file__).resolve().parents[2] / "data"
NEW = Path("/tmp/emeans_data/new")
ANN = {"SIFT": "sift-128-euclidean", "DEEP": "deep-image-96-angular", "GloVe-100": "glove-100-angular",
       "MNIST": "mnist-784-euclidean", "Fashion": "fashion-mnist-784-euclidean"}


def load_any(name, n=None, perm_seed=0):
    """name in ANN (first n rows), 'T2I' (nested random subsets of the first 3M rows, fixed permutation),
    'T2I-1M' (first 1M rows, as in held-out 2), 'dbpedia' (1M OpenAI embeddings)."""
    if name in ANN:
        for p in (DATA / f"{ANN[name]}.hdf5", DATA / "fresh" / f"{ANN[name]}.hdf5"):
            if p.exists():
                with h5py.File(p, "r") as f:
                    return np.asarray(f["train"][:n] if n else f["train"][:], dtype=np.float32)
        raise FileNotFoundError(name)
    if name == "T2I-1M":
        return np.load(NEW / "t2i-1M.npy")[:n]
    if name == "dbpedia":
        return np.load(NEW / "dbpedia-openai-1M.npy")[:n]
    if name == "T2I":
        raw = np.memmap(NEW / "t2i-3M.fbin", dtype=np.float32, mode="r", offset=8)
        full = raw.reshape(-1, 200)
        perm = np.random.default_rng(perm_seed).permutation(full.shape[0])
        return np.ascontiguousarray(full[np.sort(perm[:n])])
    raise ValueError(name)


def grid_rule(n):
    """Frozen grid size: 2^round(log2(8 n^(1/4))), clipped to [64, 512] (128 at 30k-100k, 256 at 300k-3M)."""
    return int(min(512, max(64, 2 ** round(math.log2(8 * n ** 0.25)))))


def arm_kwargs(arm, n):
    if arm == "std":  # FIt-SNE schedule
        return {}
    if arm == "tuned":  # openTSNE post-EE learning rate + momentum 0.9 (strongest standard setting found)
        return dict(lr_late=float(n), momentum_late=0.9)
    if arm == "fourier":
        return dict(precond=FourierPreconditioner(grid_rule(n), 16.0))
    raise ValueError(arm)


def run(x, aff, arm, seed, n_iter_total, record, extra=None):
    kw = arm_kwargs(arm, x.shape[0])
    kw.update(extra or {})
    y, hist, _ = tsne(x, affin=aff, seed=seed, init_jitter=0.1, n_iter=n_iter_total - 250, record=record,
                      repulsion="grid", grid_h=0.25, max_grid=1536, **kw)
    return y, {"kl": {str(k): w[0] for k, w in hist.items()}, "sec": {str(k): w[1] for k, w in hist.items()}}
