"""Paper additions: head-to-head with optimiser-level competitors and labelled / single-cell
benchmarks. All arms share affinities (perplexity 30) and the grid repulsion; only the optimiser differs.
Modes:
  python paper_competitors.py main        5 datasets at ~1M x {lambert (grid 1536), lambert (grid 4096), ars} x 3 seeds
  python paper_competitors.py labeled     MNIST-70k, Fashion-70k (OpenML labels) x {std, tuned, fourier_v2, lambert, ars}
  python paper_competitors.py singlecell  10x 1.3M mouse brain (PCA-50, 10x graph-based cluster labels), same arms, 2 seeds
Metrics: exact KL after 3000 iterations, 10-NN preservation, map extent, seconds; with labels also the leave-one-out
10-NN classification accuracy in the map and trustworthiness (10 neighbours), both on 10,000 random points.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from competitors import ars, lambert  # noqa: E402
from fatsne import FourierPreconditioner, affinities, exact_z, kl_divergence, tsne  # noqa: E402
from paperlib import load_any, sampled_knn_preservation  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
DATA = Path("/tmp/emeans_data/new")


def grid_v1(n):
    return int(min(512, max(64, 2 ** round(math.log2(8 * n ** 0.25)))))


def grid_v2(n):
    return int(min(512, max(128, 2 ** round(math.log2(8 * n ** 0.25)))))


def knn_accuracy(y, labels, m=10000, k=10, seed=0):
    g = torch.Generator(device=y.device).manual_seed(seed)
    idx = torch.randperm(y.shape[0], generator=g, device=y.device)[:m]
    lab = torch.as_tensor(labels, device=y.device)
    correct = 0
    for s in range(0, m, 1024):
        q = idx[s:s + 1024]
        d = torch.cdist(y[q], y)
        d[torch.arange(len(q)), q] = float("inf")
        nb = d.topk(k, largest=False).indices
        votes = torch.zeros(len(q), int(lab.max()) + 1, device=y.device).scatter_add_(
            1, lab[nb], torch.ones_like(nb, dtype=torch.float32))
        correct += int((votes.argmax(1) == lab[q]).sum())
    return correct / m


def trustworthiness(x, y, m=10000, k=10, seed=0):
    """Venna-Kaski trustworthiness on a random subset (ranks computed within the subset)."""
    g = torch.Generator(device=y.device).manual_seed(seed)
    idx = torch.randperm(y.shape[0], generator=g, device=y.device)[:m]
    xs, ys = x[idx], y[idx]
    dx = torch.cdist(xs, xs)
    dx.fill_diagonal_(float("inf"))
    rank_x = dx.argsort(1).argsort(1)  # 0-based rank of j among i's neighbours in the data
    dy = torch.cdist(ys, ys)
    dy.fill_diagonal_(float("inf"))
    nb_y = dy.topk(k, largest=False).indices
    r = rank_x.gather(1, nb_y).float() + 1.0
    pen = (r - k).clamp_min(0).sum()
    return float(1.0 - 2.0 / (m * k * (2 * m - 3 * k - 1)) * pen)


def run_arm(arm, x, aff, seed, n_iter=3000):
    n = x.shape[0]
    common = dict(seed=seed, record=())
    t = time.perf_counter()
    if arm == "std":
        y, _, _ = tsne(x, affin=aff, init_jitter=0.1, n_iter=n_iter - 250, repulsion="grid", grid_h=0.25,
                       max_grid=1536, **common)
    elif arm == "tuned":
        y, _, _ = tsne(x, affin=aff, init_jitter=0.1, n_iter=n_iter - 250, repulsion="grid", grid_h=0.25,
                       max_grid=1536, lr_late=float(n), momentum_late=0.9, **common)
    elif arm == "fourier_v1":  # the engine of the main table (no gains cap); added 2026-11-12 after the main table
        y, _, _ = tsne(x, affin=aff, init_jitter=0.1, n_iter=n_iter - 250, repulsion="grid", grid_h=0.25,
                       max_grid=1536, precond=FourierPreconditioner(grid_v1(n), 16.0), **common)
    elif arm == "fourier_v2":
        y, _, _ = tsne(x, affin=aff, init_jitter=0.1, n_iter=n_iter - 250, repulsion="grid", grid_h=0.25,
                       max_grid=1536, precond=FourierPreconditioner(grid_v2(n), 16.0), max_gain=10.0, **common)
    elif arm == "lambert":
        y, _ = lambert(x, aff, n_iter=n_iter, max_grid=1536, **common)
    elif arm == "lambert_fourier":  # r30: Lambert's scheme with the Fourier preconditioner on the gradient
        y, _ = lambert(x, aff, n_iter=n_iter, max_grid=1536, precond=FourierPreconditioner(grid_v1(n), 16.0), **common)
    elif arm == "lambert_g4096":
        y, _ = lambert(x, aff, n_iter=n_iter, max_grid=4096, **common)
    elif arm == "ars":
        y, _ = ars(x, aff, n_iter=n_iter, **common)
    else:
        raise ValueError(arm)
    torch.cuda.synchronize()
    return y, time.perf_counter() - t


def evaluate(name, x, labels, arms, seeds, path):
    res = json.loads(path.read_text()) if path.exists() else {}
    x = x - x.mean(0)
    aff = affinities(x, 30.0)
    i, j, v = aff
    rows = res.get(name, {})
    for arm in arms:
        runs = rows.get(arm, [])
        for seed in range(len(runs), seeds):
            y, sec = run_arm(arm, x, aff, seed)
            ok = bool(torch.isfinite(y).all())
            r = {"sec": sec, "finite": ok, "extent": float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt()) if ok else None}
            if ok:
                r["kl_exact"] = kl_divergence(y, i, j, v, torch.tensor(exact_z(y), device=y.device, dtype=torch.float64))
                r["knn10"] = sampled_knn_preservation(x, y)
                if labels is not None:
                    r["knn_acc"] = knn_accuracy(y, labels)
                    r["trust"] = trustworthiness(x, y)
            runs.append(r)
            rows[arm] = runs
            res[name] = rows
            path.write_text(json.dumps(res, indent=1))
            print(f"{name:11s} {arm:13s} seed {seed}: " + " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                                for k, v in r.items()), flush=True)
    del aff
    torch.cuda.empty_cache()


def arms_from_env(default):
    """COMP_ARMS=a,b,c overrides the arm list (finished runs are kept and skipped either way)."""
    env = os.environ.get("COMP_ARMS")
    return tuple(env.split(",")) if env else default


def main_mode():
    path = OUT / "paper_competitors_main.json"
    for label, name, n in (("SIFT1M", "SIFT", 1_000_000), ("DEEP1M", "DEEP", 1_000_000), ("GloVe-100", "GloVe-100", None),
                           ("T2I-1M", "T2I-1M", None), ("dbpedia-1M", "dbpedia", None)):
        x = torch.as_tensor(load_any(name, n), device="cuda")
        evaluate(label, x, None, arms_from_env(("lambert", "lambert_g4096", "ars")), 3, path)


def labeled_mode():
    path = OUT / "paper_competitors_labeled.json"
    for label in ("MNIST-70k", "Fashion-70k"):
        d = np.load(DATA / f"{label}.npz")
        x = torch.as_tensor(d["x"].astype(np.float32), device="cuda")
        evaluate(label, x, d["y"].astype(np.int64),
                 arms_from_env(("std", "tuned", "fourier_v2", "lambert", "lambert_g4096", "ars", "fourier_v1",
                                "lambert_fourier")), 3, path)


def singlecell_mode():
    path = OUT / "paper_competitors_singlecell.json"
    d = np.load(DATA / "10x-1.3M-pca50.npz")
    x = torch.as_tensor(d["x"].astype(np.float32), device="cuda")
    evaluate("10x-1.3M", x, d["y"].astype(np.int64),
             arms_from_env(("std", "tuned", "fourier_v1", "fourier_v2", "lambert", "lambert_g4096", "ars", "lambert_fourier")),
             2, path)


if __name__ == "__main__":
    {"main": main_mode, "labeled": labeled_mode, "singlecell": singlecell_mode}[sys.argv[1]]()
