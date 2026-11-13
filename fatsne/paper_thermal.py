"""Mermin-Wagner test for t-SNE maps (predictions P1-P3 sealed in novel/r16b_unaided.md before any run).
For each T2I subset size: converge a tuned t-SNE map (3000 iterations), then sample pi(y) ~ exp(-c n KL) around it with
Fourier-accelerated Langevin (thermal.py) and record
  P1 median variance of absolute positions (after removing translation and rotation) and of displacements relative to
     the 10 nearest map neighbours, against n;
  P2 the radially averaged displacement spectrum and its log-log slope over k = 1..6 grid units;
  P3 at n = 1e5 only: integrated autocorrelation time of the m = 1 shear amplitude, Fourier vs plain Langevin.
Usage: python paper_thermal.py [smoke]. One GPU.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner, affinities, tsne  # noqa: E402
from paperlib import load_any  # noqa: E402
from thermal import autocorr_time, langevin  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def grid_v2(n):
    return int(min(512, max(128, 2 ** round(math.log2(8 * n ** 0.25)))))


def map_neighbours(y, idx, k=10):
    d = torch.cdist(y[idx], y)
    d[torch.arange(idx.shape[0]), idx] = float("inf")
    return d.topk(k, largest=False).indices


def slope(spec, k0=1, k1=6):
    ks = np.arange(k0, k1 + 1)
    return float(np.polyfit(np.log(ks), np.log(spec[k0:k1 + 1].numpy()), 1)[0])


def one_size(x, n, steps, burn, plain=False, seed=0):
    aff = affinities(x, 30.0)
    y, _, _ = tsne(x, affin=aff, seed=seed, init_jitter=0.1, n_iter=2750, repulsion="grid", grid_h=0.25, max_grid=1536,
                   lr_late=float(n), momentum_late=0.9)
    g = torch.Generator(device=y.device).manual_seed(seed)
    track = torch.randperm(n, generator=g, device=y.device)[:min(n, 5000)]
    nbr = map_neighbours(y, track)
    s = float((y[nbr[:, -1]] - y[track]).norm(dim=1).median())  # r22 scale: median distance to the 10th map neighbour
    out = {"s": s}
    temps = [float(t) for t in __import__("os").environ.get("THERMAL_C", "10,100").split(",")]
    arms = [(f"fourier_c{c:g}", FourierPreconditioner(grid_v2(n), 16.0), c) for c in temps]
    arms += [("plain_c100", None, 100.0)] if plain else []
    for name, pc, c in arms:  # c = observations per point (inverse temperature); too hot a posterior melts the map
        r = langevin(y, aff, c=c, n_steps=steps, burn=burn, pc=pc, every=10, track=track, nbr=nbr, seed=seed)
        out[name] = {"var_abs_median": float(r["var_abs"].median()), "var_rel_median": float(r["var_rel"].median()),
                     "spectrum": r["spectrum"].tolist(), "slope_k1_6": slope(r["spectrum"]),
                     "tau_m1_snapshots": autocorr_time(r["amp_m1"]), "msd": r["msd"][::10], "h": r["h"]}
        print(f"  n={n:>7d} {name:8s} var_abs {out[name]['var_abs_median']:.4g} var_rel {out[name]['var_rel_median']:.4g} "
              f"slope {out[name]['slope_k1_6']:+.2f} tau_m1 {out[name]['tau_m1_snapshots']:.1f} (x10 steps)", flush=True)
    return out


DATASET = __import__("os").environ.get("THERMAL_DATA", "T2I")


def main():
    if torch.cuda.is_available() and DATASET in ("SIFT-local", "DEEP-local"):  # shared laptop GPU: stay under ~5 GB
        torch.cuda.set_per_process_memory_fraction(0.2)
        import fatsne as F  # smaller k-NN blocks so n = 3e5 fits the cap (the first attempt ran out of memory there)
        _knn = F.knn
        F.knn = lambda x, k, chunk=None: _knn(x, k, chunk=256)
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    sizes = (2000,) if smoke else ((10_000, 30_000, 100_000) if DATASET == "DEEP-local" else (10_000, 30_000, 100_000, 300_000))
    steps, burn = (60, 20) if smoke else (4000, 1000)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    path = OUT / ("paper_thermal_smoke.json" if smoke else f"paper_thermal_{DATASET}.json")
    res = json.loads(path.read_text()) if path.exists() and not smoke else {}
    for n in sizes:
        if str(n) in res:
            continue
        if smoke:
            x = torch.randn(n, 16, device=dev) + 4.0 * torch.randn(8, 16, device=dev)[torch.arange(n, device=dev) % 8]
        elif DATASET == "DEEP-local":  # r22: confirmatory run on fresh data (DEEP prefixes)
            import h5py
            f = Path(__file__).resolve().parents[2] / "e-means" / "data" / "deep-image-96-angular.hdf5"
            with h5py.File(f, "r") as h:
                x = torch.as_tensor(h["train"][:n].astype(np.float32), device=dev)
        elif DATASET == "SIFT-local":  # amendment r16d: laptop run on nested SIFT1M prefixes
            import h5py
            f = Path(__file__).resolve().parents[2] / "e-means" / "data" / "sift-128-euclidean.hdf5"
            with h5py.File(f, "r") as h:
                x = torch.as_tensor(h["train"][:n].astype(np.float32), device=dev)
        else:
            x = torch.as_tensor(load_any("T2I", n), device=dev)
        x = x - x.mean(0)
        res[str(n)] = one_size(x, n, steps, burn, plain=((n == 100_000 and DATASET != "DEEP-local") or smoke))
        path.write_text(json.dumps(res))
    ns = sorted(int(k) for k in res)
    if len(ns) >= 3:
        for arm in ("fourier_c10", "fourier_c100"):
            va = [res[str(n)][arm]["var_abs_median"] for n in ns]
            vr = [res[str(n)][arm]["var_rel_median"] for n in ns]
            print(f"P1 {arm}: slope in log n of var_abs {np.polyfit(np.log(ns), va, 1)[0]:.4g}, of var_rel "
                  f"{np.polyfit(np.log(ns), vr, 1)[0]:.4g}")


if __name__ == "__main__":
    main()
