"""Paper, corrected suite (round 14; sealed predictions in novel/r14_unaided.md).
Fixes found in round 12-13: (1) KL is evaluated with the EXACT normaliser (the grid normaliser underestimates
Z by 1-3% for the larger Fourier maps); (2) engine v2 caps the delta-bar-delta gains at 10 and never uses a
grid below 128 (v1 diverged at 10k points with grid 64 and inflated in long runs at 30k-100k).
Snapshots of the map are kept at a few iterations during the run and their exact KL is computed after the
run, so the timing is not affected. Modes:
  python paper_v2.py main      5 datasets x {std, tuned, fourier_v1, fourier_v2} x 3 seeds x 3000 iterations
  python paper_v2.py scaling   T2I subsets 10k..3M x {std, tuned, fourier_v2} x 3 seeds x 6000 iterations
One GPU alone.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner, affinities, exact_z, kl_divergence, tsne  # noqa: E402
from paperlib import load_any, sampled_knn_preservation  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MAIN_SEEDS = int(__import__("os").environ.get("MAIN_SEEDS", "3"))  # 5 seeds for the headline table
EMB = OUT / "embeddings_v2"
EMB.mkdir(parents=True, exist_ok=True)


def grid_v1(n):
    return int(min(512, max(64, 2 ** round(math.log2(8 * n ** 0.25)))))


def grid_v2(n):
    return max(128, grid_v1(n))


def make(arm, n):
    if arm == "std":
        return {}
    if arm == "tuned":
        return dict(lr_late=float(n), momentum_late=0.9)
    if arm == "fourier_v1":
        return dict(precond=FourierPreconditioner(grid_v1(n), 16.0))
    if arm == "fourier_v2":
        return dict(precond=FourierPreconditioner(grid_v2(n), 16.0), max_gain=10.0)
    raise ValueError(arm)


def run(x, aff, arm, seed, total, record, snaps):
    i, j, v = aff
    keep = {}

    def cb(it, y):
        if it in snaps:
            keep[it] = y.detach().clone()

    y, hist, _ = tsne(x, affin=aff, seed=seed, init_jitter=0.1, n_iter=total - 250, record=record, repulsion="grid",
                      grid_h=0.25, max_grid=1536, callback=cb, **make(arm, x.shape[0]))
    r = {"kl_grid": {str(k): w[0] for k, w in hist.items()}, "sec": {str(k): w[1] for k, w in hist.items()}}
    r["kl_exact"] = {str(k): kl_divergence(ys, i, j, v, torch.tensor(exact_z(ys), device=ys.device, dtype=torch.float64))
                     for k, ys in sorted(keep.items())}
    r["extent"] = float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt())
    r["knn10"] = sampled_knn_preservation(x, y)
    return y, r


def main_suite():
    res_path = OUT / "paper_v2_main.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    record = tuple(range(300, 3001, 100))
    snaps = {500, 1000, 1500, 2000, 2500, 3000}
    for label, name, n in (("SIFT1M", "SIFT", 1_000_000), ("DEEP1M", "DEEP", 1_000_000), ("GloVe-100", "GloVe-100", None),
                           ("T2I-1M", "T2I-1M", None), ("dbpedia-1M", "dbpedia", None)):
        rows = res.get(label, {})
        arms = ("std", "tuned", "fourier_v1", "fourier_v2")
        if all(len(rows.get(a, [])) == MAIN_SEEDS for a in arms):
            continue
        x = torch.as_tensor(load_any(name, n), device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        tsne(x[:20000], n_iter=20, early_iter=10, seed=99, repulsion="grid")
        for arm in arms:
            runs = rows.get(arm, [])
            for seed in range(len(runs), MAIN_SEEDS):
                y, r = run(x, aff, arm, seed, 3000, record, snaps)
                runs.append(r)
                rows[arm] = runs
                res[label] = rows
                res_path.write_text(json.dumps(res, indent=1))
                np.save(EMB / f"{label}_{arm}_seed{seed}.npy", y.cpu().numpy().astype(np.float16))
                print(f"  {label:11s} {arm:10s} seed {seed}: exact KL@1000 {r['kl_exact']['1000']:.4f} @3000 {r['kl_exact']['3000']:.4f} "
                      f"(grid {r['kl_grid']['3000']:.4f}) 10-NN {r['knn10']:.4f} extent {r['extent']:.0f} {r['sec']['3000']:.1f}s", flush=True)
        del x, aff
        torch.cuda.empty_cache()


def scaling_suite():
    res_path = OUT / "paper_v2_scaling.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    record = tuple(sorted(set(range(300, 1001, 50)) | set(range(1100, 6001, 100))))
    snaps = {500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000}
    for n in (10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000):
        key = str(n)
        rows = res.get(key, {})
        arms = ("std", "tuned", "fourier_v2")
        if all(len(rows.get(a, [])) == 3 for a in arms):
            continue
        x = torch.as_tensor(load_any("T2I", n), device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        tsne(x[:min(n, 20000)], n_iter=20, early_iter=10, seed=99, repulsion="grid")
        for arm in arms:
            runs = rows.get(arm, [])
            for seed in range(len(runs), 3):
                _, r = run(x, aff, arm, seed, 6000, record, snaps)
                runs.append(r)
                rows[arm] = runs
                res[key] = rows
                res_path.write_text(json.dumps(res, indent=1))
                print(f"  n={n:>8d} {arm:10s} seed {seed}: exact KL " + " ".join(f"{k}:{v:.4f}" for k, v in r["kl_exact"].items())
                      + f" extent {r['extent']:.0f} {r['sec']['6000']:.0f}s", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__" and sys.argv[1] in ("main", "scaling"):
    {"main": main_suite, "scaling": scaling_suite}[sys.argv[1]]()


class FlatLowPass(FourierPreconditioner):
    """Same cap, no 1/k^2 shape: multiplier 16 for |k| < 0.18 k_max, 1 elsewhere."""

    def _multiplier(self, y, kk, span):
        kmax2 = float(kk.max())
        m = torch.where(kk < (0.18 ** 2) * kmax2, torch.full_like(kk, self.cap), torch.ones_like(kk))
        m[0, 0] = 1.0
        return m


def ablate_suite():
    """Ablations of engine v2 (sealed in r15_unaided.md): SIFT1M and GloVe-100, 2 seeds, 3000 iterations, exact KL."""
    res_path = OUT / "paper_v2_ablate.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    arms = {"cap4": (dict(max_gain=10.0), lambda n: FourierPreconditioner(grid_v2(n), 4.0), 250),
            "cap64": (dict(max_gain=10.0), lambda n: FourierPreconditioner(grid_v2(n), 64.0), 250),
            "grid128": (dict(max_gain=10.0), lambda n: FourierPreconditioner(128, 16.0), 250),
            "grid512": (dict(max_gain=10.0), lambda n: FourierPreconditioner(512, 16.0), 250),
            "flat16": (dict(max_gain=10.0), lambda n: FlatLowPass(grid_v2(n), 16.0), 250),
            "EE100": (dict(max_gain=10.0), lambda n: FourierPreconditioner(grid_v2(n), 16.0), 100),
            "v2+mom0.9": (dict(max_gain=10.0, momentum_late=0.9), lambda n: FourierPreconditioner(grid_v2(n), 16.0), 250)}
    record = tuple(range(300, 3001, 100))
    for label, name, n0 in (("SIFT1M", "SIFT", 1_000_000), ("GloVe-100", "GloVe-100", None)):
        x = torch.as_tensor(load_any(name, n0), device="cuda")
        x = x - x.mean(0)
        n = x.shape[0]
        aff = affinities(x, 30.0)
        i, j, v = aff
        tsne(x[:20000], n_iter=20, early_iter=10, seed=99, repulsion="grid")
        rows = res.get(label, {})
        for arm, (kw, pc, ee) in arms.items():
            runs = rows.get(arm, [])
            for seed in range(len(runs), 2):
                y, hist, _ = tsne(x, affin=aff, seed=seed, init_jitter=0.1, early_iter=ee, n_iter=3000 - ee, record=record,
                                  repulsion="grid", grid_h=0.25, max_grid=1536, precond=pc(n), **kw)
                ke = kl_divergence(y, i, j, v, torch.tensor(exact_z(y), device=y.device, dtype=torch.float64))
                runs.append({"kl_exact_3000": ke, "kl_grid": {str(k): w[0] for k, w in hist.items()},
                             "knn10": sampled_knn_preservation(x, y), "extent": float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt())})
                rows[arm] = runs
                res[label] = rows
                res_path.write_text(json.dumps(res, indent=1))
                print(f"  {label:10s} {arm:10s} seed {seed}: exact KL@3000 {ke:.4f} 10-NN {runs[-1]['knn10']:.4f} extent {runs[-1]['extent']:.0f}", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "ablate":
    ablate_suite()


def scaling_lambert_suite():
    """r31: Lambert et al.'s Nesterov scheme on the same T2I subsets as scaling_suite (n <= 1e6), 3 seeds, 6000
    iterations from the start, exact KL at the same snapshots. Separate results file."""
    import time
    from competitors import lambert
    import os
    res_path = Path(os.environ.get("SCAL_OUT", str(OUT / "paper_v2_scaling_lambert.json")))
    sizes = tuple(int(s) for s in os.environ.get("SCAL_SIZES", "10000,30000,100000,300000,1000000").split(","))
    total, nseeds = int(os.environ.get("SCAL_ITERS", "6000")), int(os.environ.get("SCAL_SEEDS", "3"))
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    record = tuple(sorted(set(range(300, 1001, 50)) | set(range(1100, 6001, 100))))
    snaps = {500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000}
    for n in sizes:
        key = str(n)
        runs = res.get(key, {}).get("lambert", [])
        if len(runs) == nseeds:
            continue
        x = torch.as_tensor(load_any("T2I", n), device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        i, j, v = aff
        for seed in range(len(runs), nseeds):
            keep = {}

            def cb(it, y):
                if it in snaps:
                    keep[it] = y.detach().clone()

            t = time.perf_counter()
            y, hist = lambert(x, aff, n_iter=total, seed=seed, record=tuple(k for k in record if k <= total), max_grid=1536,
                              callback=cb)
            torch.cuda.synchronize()
            r = {"kl_grid": {str(k): float(w) for k, w in hist.items()}, "sec_total": time.perf_counter() - t,
                 "kl_exact": {str(k): kl_divergence(ys, i, j, v, torch.tensor(exact_z(ys), device=ys.device, dtype=torch.float64))
                              for k, ys in sorted(keep.items())},
                 "extent": float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt()), "knn10": sampled_knn_preservation(x, y)}
            runs.append(r)
            res[key] = {"lambert": runs}
            res_path.write_text(json.dumps(res, indent=1))
            print(f"  n={n:>8d} lambert    seed {seed}: exact KL " + " ".join(f"{k}:{v:.4f}" for k, v in r["kl_exact"].items())
                  + f" extent {r['extent']:.0f} {r['sec_total']:.0f}s", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "scaling_lambert":
    scaling_lambert_suite()
