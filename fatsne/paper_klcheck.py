"""Paper check: bias of the grid-normaliser KL against the exact all-pairs normaliser at 10^6 points, for the
saved seed-0 embeddings (std, tuned, fourier) of SIFT1M and DEEP1M. Exact Z = sum_{i != j} 1/(1+|y_i-y_j|^2),
computed in float64 chunks on the GPU.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import affinities, kl_divergence, repulsion_grid  # noqa: E402
from paperlib import load_any  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def exact_z(y, chunk=1024):
    y = y.double()
    yy = (y * y).sum(1)
    z = torch.zeros((), dtype=torch.float64, device=y.device)
    for s in range(0, y.shape[0], chunk):
        d2 = (yy[s:s + chunk, None] - 2 * y[s:s + chunk] @ y.T + yy[None, :]).clamp_min(0)
        z += (1.0 / (1.0 + d2)).sum() - min(chunk, y.shape[0] - s)  # remove the diagonal terms (d2 = 0)
    return float(z)


def main():
    sets = [a for a in sys.argv[1:]] or ["SIFT1M", "DEEP1M"]
    path = OUT / f"paper_klcheck_{'_'.join(sets)}.json"
    res = {}
    names = {"SIFT1M": ("SIFT", 1_000_000), "DEEP1M": ("DEEP", 1_000_000), "GloVe-100": ("GloVe-100", None),
             "T2I-1M": ("T2I-1M", None), "dbpedia-1M": ("dbpedia", None)}
    for label in sets:
        name, nn = names[label]
        x = torch.as_tensor(load_any(name, nn), device="cuda")
        x = x - x.mean(0)
        i, j, v = affinities(x, 30.0)
        for arm, seed in [(a, s) for s in (0, 2) for a in ("std", "tuned", "fourier")]:
            f = OUT / "embeddings" / f"{label}_{arm}_seed{seed}.npy"
            if not f.exists():
                continue
            y = torch.as_tensor(np.load(f).astype(np.float32), device="cuda")
            _, zg = repulsion_grid(y, 0.25, 1536)
            ze = exact_z(y)
            kg = kl_divergence(y, i, j, v, zg)
            ke = kl_divergence(y, i, j, v, torch.tensor(ze, device="cuda"))
            res[f"{label} {arm} seed{seed}"] = {"kl_grid": kg, "kl_exact": ke, "z_grid": float(zg), "z_exact": ze}
            print(f"  {label} seed {seed} {arm:8s} KL grid {kg:.5f} exact {ke:.5f} (difference {kg - ke:+.5f})", flush=True)
        path.write_text(json.dumps(res, indent=1))
        del x


if __name__ == "__main__":
    main()
