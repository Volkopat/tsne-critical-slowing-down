"""Dev check (not a sealed round): does the Nesterov scheme of Lambert et al. inflate the map at small n with EXACT
repulsion, or only with our grid repulsion? SIFT prefixes 1e4 and 3e4, 6000 iterations, seed 0, exact KL and RMS extent
every 1000 iterations. Laptop GPU, memory capped at 0.25 (shared-laptop protocol)."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from competitors import lambert
from fatsne import affinities, exact_z, kl_divergence

torch.cuda.set_per_process_memory_fraction(0.25)
SRC = Path(__file__).resolve().parents[2] / "e-means" / "data" / "sift-128-euclidean.hdf5"
OUT = Path(__file__).resolve().parent / "results" / "lambert_exact_dev.json"
res = {}
for n in (10_000, 30_000):
    with h5py.File(SRC, "r") as f:
        x = torch.as_tensor(np.asarray(f["train"][:n], dtype=np.float32), device="cuda")
    x = x - x.mean(0)
    aff = affinities(x, 30.0)
    i, j, v = aff
    for rep in ("grid", "exact"):
        track = {}

        def cb(it, y):
            if it % 1000 == 0:
                z = torch.tensor(exact_z(y, chunk=2048), device=y.device, dtype=torch.float64)
                track[it] = (round(kl_divergence(y, i, j, v, z), 4), round(float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt()), 1))

        try:
            lambert(x, aff, n_iter=6000, seed=0, exact=(rep == "exact"), callback=cb)
        except ValueError as e:  # the grid size becomes NaN once the map has blown up
            track["failed"] = f"after iteration {max([k for k in track if isinstance(k, int)], default=0)}: {e}"
        res[f"{n}_{rep}"] = track
        print(f"n={n:>6d} {rep:5s} (exact KL, extent) by iteration: {track}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    del x, aff
    torch.cuda.empty_cache()
