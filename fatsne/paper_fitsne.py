"""Paper, block D2: the original FIt-SNE C++ implementation (KlugerLab, Linderman et al. 2019), built from
source on the pod, run through its own Python wrapper with its defaults (perplexity 30, Annoy kNN, early
exaggeration 12 for 250 iterations, momentum 0.5 -> 0.8, learning rate auto), 750 and 3000 iterations,
all container CPU threads. Scored under OUR affinities with the same KL (grid normaliser) and sampled 10-NN
as every other arm; wall-clock end-to-end.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import affinities, kl_divergence, repulsion_grid  # noqa: E402
from paperlib import load_any, sampled_knn_preservation  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
FITSNE = Path("/tmp/FIt-SNE")


def cpus():
    n = len(os.sched_getaffinity(0))
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            n = min(n, max(1, int(int(q) / int(per))))
    except Exception:  # noqa: BLE001
        pass
    return n


def main():
    sys.path.insert(0, str(FITSNE))
    from fast_tsne import fast_tsne
    res_path = OUT / "paper_fitsne.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    nt = cpus()
    for label, name, n in (("MNIST-70k", "MNIST", None), ("SIFT1M", "SIFT", 1_000_000), ("T2I-1M", "T2I-1M", None)):
        xn = load_any(name, n)
        x = torch.as_tensor(xn, device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        rows = res.get(label, {})
        xc = np.ascontiguousarray((xn - xn.mean(0)).astype(np.float64))
        for its in (750, 3000):
            key = f"FIt-SNE {its} it ({nt} cpu)"
            if key in rows:
                continue
            cwd = os.getcwd()
            os.chdir(FITSNE)  # the wrapper calls bin/fast_tsne relative to its folder
            t = time.perf_counter()
            y = fast_tsne(xc, perplexity=30, max_iter=its, nthreads=nt, seed=0)
            sec = time.perf_counter() - t
            os.chdir(cwd)
            yt = torch.as_tensor(np.asarray(y), dtype=torch.float32, device="cuda")
            _, z = repulsion_grid(yt, 0.25, 1536)
            kl = kl_divergence(yt, *aff, z)
            nn = sampled_knn_preservation(x, yt)
            rows[key] = {"kl": kl, "knn10": nn, "seconds_total": sec}
            res[label] = rows
            res_path.write_text(json.dumps(res, indent=1))
            print(f"  {label:10s} {key:26s} KL {kl:.4f}  10-NN {nn:.4f}  {sec:.1f}s end-to-end", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
