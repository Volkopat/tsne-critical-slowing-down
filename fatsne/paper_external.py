"""Paper, block D: established implementations, scored with the same metrics. openTSNE (CPU, FFT
gradients, its own kNN and affinities, n_jobs = all cores) with its defaults (750 iterations) and with
3000 iterations; NVIDIA cuML TSNE (GPU, FFT) with defaults and with FIt-SNE-style settings, 1000 and 3000
iterations. Every embedding is scored under OUR affinities (perplexity 30, 90 neighbours): KL with the
grid normaliser (as for our arms) and sampled 10-NN preservation; wall-clock is end-to-end (including each
package's own neighbour search). Datasets: MNIST-70k, SIFT1M, T2I-1M. Runs on its own pod (CPU work must
not overlap GPU timing).
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
EMB = OUT / "embeddings"
EMB.mkdir(parents=True, exist_ok=True)


def score(x, aff, y):
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32, device="cuda")
    i, j, v = aff
    _, z = repulsion_grid(y, 0.25, 1536)
    return kl_divergence(y, i, j, v, z), sampled_knn_preservation(x, y)


def main():
    res_path = OUT / "paper_external.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    ncpu = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:  # container CPU quota (cgroup v2), which os.cpu_count() does not see
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            ncpu = min(ncpu, max(1, int(int(q) / int(per))))
    except Exception:  # noqa: BLE001
        pass
    for label, name, n in (("MNIST-70k", "MNIST", None), ("SIFT1M", "SIFT", 1_000_000), ("T2I-1M", "T2I-1M", None)):
        xn = load_any(name, n)
        x = torch.as_tensor(xn, device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        rows = res.get(label, {})
        xc = np.ascontiguousarray(xn - xn.mean(0))

        def record(key, y, sec):
            kl, nn = score(x, aff, y)
            rows[key] = {"kl": kl, "knn10": nn, "seconds_total": sec}
            res[label] = rows
            res_path.write_text(json.dumps(res, indent=1))
            np.save(EMB / f"{label}_{key.replace(' ', '_')}.npy", np.asarray(y, dtype=np.float16))
            print(f"  {label:10s} {key:34s} KL {kl:.4f}  10-NN {nn:.4f}  {sec:.1f}s end-to-end", flush=True)

        from openTSNE import TSNE as OTSNE
        for its in (750, 3000):
            key = f"openTSNE {its} it ({ncpu} cpu)"
            if key in rows:
                continue
            t = time.perf_counter()
            y = OTSNE(perplexity=30, n_jobs=ncpu, negative_gradient_method="fft", initialization="pca",
                      early_exaggeration_iter=250, n_iter=its - 250, random_state=0, verbose=False).fit(xc)
            record(key, np.asarray(y), time.perf_counter() - t)
        try:
            from cuml.manifold import TSNE as CuTSNE
            import inspect
            kw_it = "max_iter" if "max_iter" in inspect.signature(CuTSNE.__init__).parameters else "n_iter"
            CuTSNE(n_components=2, method="fft", random_state=0, **{kw_it: 300}).fit_transform(xc[:20000])  # warm-up
            for its in (1000, 3000):
                for flavour, extra in (("defaults", {}), ("FIt-SNE settings", dict(perplexity=30.0, n_neighbors=90, early_exaggeration=12.0,
                                                                                  exaggeration_iter=250, learning_rate=max(len(xc) / 12.0, 200.0)))):
                    key = f"cuML {flavour} {its} it"
                    if key in rows:
                        continue
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    y = CuTSNE(n_components=2, method="fft", random_state=0, **extra, **{kw_it: its}).fit_transform(xc)
                    torch.cuda.synchronize()
                    record(key, np.asarray(y.get() if hasattr(y, "get") else y), time.perf_counter() - t)
        except Exception as e:  # noqa: BLE001
            print(f"  cuML unavailable or failed: {repr(e)[:200]}", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
