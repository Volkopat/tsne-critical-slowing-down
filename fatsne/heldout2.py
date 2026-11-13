"""FA-tSNE HELD-OUT 2 (sealed predictions: novel/r11_unaided.md). Frozen engine: FourierPreconditioner
grid 256 (n >= 300k rule), cap 16, on after early exaggeration; grid repulsion h = 0.25, max grid 1536;
init PCA + seeded jitter 0.1; 3000 iterations (EE 12x for 250). Baselines: tuned (lr n after EE +
momentum 0.9) and std (FIt-SNE schedule). Data never used before: dbpedia-entities-openai-1M (text,
1536-d) and Yandex Text-to-Image first 1M (images, 200-d). 2 seeds each. Scored once.
One GPU alone.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp4 import sampled_knn_preservation  # noqa: E402
from fatsne import FourierPreconditioner, affinities, tsne  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
CHECK = tuple(sorted(set(range(300, 1001, 50)) | set(range(1100, 3001, 100))))
DATA = "/tmp/emeans_data/new"


def main():
    res_path = OUT / "heldout2.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for label, f in (("T2I-1M", "t2i-1M.npy"), ("dbpedia-openai-1M", "dbpedia-openai-1M.npy")):
        if label in res:
            continue
        x = torch.as_tensor(np.load(f"{DATA}/{f}"), device="cuda")
        x = x - x.mean(0)
        n = x.shape[0]
        aff = affinities(x, 30.0)
        tsne(x[:20000], n_iter=20, early_iter=10, seed=99, repulsion="grid")
        ARMS = [("std", {}, None), ("tuned", dict(lr_late=float(n), momentum_late=0.9), None),
                ("fourier", {}, lambda: FourierPreconditioner(256, 16.0))]
        rows = {}
        for arm, kw, pc in ARMS:
            runs = []
            for seed in (0, 1):
                y, hist, _ = tsne(x, affin=aff, seed=seed, init_jitter=0.1, n_iter=2750, record=CHECK, repulsion="grid",
                                  grid_h=0.25, max_grid=1536, precond=pc() if pc else None, **kw)
                runs.append({"kl": {str(k): w[0] for k, w in hist.items()}, "sec": {str(k): w[1] for k, w in hist.items()},
                             "knn10": sampled_knn_preservation(x, y)})
            rows[arm] = runs
        for arm, runs in rows.items():
            its, secs = [], []
            for s, r in enumerate(runs):
                target = rows["tuned"][s]["kl"]["3000"]
                hit = [k for k in CHECK if r["kl"][str(k)] <= target]
                its.append(hit[0] - 250 if hit else None)
                secs.append(round(r["sec"][str(hit[0])], 1) if hit else None)
            print(f"  {label:18s} {arm:8s} " + " ".join(f"{k}:{np.median([r['kl'][str(k)] for r in runs]):.4f}" for k in (1000, 2000, 3000))
                  + f" | 10-NN {[round(r['knn10'], 4) for r in runs]} | KL@3000 {[round(r['kl']['3000'], 4) for r in runs]}"
                  + f" | post-EE its to tuned@3000 {its} s {secs}", flush=True)
        res[label] = rows
        res_path.write_text(json.dumps(res, indent=1))
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
