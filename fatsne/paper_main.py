"""Paper, block A (main table; sealed predictions in novel/r12_unaided.md). Five 1M-scale datasets:
SIFT1M, DEEP1M, GloVe-100 (1.18M), and the held-out T2I-1M and dbpedia-1M; arms std / tuned / fourier;
seeds 0, 1, 2; 3000 iterations (EE 12x for 250). Records KL every 100 iterations, 10-NN preservation,
embedding extent, and saves the seed-0 embeddings (float16) for the figures. For T2I-1M and dbpedia the
seed-0/1 runs of held-out 2 (same frozen configuration) are reused and only seed 2 is added.
One GPU alone.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import affinities, tsne  # noqa: E402
from paperlib import load_any, run, sampled_knn_preservation  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
EMB = OUT / "embeddings"
EMB.mkdir(parents=True, exist_ok=True)
CHECK = tuple(range(300, 3001, 100))
SETS = [("SIFT1M", "SIFT", 1_000_000), ("DEEP1M", "DEEP", 1_000_000), ("GloVe-100", "GloVe-100", None),
        ("T2I-1M", "T2I-1M", None), ("dbpedia-1M", "dbpedia", None)]


def main():
    res_path = OUT / "paper_main.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    prior = json.loads((OUT / "heldout2.json").read_text()) if (OUT / "heldout2.json").exists() else {}
    prior_key = {"T2I-1M": "T2I-1M", "dbpedia-1M": "dbpedia-openai-1M"}
    for label, name, n in SETS:
        done = res.get(label, {})
        if all(len(done.get(a, [])) == 3 for a in ("std", "tuned", "fourier")):
            continue
        x = torch.as_tensor(load_any(name, n), device="cuda")
        x = x - x.mean(0)
        aff = affinities(x, 30.0)
        tsne(x[:20000], n_iter=20, early_iter=10, seed=99, repulsion="grid")
        rows = {a: list(done.get(a, [])) for a in ("std", "tuned", "fourier")}
        if label in prior_key and prior_key[label] in prior and not any(rows.values()):
            for a in rows:
                rows[a] = [dict(r, reused_from="heldout2") for r in prior[prior_key[label]][a]]
        for arm in ("std", "tuned", "fourier"):
            for seed in range(len(rows[arm]), 3):
                y, r = run(x, aff, arm, seed, 3000, CHECK)
                r["knn10"] = sampled_knn_preservation(x, y)
                r["extent"] = float((y - y.mean(0)).norm(dim=1).pow(2).mean().sqrt())
                rows[arm].append(r)
                if seed == 0 or (seed == 2 and label in prior_key):
                    np.save(EMB / f"{label}_{arm}_seed{seed}.npy", y.cpu().numpy().astype(np.float16))
                print(f"  {label:11s} {arm:8s} seed {seed}: KL@1000 {r['kl']['1000']:.4f} KL@3000 {r['kl']['3000']:.4f} "
                      f"10-NN {r['knn10']:.4f} {r['sec']['3000']:.1f}s", flush=True)
                res[label] = rows
                res_path.write_text(json.dumps(res, indent=1))
        for arm in ("std", "tuned", "fourier"):
            ks = [r["kl"]["3000"] for r in rows[arm]]
            print(f"  {label:11s} {arm:8s} KL@3000 median {np.median(ks):.4f} {[round(k, 4) for k in ks]} | 10-NN "
                  f"{np.median([r['knn10'] for r in rows[arm]]):.4f}", flush=True)
        del x, aff
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
