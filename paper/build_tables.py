"""Build the paper's result tables from the raw result files (fatsne/results/*.json).
Usage: python build_tables.py [main] [competitors] [labeled] [singlecell]   (default: every table whose data exists)
Bootstrap: paired per-seed differences, 10,000 resamples of seeds (per dataset) and of (dataset, seed) pairs
stratified by dataset (pooled), percentile 95% intervals, fixed generator seed.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "fatsne" / "results"
TAB = HERE / "tables"
DATASETS = (("SIFT1M", ""), ("DEEP1M", ""), ("GloVe-100", ""), ("T2I-1M", r"$^\dagger$"), ("dbpedia-1M", r"$^\dagger$"))
B = 10_000


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def boot_ci(x, rng):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), (B, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def kl(r):
    return r["kl_exact"]["3000"] if isinstance(r["kl_exact"], dict) else r["kl_exact"]


def main_table():
    m = load("paper_v2_main.json")
    rng = np.random.default_rng(0)
    rows, pooled = [], []
    for lab, dag in DATASETS:
        a = {arm: [kl(r) for r in m[lab][arm]] for arm in ("std", "tuned", "fourier_v1")}
        nn = {arm: [r["knn10"] for r in m[lab][arm]] for arm in ("std", "tuned", "fourier_v1")}
        s = min(len(v) for v in a.values())
        best = [min(a["std"][i], a["tuned"][i]) for i in range(s)]
        gap = [100 * (a["fourier_v1"][i] - best[i]) / best[i] for i in range(s)]
        pooled.append(gap)
        lo, hi = boot_ci(gap, rng)
        nb = max(np.median(nn["std"][:s]), np.median(nn["tuned"][:s]))
        rows.append(rf"{lab}{dag} & {np.median(a['std'][:s]):.3f} & {np.median(a['tuned'][:s]):.3f} & "
                    rf"{np.median(a['fourier_v1'][:s]):.3f} & ${np.mean(gap):+.1f}\%$ [${lo:+.1f}$, ${hi:+.1f}$] & "
                    rf"{sum(g < 0 for g in gap)}/{s} & {np.median(nn['tuned'][:s]):.3f} & "
                    rf"{np.median(nn['fourier_v1'][:s]):.3f} & ${100 * (np.median(nn['fourier_v1'][:s]) / nb - 1):+.0f}\%$\\")
    # pooled: resample seeds within each dataset, average the dataset means
    means = np.array([[np.asarray(g)[rng.integers(0, len(g), len(g))].mean() for g in pooled] for _ in range(B)]).mean(1)
    plo, phi = np.percentile(means, [2.5, 97.5])
    pm = np.mean([np.mean(g) for g in pooled])
    s = min(len(g) for g in pooled)
    head = (r"\begin{tabular}{lrrrlrrrr}\toprule" "\n"
            rf" & \multicolumn{{5}}{{c}}{{exact KL after 3000 iterations (median of {s} seeds)}} & "
            r"\multicolumn{3}{c}{10-NN preservation (median)}\\" "\n"
            r"\cmidrule(lr){2-6}\cmidrule(lr){7-9}" "\n"
            r"dataset & std & tuned & Fourier & mean change vs better baseline [95\% CI] & wins & tuned & Fourier & vs best\\\midrule" "\n")
    foot = (r"\midrule" "\n" rf"all five & & & & ${pm:+.1f}\%$ [${plo:+.1f}$, ${phi:+.1f}$] & "
            rf"{sum(sum(x < 0 for x in g) for g in pooled)}/{sum(len(g) for g in pooled)} & & & \\" "\n"
            r"\bottomrule\end{tabular}" "\n")
    (TAB / "main_exact.tex").write_text(head + "\n".join(rows) + "\n" + foot, encoding="utf-8")
    print("\n".join(rows)); print(f"pooled {pm:+.2f}% [{plo:+.2f}, {phi:+.2f}]")


ARM_NAMES = {"std": "standard", "tuned": "tuned", "fourier_v1": "Fourier (ours)", "fourier_v2": "Fourier, gains cap 10",
             "lambert": "Nesterov (Lambert et al.)", "lambert_g4096": r"Nesterov, $4096^2$ grid", "lambert_fourier": "Nesterov + Fourier", "ars": r"ARS (Lu \& Calder)"}


def competitors_table():
    c = load("paper_competitors_main.json")
    m = load("paper_v2_main.json")
    arms = ("tuned", "fourier_v1", "lambert", "lambert_g4096", "lambert_fourier", "ars")
    head = (r"\begin{tabular}{l" + "r" * len(arms) + "}\\toprule\n"
            "exact KL / 10-NN & " + " & ".join(ARM_NAMES[a] for a in arms) + r"\\\midrule" + "\n")
    rows = []
    for lab, dag in DATASETS:
        if lab not in c:
            continue
        cells = []
        for a in arms:
            runs = (m[lab] if a in ("tuned", "fourier_v1") else c[lab]).get(a, [])
            runs = [r for r in runs if r.get("finite", True) and "kl_exact" in r]
            cells.append("--" if not runs else f"{np.median([kl(r) for r in runs]):.3f} / {np.median([r['knn10'] for r in runs]):.3f}"
                         + ("" if len(runs) >= 3 else f" ({len(runs)})"))
        rows.append(f"{lab}{dag} & " + " & ".join(cells) + r"\\")
    (TAB / "competitors.tex").write_text(head + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    print("\n".join(rows))


def labelled_table(fname, out, datasets):
    r = load(fname)
    arms = ("std", "tuned", "fourier_v1", "fourier_v2", "lambert", "lambert_g4096", "lambert_fourier", "ars")
    head = (r"\begin{tabular}{llrrrr}\toprule" "\n"
            r"data & optimiser & exact KL & 10-NN & $k$-NN accuracy & trustworthiness\\\midrule" "\n")
    rows = []
    for d in datasets:
        if d not in r:
            continue
        first = True
        for a in arms:
            runs = [x for x in r[d].get(a, []) if x.get("finite", True) and "kl_exact" in x]
            if not runs:
                continue
            med = {k: np.median([x[k] for x in runs]) for k in ("knn10", "knn_acc", "trust")}
            rows.append(f"{d if first else ''} & {ARM_NAMES[a]} & {np.median([kl(x) for x in runs]):.3f} & {med['knn10']:.3f} & "
                        f"{med['knn_acc']:.3f} & {med['trust']:.4f}\\\\")
            first = False
    (TAB / out).write_text(head + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    print("\n".join(rows))


def scaling_table():
    s = load("paper_v2_scaling.json")
    sizes = [n for n in ("10000", "30000", "100000", "300000", "1000000", "3000000") if n in s]
    lab = {"10000": r"$10^4$", "30000": r"$3\times10^4$", "100000": r"$10^5$", "300000": r"$3\times10^5$",
           "1000000": r"$10^6$", "3000000": r"$3\times10^6$"}
    head = (r"\begin{tabular}{l" + "r" * len(sizes) + r"}\toprule" + "\n$n$ & " + " & ".join(lab[n] for n in sizes)
            + r"\\\midrule" + "\n")
    rows = []
    med = lambda n, a, it: float(np.median([r["kl_exact"][it] for r in s[n].get(a, [])])) if s[n].get(a) else None
    for it in ("1000", "3000", "6000"):
        for a, name in (("std", "standard"), ("tuned", "tuned"), ("fourier_v2", "Fourier, gains cap 10")):
            cells = [med(n, a, it) for n in sizes]
            rows.append(f"{name}, {it} it. & " + " & ".join("--" if c is None else f"{c:.3f}" for c in cells) + r"\\")
        gaps = [(med(n, "fourier_v2", it), med(n, "tuned", it)) for n in sizes]
        rows.append(r"\quad Fourier vs tuned & " + " & ".join(
            "--" if f is None or t is None else (r"div." if f > 2 * t else f"${100 * (f / t - 1):+.1f}\\%$") for f, t in gaps)
            + r"\\" + ("" if it == "6000" else r"[2pt]"))
    (TAB / "scaling.tex").write_text(head + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}" + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    if "scaling" in sys.argv[1:]:
        scaling_table()
    want = set(sys.argv[1:]) or {"main", "competitors", "labeled", "singlecell"}
    if "main" in want and load("paper_v2_main.json"):
        main_table()
    if "competitors" in want and load("paper_competitors_main.json"):
        competitors_table()
    if "labeled" in want and load("paper_competitors_labeled.json"):
        labelled_table("paper_competitors_labeled.json", "labelled.tex", ("MNIST-70k", "Fashion-70k"))
    if "singlecell" in want and load("paper_competitors_singlecell.json"):
        labelled_table("paper_competitors_singlecell.json", "singlecell.tex", ("10x-1.3M",))
