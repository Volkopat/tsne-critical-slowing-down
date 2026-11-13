"""Figure 2 from the runs of Table 1 (paper_v2_main.json): KL against iteration for std / tuned / Fourier.
Lines: median over seeds with the approximate (grid) normaliser; band: range over seeds; markers: median exact KL."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
m = json.loads((HERE.parent / "fatsne" / "results" / "paper_v2_main.json").read_text())
ARMS = (("std", "standard", "#E69F00"), ("tuned", "tuned", "#56B4E9"), ("fourier_v1", "Fourier", "#009E73"))
DATA = ("SIFT1M", "DEEP1M", "GloVe-100", "T2I-1M", "dbpedia-1M")

fig, axes = plt.subplots(1, 5, figsize=(13, 2.9), constrained_layout=True)
for ax, lab in zip(axes, DATA):
    for arm, name, col in ARMS:
        runs = m[lab][arm]
        its = sorted(int(k) for k in runs[0]["kl_grid"])
        g = np.array([[r["kl_grid"][str(i)] for i in its] for r in runs])
        ax.fill_between(its, g.min(0), g.max(0), color=col, alpha=0.18, linewidth=0)
        ax.plot(its, np.median(g, 0), color=col, linewidth=1.6, label=name)
        snaps = sorted(int(k) for k in runs[0]["kl_exact"])
        e = np.median([[r["kl_exact"][str(i)] for i in snaps] for r in runs], 0)
        ax.plot(snaps, e, linestyle="none", marker="o", markersize=4, markerfacecolor="white", markeredgecolor=col,
                markeredgewidth=1.3)
    lo = min(np.median([r["kl_exact"]["3000"] for r in m[lab][a]]) for a, _, _ in ARMS)
    hi = max(np.median([[r["kl_grid"]["500"]] for r in m[lab][a]]) for a, _, _ in ARMS)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("iteration", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("KL divergence", fontsize=9)
axes[0].legend(fontsize=8, frameon=False, loc="upper right")
out = HERE / "figures" / "kl_paper_main.pdf"
fig.savefig(out)
fig.savefig(out.with_suffix(".png"), dpi=110)
print("saved", out)
