# Critical Slowing Down in t-SNE Optimisation

D. Patil, November 2026. Paper: [`paper/tsne-critical-slowing-down.pdf`](paper/tsne-critical-slowing-down.pdf).

t-SNE needs more iterations as datasets grow. This project treats that as *critical slowing down* in the sense of
statistical physics, measures it, and tests one physics remedy against standard and recent optimisers at up to
3 million points, with exact KL divergence and predictions sealed before each experiment.

## Findings in brief

- **Relaxation spectrum.** Around an optimised 100k-point map, the short-time relaxation rate of a collective
  displacement mode grows as k², and shear modes relax 3–4× more slowly than compression modes. This was measured at
  one size.
- **Two exact results.** A virial identity holds at every stationary point and certifies non-convergence along the
  dilation. The objective also flattens along the dilation at large scale.
- **Fourier acceleration** (a capped 1/k² boost of the gradient's spatial Fourier modes, two FFTs per iteration) helps
  the standard optimiser at large n:
  - the pre-registered (capped) engine won 11 of 15 runs at about 10⁶ points (median gain 2.1%);
  - an uncapped variant, chosen after seeing those results, won 24 of 25 runs (3.1% on average);
  - against a tuned optimiser (capped engine, 3000 iterations) it trails by about 4.4% at 10⁴ points and leads by
    4.7% at 3 × 10⁶.
- **It is not the best optimiser.**
  - The Nesterov scheme of Lambert et al. (2023) reaches 2.6–3.7% lower KL on four of six million-point datasets.
  - Preconditioning that scheme with the Fourier operator makes it worse.
  - Fourier acceleration is best on 1.3 million single cells.
  - The Nesterov scheme inflates small maps, with grid or exact repulsion.
- **A methodological caveat for t-SNE benchmarks.** The grid approximation of the normaliser Z understates it by
  1.4–3.5% for wide maps against 0.1–0.8% for standard ones, which biases approximate-KL comparisons. All KL values
  in the paper's tables are exact.

All sealed predictions, including those that failed, are listed in the paper's appendix. The original texts are in
[`preregistration/`](preregistration/).

## Layout

| Path | Contents |
|---|---|
| `paper/` | LaTeX source, tables, figures, the table and figure builders (`build_tables.py`, `make_fig_main.py`), and the PDF |
| `fatsne/fatsne.py` | GPU t-SNE with grid and exact repulsion, the Fourier preconditioner and the exact normaliser |
| `fatsne/competitors.py` | Reimplementations of the Nesterov scheme of Lambert et al. and the swarming scheme of Lu & Calder |
| `fatsne/paper_*.py` | The experiment drivers behind each table (main table, scaling, ablations, competitors, spectrum, external tools) |
| `fatsne/thermal.py`, `fatsne/paper_thermal.py` | The tempered Langevin study (inconclusive; see the paper's appendix) |
| `fatsne/results/` | Raw results as JSON; the tables are built from these |
| `preregistration/` | The sealed predictions, as written before each experiment ran |
| `capgpu.py` | Runs a script with a cap on its GPU memory, so jobs can share a GPU |

## Reproducing

Requirements: Python 3.10+, PyTorch with CUDA, NumPy, SciPy, scikit-learn, h5py and matplotlib (`requirements.txt`).
openTSNE and FIt-SNE are needed only for the external comparison. The paper builds with
[tectonic](https://tectonic-typesetting.github.io/).

Data are public and are not included:
- SIFT, DEEP and GloVe from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks), placed in a `data/` folder
  next to this repository;
- Yandex Text-to-Image (first 3M base vectors) and dbpedia OpenAI embeddings (1M);
- MNIST and Fashion-MNIST from OpenML (`fatsne/fetch_labeled.py`);
- the 10x Genomics 1.3M mouse brain cells (`fatsne/fetch_singlecell.py`).

`fatsne/paperlib.py` holds the data paths.

```bash
cd fatsne
python paper_v2.py main              # main table (MAIN_SEEDS=5 for five seeds)
python paper_v2.py scaling           # scaling study
python paper_competitors.py main     # other optimisers at ~1M points
cd ../paper
python build_tables.py               # rebuild the tables from fatsne/results
python make_fig_main.py              # rebuild Figure 2
tectonic -X compile main.tex
```

The million-point experiments were run on single NVIDIA B200 GPUs. Timings in the paper come only from runs that had
a GPU to themselves.

## Future work

The paper's "Future work" section lists the experiments that would support the analysis more firmly:
- the relaxation spectrum at several sizes;
- the spectrum of the Nesterov scheme;
- a corrected sampler for the Langevin study;
- a second dataset for the scaling study.

## Citation

```bibtex
@misc{patil2026tsne,
  author       = {Patil, D.},
  title        = {Critical Slowing Down in t-SNE Optimisation},
  year         = {2026},
  howpublished = {\url{https://github.com/Volkopat/tsne-critical-slowing-down}}
}
```

## Licence

The code is under the MIT licence ([`LICENSE`](LICENSE)). The paper (text, tables and figures in `paper/`) is under
[CC BY 4.0](paper/LICENSE).
