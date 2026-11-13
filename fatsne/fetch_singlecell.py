"""10x Genomics 1.3M mouse brain cells (public) to pod-local /tmp, standard preprocessing:
library-size normalisation to the median, log1p, the 1000 genes with the largest variance of log expression (estimated on
200k random cells), centring, 50 principal components on the GPU. Labels: 10x's own graph-based clustering
(analysis/clustering/graphclust/clusters.csv) from the published analysis bundle. Output: /tmp/emeans_data/new/10x-1.3M-pca50.npz
"""

import csv
import io
import subprocess
import tarfile
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch

D = Path("/tmp/emeans_data/new")
D.mkdir(parents=True, exist_ok=True)
H5 = D / "1M_neurons_filtered_gene_bc_matrices_h5.h5"
TAR = D / "1M_neurons_analysis.tar.gz"
BASE = "https://cf.10xgenomics.com/samples/cell-exp/1.3.0/1M_neurons/"
for f in (H5, TAR):
    if not f.exists() or f.stat().st_size == 0:
        subprocess.run(["curl", "-s", "--fail", "--retry", "4", "-o", str(f), BASE + f.name], check=True)
    print(f.name, f.stat().st_size, flush=True)

with h5py.File(H5, "r") as h:
    g = h[list(h.keys())[0]]
    n_genes, n_cells = (int(v) for v in g["shape"][:])
    data = g["data"][:].astype(np.float32)
    indices = g["indices"][:]
    indptr = g["indptr"][:]
    barcodes = [b.decode() for b in g["barcodes"][:]]
print("matrix", n_genes, "genes x", n_cells, "cells, nnz", len(data), flush=True)
lib = np.add.reduceat(data, indptr[:-1].clip(max=len(data) - 1)).astype(np.float64)
lib[np.diff(indptr) == 0] = 0.0
scale = np.median(lib[lib > 0]) / np.maximum(lib, 1.0)
data *= np.repeat(scale, np.diff(indptr)).astype(np.float32)
np.log1p(data, out=data)
M = sp.csc_matrix((data, indices, indptr), shape=(n_genes, n_cells))
rng = np.random.default_rng(0)
sub = np.sort(rng.choice(n_cells, 200000, replace=False))
S = M[:, sub].tocsr()
mean = np.asarray(S.mean(1)).ravel()
sq = np.asarray(S.multiply(S).mean(1)).ravel()
hvg = np.argsort(-(sq - mean ** 2))[:1000]
print("selected 1000 genes", flush=True)
X = M[hvg, :].T.tocsr().astype(np.float32)  # cells x 1000
del M, S, data, indices
Xd = torch.zeros((n_cells, 1000), dtype=torch.float32)
for s in range(0, n_cells, 100000):
    Xd[s:s + 100000] = torch.as_tensor(X[s:s + 100000].toarray())
Xg = Xd.cuda()
Xg -= Xg.mean(0)
U, Sv, V = torch.svd_lowrank(Xg, q=60, niter=4)
pcs = (Xg @ V[:, :50]).cpu().numpy().astype(np.float32)
print("PCA done", pcs.shape, "explained variance share", float((Sv[:50] ** 2).sum() / (Xg ** 2).sum()), flush=True)

with tarfile.open(TAR) as tf:
    member = next(m for m in tf.getmembers() if m.name.endswith("graphclust/clusters.csv"))
    rows = list(csv.reader(io.TextIOWrapper(tf.extractfile(member), encoding="utf-8")))[1:]
lab = {bc: int(c) for bc, c in rows}
y = np.array([lab.get(bc, -1) for bc in barcodes], dtype=np.int64)
keep = y >= 0
print("labelled cells", int(keep.sum()), "of", n_cells, "clusters", len(set(y[keep])), flush=True)
np.savez(D / "10x-1.3M-pca50.npz", x=pcs[keep], y=y[keep] - y[keep].min())
print("SINGLECELL_DONE")
