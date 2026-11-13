"""Public labelled benchmarks to pod-local /tmp: MNIST and Fashion-MNIST (70k each) from OpenML."""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml

D = Path("/tmp/emeans_data/new")
D.mkdir(parents=True, exist_ok=True)
for label, name in (("MNIST-70k", "mnist_784"), ("Fashion-70k", "Fashion-MNIST")):
    out = D / f"{label}.npz"
    if out.exists():
        continue
    X, y = fetch_openml(name, version=1, as_frame=False, parser="liac-arff", return_X_y=True, data_home="/tmp/openml")
    np.savez(out, x=(np.asarray(X, dtype=np.float32) / 255.0), y=np.asarray(y).astype(np.int64))
    print(label, X.shape, np.bincount(np.asarray(y).astype(np.int64)), flush=True)
print("LABELED_DONE")
