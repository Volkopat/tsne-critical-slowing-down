"""Invariants of the Fourier-accelerated t-SNE engine (CPU, small)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import (FourierPreconditioner, HelmholtzPreconditioner, affinities, kl_divergence,  # noqa: E402
                    repulsion_exact, repulsion_grid, tsne)

torch.set_num_threads(4)


@pytest.fixture(scope="module")
def blobs():
    g = np.random.default_rng(0)
    centres = g.normal(0, 6, (6, 10))
    x = centres[g.integers(0, 6, 1500)] + g.normal(0, 1, (1500, 10))
    return torch.as_tensor(x, dtype=torch.float32)


@pytest.mark.parametrize("pc", [FourierPreconditioner(64, 16.0), HelmholtzPreconditioner(64, 64.0, 4.0)])
def test_preconditioner_is_symmetric_and_at_least_identity(pc):
    g = torch.Generator().manual_seed(1)
    y = torch.randn(2000, 2, generator=g, dtype=torch.float64) * 20
    a = torch.randn(2000, 2, generator=g, dtype=torch.float64)
    b = torch.randn(2000, 2, generator=g, dtype=torch.float64)
    lhs, rhs = float((a * pc(y, b)).sum()), float((pc(y, a) * b).sum())
    assert abs(lhs - rhs) < 1e-6 * (abs(lhs) + 1)
    assert float((a * pc(y, a)).sum()) >= float((a * a).sum()) - 1e-9  # I + PSD: never shrinks a direction


def test_cap_one_is_the_identity():
    y = torch.randn(500, 2) * 5
    g = torch.randn(500, 2)
    assert torch.allclose(FourierPreconditioner(32, 1.0)(y, g), g, atol=1e-6)


def test_grid_repulsion_matches_exact():
    g = torch.Generator().manual_seed(2)
    y = torch.randn(3000, 2, generator=g) * 8
    fe, ze = repulsion_exact(y)
    fg, zg = repulsion_grid(y, h_target=0.1, max_grid=2048)
    assert abs(zg - ze) / ze < 5e-3
    assert float((fg - fe).norm() / fe.norm()) < 2e-2


def test_affinities_are_a_symmetric_distribution(blobs):
    i, j, v = affinities(blobs, 30.0)
    assert abs(float(v.sum()) - 1.0) < 1e-5 and float(v.min()) >= 0
    P = torch.sparse_coo_tensor(torch.stack([i, j]), v, (1500, 1500)).to_dense()
    assert torch.allclose(P, P.T, atol=1e-7)


def test_kl_is_finite_with_zero_affinities(blobs):
    i, j, v = affinities(blobs, 30.0)
    v = v.clone()
    v[:100] = 0.0
    y = torch.randn(1500, 2)
    _, z = repulsion_exact(y)
    assert np.isfinite(kl_divergence(y, i, j, v, z))


def test_engine_lowers_kl_faster_than_standard(blobs):
    aff = affinities(blobs, 30.0)
    _, base, _ = tsne(blobs, device="cpu", affin=aff, n_iter=250, early_iter=250, record=(400, 500))
    _, fast, _ = tsne(blobs, device="cpu", affin=aff, n_iter=250, early_iter=250, record=(400, 500),
                      precond=FourierPreconditioner(64, 16.0))
    assert fast[500][0] < base[500][0]


def test_without_preconditioner_the_engine_is_the_standard_optimizer(blobs):
    aff = affinities(blobs, 30.0)
    y1, h1, _ = tsne(blobs, device="cpu", affin=aff, n_iter=50, early_iter=50, record=(100,))
    y2, h2, _ = tsne(blobs, device="cpu", affin=aff, n_iter=50, early_iter=50, record=(100,), precond=FourierPreconditioner(64, 1.0))
    assert torch.allclose(y1, y2, atol=1e-4)
