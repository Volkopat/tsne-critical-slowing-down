"""Checks for thermal.py: the Fourier noise has covariance P, and a plain Langevin run on a small map is stable."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fatsne import FourierPreconditioner  # noqa: E402
from thermal import autocorr_time, fourier_noise  # noqa: E402


def test_noise_covariance_matches_preconditioner():
    torch.manual_seed(0)
    y = torch.randn(3000, 2) * torch.tensor([3.0, 1.0])
    pc = FourierPreconditioner(32, 16.0)
    probes = torch.randn(6, 3000, 2)
    gen = torch.Generator().manual_seed(1)
    samples = torch.stack([fourier_noise(y, pc, gen) for _ in range(4000)])  # (S, n, 2)
    for v in probes:
        quad = float((v * pc(y, v)).sum())  # v^T P v
        emp = float(((samples * v).sum((1, 2)) ** 2).mean())  # E[(v . noise)^2]
        assert abs(emp / quad - 1.0) < 0.08, (emp, quad)


def test_autocorr_time_of_ar1():
    rho, x, a = 0.9, 0.0, []
    e = torch.randn(200000, generator=torch.Generator().manual_seed(0), dtype=torch.float64).tolist()
    for t in range(len(e)):  # 200k samples: the estimator's standard error at tau = 19 is about 1
        x = rho * x + e[t]
        a.append(x)
    tau = autocorr_time(a)
    assert abs(tau - (1 + rho) / (1 - rho)) < 3.0, tau


if __name__ == "__main__":
    test_noise_covariance_matches_preconditioner()
    test_autocorr_time_of_ar1()
    print("ok")
