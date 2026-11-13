# novel_engines round 14: corrected paper suite (sealed before paper_v2.py runs)

What rounds 12-13 revealed: (i) the grid KL normaliser underestimates Z by 1.1-3.3% for the larger
Fourier maps (exact check at 1M; 0.05-0.6% for the baselines), so earlier KL margins were overstated by
0.3-0.7 points; (ii) engine v1 (cap 16, grid rule with a 64 floor, unbounded gains) diverged at 10k points
(grid 64) and inflated in 6000-iteration runs at 30k-100k; (iii) the earlier scaling exponents used a
single-best-run reference with biased KL and excluded sizes where v1 failed.

Engine v2: FourierPreconditioner cap 16, grid max(128, rule), per-coordinate gains capped at 10, on after
early exaggeration. All KL below is EXACT (computed from map snapshots after each run).

Main (5 datasets, 3 seeds, 3000 iterations; arms std, tuned, fourier_v1, fourier_v2):
(M1) fourier_v2's exact KL@3000 is below the better of std and tuned (same seed) in >= 13 of 15 runs.
(M2) the median relative gap in (M1) is >= 1.5%.
(M3) fourier_v2's 10-NN >= tuned's in >= 13 of 15 runs.
(M4) fourier_v2's exact KL@3000 is within 0.5% of fourier_v1's (the cap does not cost quality at 1M).
Scaling (T2I subsets 10k..3M, 3 seeds, 6000 iterations; arms std, tuned, fourier_v2):
(S1) no fourier_v2 run diverges: exact KL at 6000 <= exact KL at 1000 in every run.
(S2) at n >= 1M, fourier_v2's median exact KL@3000 is below tuned's median exact KL@6000.
(S3) at n <= 30k, tuned's median exact KL@6000 is within 1% of or below fourier_v2's.
(S4) define r(n) = (median exact KL@3000 of fourier_v2) / (median exact KL@3000 of tuned); r decreases
monotonically from n = 100k to 3M (the advantage grows with n).
