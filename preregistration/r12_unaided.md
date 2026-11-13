# novel_engines round 12: paper evidence suite (sealed before any of these runs)

Paper: "Critical slowing down in t-SNE and its cure by Fourier acceleration". Frozen engine: Fourier
preconditioner, cap 16, after EE; grid by the frozen rule 2^round(log2(8 n^(1/4))) clipped to [64, 512]
(64 at 10k, 128 at 30k-100k, 256 at 300k-3M); grid repulsion h = 0.25; PCA init + seeded jitter 0.1.
Arms: std (FIt-SNE schedule), tuned (lr n after EE + momentum 0.9), fourier.

Block A (paper_main.py): SIFT1M, DEEP1M, GloVe-100, T2I-1M, dbpedia-1M; 3 seeds; 3000 iterations.
 (A1) fourier's KL@3000 < tuned's in >= 13 of 15 runs; (A2) median relative gap >= 2%;
 (A3) fourier reaches tuned's KL@3000 within <= 1375 post-EE iterations in >= 11 of 15 runs;
 (A4) fourier's 10-NN >= tuned's in >= 11 of 15 runs.
Block B (paper_scaling.py): T2I subsets 10k..3M; 3 seeds; 6000 iterations.
 (B1) the fitted exponent of post-EE iterations to within 4% of KL_ref is lower for fourier than for tuned
 by >= 0.2, with non-overlapping 90% seed-bootstrap intervals (if tuned reaches the target at >= 3 sizes);
 (B2) fourier's KL@3000 < tuned's in every run at n >= 1M; (B3) at n <= 30k tuned's KL@6000 is within 1%
 of or below fourier's (no claim of benefit at small n).
Block C (paper_modes.py): n = 100k T2I, converged reference, single-mode perturbations m = 1..32.
 (C1) for std, the decay rate grows with m with a log-log slope >= 1 for m <= 8 (transverse modes);
 (C2) at m <= 2 fourier's transverse decay rate is >= 3x std's.
