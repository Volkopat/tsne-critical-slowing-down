# novel_engines round 22: confirmatory test of the determinacy gap (sealed before any run)
Exploratory observation from r16d (SIFT, c = 100, not claimed): median var_abs / var_rel = 4.4, 14, 51 at n = 1e4, 3e4,
1e5. Hypothesis: the global layout of a t-SNE map becomes progressively less determined than its local structure.
Fresh data: DEEP1M prefixes (never used in r16d), n = 1e4, 3e4, 1e5; same code (paper_thermal.py, tuned map 3000
iterations, Fourier-accelerated Langevin, 1000 burn-in + 4000 steps), inverse temperature c = 100 only.
Scale-normalised measures (fixed now): s = median distance to the 10th map neighbour of the tracked points in the
converged map; A(n) = median var_abs / s^2, Rl(n) = median var_rel / s^2, R(n) = median var_abs / median var_rel.
Predictions:
(Q1) R(n) increases strictly from 1e4 to 3e4 to 1e5.
(Q2) R(1e5) / R(1e4) >= 4.
(Q3) A(n) increases strictly with n while Rl(n) does not increase.
Kill rule: if Q1 or Q2 fails, the determinacy gap is not claimed.
