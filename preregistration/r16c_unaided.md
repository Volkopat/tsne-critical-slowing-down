# novel_engines round 16c: protocol for the Mermin-Wagner test (fixed before any real run; predictions in r16b)
Code: novel/fatsne/thermal.py (sampler; noise covariance = P verified by test_thermal.py), paper_thermal.py (driver).
Data: T2I prefix subsets n = 1e4, 3e4, 1e5, 3e5. Map: tuned t-SNE, 3000 iterations, seed 0.
Sampler: unadjusted Fourier-accelerated Langevin (grid max(128, rule), cap 16), lr n/12, h = lr/(c n),
inverse temperatures c = 10 and 100 observations per point; 1000 burn-in + 4000 steps, snapshot every 10.
Plain Langevin at c = 100 for P3 at n = 1e5 only. Tracked: 5000 random points, 10 nearest map neighbours.
Decision rules: P1 holds if the fitted slope of median var_abs in log n is > 0 for both c and at least 3x the slope of
median var_rel; P2 holds if the spectrum's log-log slope over k = 1..6 is within -2 +- 0.3 at n >= 1e5;
P3 holds if tau_m1(plain) / tau_m1(Fourier) >= 5 at n = 1e5.
Smoke test only so far (2000 synthetic points, CPU): variance scales as 1/c as equipartition requires.
