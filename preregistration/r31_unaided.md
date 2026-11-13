# novel_engines round 31: does the Nesterov optimiser's advantage grow with n? (sealed before any run)
Context: r30 failed (Fourier + Nesterov is much worse than Nesterov). Lambert et al.'s Nesterov optimiser beat fourier_v1 at
1M on SIFT1M (-3.7%) and DEEP1M (-3.0%), tied on GloVe-100 (one seed), and trailed tuned on MNIST-70k (+3.9%).
Interpretation to test (critical slowing down): the long modes relax at rates ~ k^2; a heavy momentum method needs about
sqrt(kappa) iterations instead of kappa, while the capped Fourier boost and the tuned learning rate divide the iteration
count by bounded factors. So Nesterov's advantage should grow with n.
Test: the T2I nested subsets of the running scaling study (n = 1e4, 3e4, 1e5, 3e5, 1e6), same affinities, 3 seeds, 6000
iterations, Lambert's scheme (grid 1536), exact KL at 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000 iterations
(counted from the start for every arm). Comparison arms std / tuned / fourier_v2 from paper_v2_scaling.json (same runs
as the scaling study). Gap(n) = median over seeds of Lambert's exact KL at 3000 iterations, relative to the lower of the
std / tuned / fourier_v2 medians at the same n and iteration.
Predictions:
(S1) Gap(1e6) < Gap(1e5) < Gap(1e4)  (the advantage grows with n).
(S2) Gap(1e4) >= -0.5%  (no real advantage at 1e4)  and  Gap(1e6) <= -2%.
(S3) the same ordering as S1 holds at 6000 iterations.
Kill rule: S1 or S2 fails -> the paper does not claim that Nesterov's advantage grows with n; it reports the measured gaps.
