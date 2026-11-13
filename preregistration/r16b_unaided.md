# novel_engines round 16b: posterior neighbour embedding (sealed before batch T probes it)

U9 Posterior Neighbour Embedding (PNE), my own idea, compounding three fields:
- Model (Bayesian statistics): treat the neighbour pairs as N observations drawn from Q(y); the likelihood is
  exp(-N KL(P||Q(y))) up to a constant, so the embedding has a posterior pi(y) ~ exp(-beta KL), beta ~ N. Output is a
  distribution over maps: posterior mean map, per-point positional covariance (after Procrustes alignment to a
  reference), probability that j is among i's 10 nearest map neighbours, credibility of each cluster separation.
- Engine (lattice field theory): Fourier-accelerated Langevin, the original use of Fourier acceleration (Batrouni et
  al. 1985). Drift -h P grad U; noise with covariance exactly P, drawn as xi + S^T D^(-1/2) F^-1 sqrt(M-1) F zeta
  (xi, zeta standard normal), which needs no matrix square root.
- Theory (statistical mechanics of 2-D crystals): our measured k^2 law makes the map a 2-D elastic medium, so by
  equipartition <|u_k|^2> = T/(mu k^2) and the variance of a point's absolute position grows like log(n)
  (Mermin-Wagner: no long-range positional order in 2-D), while relative positions of neighbours stay O(1/beta).

Sealed predictions (to be tested when a GPU is available; dev data MNIST-70k and T2I subsets):
(P1) Under PNE at fixed beta per point, the posterior variance of absolute positions grows with log n across
     10^4 -> 10^6 (fitted slope in log n significantly > 0), while the variance of neighbour displacements does not.
(P2) The spectrum of posterior displacement variance follows ~ 1/k^2 at small k (log-log slope -2 +- 0.3).
(P3) Fourier-accelerated Langevin decorrelates the slowest map mode at least 5x faster per iteration than
     unaccelerated Langevin at the same step size (integrated autocorrelation time of the m = 1 shear amplitude).
(P4) Points with high posterior positional variance are enriched for points whose high-dimensional neighbourhood
     is ambiguous (10-NN label disagreement on labelled data): Spearman > 0.2.
Kill rule: if batch T or a later probe finds published posterior sampling of a neighbour-embedding objective with
per-point uncertainty, PNE is not claimed as new; the Mermin-Wagner analysis would then be checked separately.
