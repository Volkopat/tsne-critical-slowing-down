# novel_engines round 30: Fourier-preconditioned Nesterov (sealed before any run)
Trigger (observed, not claimed): the published Nesterov optimiser of Lambert et al. (ESANN 2023; gamma 0.995, normalised
gradient, no early exaggeration, PCA-whitened init), grid 1536, beat fourier_v1 at 1M on SIFT1M (exact KL 4.520 vs 4.696,
3 seeds) and DEEP1M (4.997 vs 5.153); 10-NN also higher (0.303 vs 0.291; 0.228 vs 0.221). On MNIST-70k it is behind tuned.
Candidate "lambert_fourier": Lambert's scheme unchanged except that the gradient at the look-ahead point is multiplied by
the Fourier preconditioner P (FourierPreconditioner(grid_v1(n), c = 16), the same operator as fourier_v1) before the
normalisation. P is SPD, so stationary points are unchanged.
Test: SIFT1M, DEEP1M, GloVe-100, T2I-1M, dbpedia-1M; 3 seeds; 3000 iterations; same affinities and repulsion grid (1536)
as lambert; exact KL; one B200 shared with another job (no timing claims). Also MNIST-70k / Fashion-70k (3 seeds) and the
10x 1.3M single-cell data (2 seeds), reported, not part of the pass rule.
Predictions:
(N1) lambert_fourier median exact KL is below lambert's on >= 4 of the 5 million-point datasets.
(N2) the median over the 5 datasets of the relative change is <= -1%.
(N3) lambert_fourier median 10-NN is within 1% of lambert's or better on >= 4 of 5.
Kill rule: N1 or N2 fails -> not claimed; the paper then states that the Nesterov optimiser of Lambert et al. is the
stronger optimiser at 1M points and presents Fourier acceleration as an analysis of critical slowing down, not as the
best optimiser.
