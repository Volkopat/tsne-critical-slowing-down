# novel_engines round 11: FA-tSNE held-out 2 against the TUNED optimizer (sealed before any run)

Owner go for new public datasets: 2026-11-12. Data never used in any dev or held-out run:
dbpedia-entities-openai-1M (text, OpenAI ada-002 1536-d, Hugging Face, MIT) and Yandex Text-to-Image
(big-ann-benchmarks images, 200-d), first 1M rows. 2 seeds each, 3000 iterations, scored once.

Frozen engine (from rounds 4-6): FourierPreconditioner grid 256 (rule: 256 for n >= 300k), cap 16, on
after early exaggeration; grid repulsion h = 0.25, max grid 1536; PCA init + seeded jitter 0.1.
Baselines: tuned (lr n after EE + momentum 0.9) and std (FIt-SNE schedule: lr n/12, momentum 0.8).

Predictions:
(a) Fourier's KL@3000 is below tuned's in all 4 runs.
(b) the median relative KL@3000 gap to tuned is >= 2%.
(c) Fourier reaches tuned's KL@3000 within <= 1375 post-EE iterations (half of tuned's 2750) in >= 3 of 4 runs.
(d) Fourier's 10-NN preservation is >= tuned's in >= 3 of 4 runs.
Kill rule: if (a) or (b) fails, the FA-tSNE claim is restricted to the dev datasets and not made in general.
