# novel_engines round 15: ablations of engine v2 (sealed before they run)
Reference: fourier_v2 (cap 16, grid max(128, rule), gains <= 10, EE 250, momentum 0.8) from paper_v2_main.json,
same seeds. SIFT1M and GloVe-100, 2 seeds, 3000 iterations, exact KL.
(E1) cap 4: higher exact KL than cap 16 on both datasets. (E2) cap 64: higher than cap 16 on at least one.
(E3) grid 128 at 1M: higher than grid 256 on both. (E4) grid 512: within 1% of grid 256 on both.
(E5) flat low-pass (same cap, no 1/k^2): higher than the 1/k^2 multiplier on both.
(E6) v2 + momentum 0.9: higher than v2 on both. (E7) EE 100: within 1% of EE 250 on both.
