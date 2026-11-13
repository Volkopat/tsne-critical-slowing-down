# novel_engines round 16d: data amendment to the Mermin-Wagner protocol (r16c), sealed before any real run
Reason: the cluster is in maintenance and T2I lives only on pod-local storage. The owner allowed short dev runs on the
shared laptop GPU (memory fraction 0.2). Change: data = nested prefixes of SIFT1M train (local ann-benchmarks file),
n = 1e4, 3e4, 1e5, 3e5, instead of T2I prefixes. Everything else (map, sampler, temperatures c = 10 and 100, steps,
tracked points, decision rules P1-P3) is unchanged from r16c. The T2I run stays queued for the cluster as the
confirmation on a second dataset.
