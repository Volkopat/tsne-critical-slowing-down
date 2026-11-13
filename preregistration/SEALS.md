# Seal records

Each file in this folder is the exact text sealed, before the experiment it concerns ran, into an append-only,
hash-chained ledger kept by the author (each ledger row stores the SHA-256 of the sealed file and the hash of the
previous row). The ledger itself is not public, so the seal times below cannot be independently verified; the
hashes let anyone confirm that these files are byte-for-byte what was sealed.

Verify with `sha256sum *_unaided.md` and compare with the table.

| file | sealed (UTC) | SHA-256 |
|---|---|---|
| `r11_unaided.md` | 2026-11-12T11:29:16+00:00 | `1689bd3b3e43d6230d18c9c63aba74a824ee73c696eb711fc9864b87bfd830b4` |
| `r12_unaided.md` | 2026-11-12T11:41:39+00:00 | `db31a5288711a54a4a6fee4bf45ea89cb68f7252b223e66deb6bbf86b247e4b7` |
| `r14_unaided.md` | 2026-11-12T13:14:14+00:00 | `dcdf46a72a4d87e5ef3c0a1c0d4558df7ff8d79cab119645179bad37621652bf` |
| `r15_unaided.md` | 2026-11-12T13:20:29+00:00 | `a4b87faf9327509cb8d5bd65d15ebf08059484c36bc2bd5483869f9bfa883e7e` |
| `r16b_unaided.md` | 2026-11-12T14:04:52+00:00 | `036062860d6369b8ca3cccb3064a9ea3250e9b4b1908d7c579fddc259a5089f6` |
| `r16c_unaided.md` | 2026-11-12T14:22:31+00:00 | `a4cb5b53db0d20a35437dbbc57f2f8167258e1e2914bf3943213da9bd7bb86ae` |
| `r16d_unaided.md` | 2026-11-12T17:17:43+00:00 | `a0c16a8b3e53da558b4cdea4e203dc07a6ed78546ef9d31fe6f690fe24749eff` |
| `r22_unaided.md` | 2026-11-12T17:44:09+00:00 | `44a25d676e4e942e9021b259fe129e60da866315de3359e53f603f2506304325` |
| `r30_unaided.md` | 2026-11-13T00:02:29+00:00 | `b0ba862d93bf4bb1863b3852d91748c1885b6025cee7bd3a780b79e74f7da34e` |
| `r31_unaided.md` | 2026-11-13T00:35:57+00:00 | `f539e494d5b7c62b4d74ca42a21bf9acef3d24ec006292868487f6ec8d65db47` |

Round 11 is the held-out test (T2I, dbpedia); 12 the first main table, spectrum and scaling pass; 14 the corrected
main table and scaling study; 15 the ablations; 16b to 16d and 22 the Langevin study; 30 the Nesterov + Fourier test;
31 the Nesterov scaling test. Outcomes are in the paper's appendix table.
