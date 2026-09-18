# CSV-0 certificate probe

Config: 32 frames, 64 tokens, gamma=9, K+text=1024, 3 sample(s): 050-1, 717-1, 496-3

## A. Correctness ceiling (paired sparse vs dense)

- verified positions: **196** across **23** rounds
- position-level sparse/dense agreement: **97.96%**
- rounds where every position agreed (the certificate ceiling): **86.96%**
- rounds where the mask variants disagreed on logits: **0** (max |delta| = 0.000e+00; the cached mask is an optimisation, not an approximation, so this must stay 0)

- strict operating point (precision 1.00): margin >= 0.403 certifies **39.13%** of rounds with **0** wrong skips
- 0.95-precision operating point: margin >= 0.403 certifies **39.13%** of rounds (0 wrong skips)

Selected curve rows (min-margin threshold -> coverage / precision):

| margin >= | certified rounds | coverage | precision | wrong skips |
|---:|---:|---:|---:|---:|
| 0.000 | 23 | 100.0% | 87.0% | 3 |
| 0.103 | 20 | 87.0% | 90.0% | 2 |
| 0.137 | 17 | 73.9% | 94.1% | 1 |
| 0.276 | 14 | 60.9% | 92.9% | 1 |
| 0.367 | 11 | 47.8% | 90.9% | 1 |
| 0.427 | 8 | 34.8% | 100.0% | 0 |
| 0.614 | 5 | 21.7% | 100.0% | 0 |
| 0.794 | 2 | 8.7% | 100.0% | 0 |

## B. Middle-pass cost (ms per round)

| stage | ms/round | note |
|---|---:|---|
| sparse draft (9 sequential passes) | 235.7 | needed only to obtain the block |
| **sparse batched pass (mask rebuilt per layer)** | **77.9** | existing implementation; all-inclusive |
| sparse batched pass (mask built once) | 30.2 | forward only |
| + one mask build per round | 0.4 | 9x1 mask |
| **= mask-cached variant, all-inclusive** | **30.6** | this probe's variant |
| dense pass (what a certificate would skip) | 30.7 | authoritative |
| bonus step, dense share | 24.3 | a skipped round avoids this; the sparse bonus still runs (combined 46.4) |

- mask builds: per-layer variant 644 (4.3 ms of CPU launch time), cached variant 644 hits / 0 misses (0 misses means the cached mask was really reused)
- standalone mask-build microbenchmark: 0.086 ms (CPU launch time, no sync; diagnostic only)

- existing middle pass / dense pass = **2.54x**
- mask-cached middle pass / dense pass = **1.00x**

## C. Gate (from `hsd_feasibility.py`)

- saving when a round skips dense: **55.1 ms/round**
- existing middle pass: IMPOSSIBLE: even a 100% certificate rate is insufficient (needs >141%)
- cached-mask middle pass: NECESSARY: certificates must hold in >56% of rounds

Compare the required rate against the ceiling in section A. A requirement above the ceiling
cannot be met by any certificate, because no rule computable from the sparse pass alone can
certify a round whose sparse and dense argmax differ.

